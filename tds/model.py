"""A tiny transformer in NumPy, forward and backward written out by hand.

Nothing here is novel and it is not meant to be: it is a pre-norm decoder with tied
embeddings, sized so a 300-step run finishes on a CPU in seconds. Its job is to be a real
consumer of the data system -- something that genuinely computes a per-token loss, so the
learning ledger is attached to measured numbers rather than to invented ones.

Three properties matter more than capacity:

  * **Determinism.** Same parameters, same batch, same loss, bit for bit. Initialisation
    comes from an explicitly seeded generator; there is no dropout and no nondeterministic
    reduction. Without this, replay could not prove anything.
  * **Segment-aware attention.** The mask handed in by `masks.py` is the only thing
    deciding what attends to what, so packed documents cannot see each other.
  * **Per-token loss.** `forward_backward` returns the loss at every scored position, not
    just a batch mean, because widgets 11 and 12 attribute learning back to individual
    shards and lanes.

Correctness is checked by finite differences in `tests/test_model.py` rather than asserted
here. Hand-written backprop that has never been gradient-checked is a guess.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hashing import sha256_bytes, tagged

# Parameter order is fixed so that flattening for checkpoints and gradient checks is
# reproducible and does not depend on dict ordering.
SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)


def gelu(x: np.ndarray) -> np.ndarray:
    return gelu_with_cache(x)[0]


def gelu_with_cache(x: np.ndarray):
    """GELU, returning the tanh it computed so the backward pass need not repeat it."""
    inner = SQRT_2_OVER_PI * (x + 0.044715 * x**3)
    tanh_inner = np.tanh(inner)
    return 0.5 * x * (1.0 + tanh_inner), tanh_inner


def gelu_grad(x: np.ndarray, tanh_inner: np.ndarray | None = None) -> np.ndarray:
    if tanh_inner is None:
        tanh_inner = np.tanh(SQRT_2_OVER_PI * (x + 0.044715 * x**3))
    d_inner = SQRT_2_OVER_PI * (1.0 + 3 * 0.044715 * x**2)
    return 0.5 * (1.0 + tanh_inner) + 0.5 * x * (1.0 - tanh_inner**2) * d_inner


def layernorm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    centred = x - mean
    var = (centred**2).mean(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    normed = centred * inv
    return normed * gamma + beta, (normed, inv, gamma)


def layernorm_grad(dout, cache):
    normed, inv, gamma = cache
    d_gamma = (dout * normed).sum(axis=tuple(range(dout.ndim - 1)))
    d_beta = dout.sum(axis=tuple(range(dout.ndim - 1)))
    d_normed = dout * gamma
    d = normed.shape[-1]
    dx = (
        d_normed
        - d_normed.mean(axis=-1, keepdims=True)
        - normed * (d_normed * normed).mean(axis=-1, keepdims=True)
    ) * inv
    return dx, d_gamma, d_beta


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


@dataclass
class ModelConfig:
    vocab_size: int
    context: int
    n_layer: int
    n_head: int
    d_model: int
    d_ff: int
    # float32 for the run; the gradient check switches to float64, because a central
    # difference of 1e-3 on float32 parameters loses the signal entirely -- a true
    # gradient of 1e-5 moves the loss by ~1e-8, which float32 cannot represent against a
    # loss of order 1. A "failing" gradient check at float32 is measuring rounding, not
    # backprop.
    dtype: type = np.float32

    @property
    def head_dim(self) -> int:
        if self.d_model % self.n_head:
            raise ValueError("d_model must be divisible by n_head")
        return self.d_model // self.n_head


class TinyTransformer:
    def __init__(self, config: ModelConfig, seed: int):
        self.config = config
        dtype = config.dtype
        rng = np.random.default_rng(seed)
        d, ff, v, ctx = config.d_model, config.d_ff, config.vocab_size, config.context

        def normal(*shape, scale):
            return (rng.standard_normal(shape) * scale).astype(dtype)

        self.params: dict[str, np.ndarray] = {
            "wte": normal(v, d, scale=0.02),
            "wpe": normal(ctx, d, scale=0.01),
            "ln_f_g": np.ones(d, dtype=dtype),
            "ln_f_b": np.zeros(d, dtype=dtype),
        }
        for layer in range(config.n_layer):
            prefix = f"h{layer}_"
            # Output projections are scaled down by depth, the usual residual-growth fix.
            residual_scale = 0.02 / np.sqrt(2 * config.n_layer)
            self.params.update(
                {
                    prefix + "ln1_g": np.ones(d, dtype=dtype),
                    prefix + "ln1_b": np.zeros(d, dtype=dtype),
                    prefix + "wq": normal(d, d, scale=0.02),
                    prefix + "wk": normal(d, d, scale=0.02),
                    prefix + "wv": normal(d, d, scale=0.02),
                    prefix + "wo": normal(d, d, scale=residual_scale),
                    prefix + "ln2_g": np.ones(d, dtype=dtype),
                    prefix + "ln2_b": np.zeros(d, dtype=dtype),
                    prefix + "w1": normal(d, ff, scale=0.02),
                    prefix + "b1": np.zeros(ff, dtype=dtype),
                    prefix + "w2": normal(ff, d, scale=residual_scale),
                    prefix + "b2": np.zeros(d, dtype=dtype),
                }
            )

    # -- identity ----------------------------------------------------------------

    @property
    def param_names(self) -> list[str]:
        return sorted(self.params)

    @property
    def parameter_count(self) -> int:
        return int(sum(p.size for p in self.params.values()))

    def state_hash(self) -> str:
        payload = b"|".join(
            np.ascontiguousarray(self.params[name], dtype=self.config.dtype).tobytes()
            for name in self.param_names
        )
        return tagged("model", sha256_bytes(payload))

    # -- forward -----------------------------------------------------------------

    def forward(self, input_ids, position_ids, attn_mask):
        """`attn_mask` is (b, T, T) bool: True where a query may attend to a key."""
        cfg = self.config
        p = self.params
        b, t = input_ids.shape
        nh, hd = cfg.n_head, cfg.head_dim

        # A fully masked row would make softmax produce NaN. Padding rows carry no loss,
        # so letting them attend to themselves is harmless and keeps the maths finite.
        mask = attn_mask.copy()
        empty = ~mask.any(axis=-1)
        if empty.any():
            rows = np.arange(t)
            for bi, ti in zip(*np.nonzero(empty)):
                mask[bi, ti, ti] = True

        x = p["wte"][input_ids] + p["wpe"][np.clip(position_ids, 0, cfg.context - 1)]
        cache = {"input_ids": input_ids, "position_ids": position_ids, "mask": mask, "x0": x}

        for layer in range(cfg.n_layer):
            prefix = f"h{layer}_"
            h, ln1_cache = layernorm(x, p[prefix + "ln1_g"], p[prefix + "ln1_b"])
            q = h @ p[prefix + "wq"]
            k = h @ p[prefix + "wk"]
            v = h @ p[prefix + "wv"]

            def split(z):
                return z.reshape(b, t, nh, hd).transpose(0, 2, 1, 3)

            qh, kh, vh = split(q), split(k), split(v)
            scores = (qh @ kh.transpose(0, 1, 3, 2)) / np.sqrt(hd)
            scores = np.where(mask[:, None, :, :], scores, -1e9)
            attn = softmax(scores)
            context = attn @ vh
            merged = context.transpose(0, 2, 1, 3).reshape(b, t, cfg.d_model)
            attn_out = merged @ p[prefix + "wo"]
            x_mid = x + attn_out

            h2, ln2_cache = layernorm(x_mid, p[prefix + "ln2_g"], p[prefix + "ln2_b"])
            pre = h2 @ p[prefix + "w1"] + p[prefix + "b1"]
            act, gelu_tanh = gelu_with_cache(pre)
            ff_out = act @ p[prefix + "w2"] + p[prefix + "b2"]
            x = x_mid + ff_out

            cache[prefix] = {
                "ln1": ln1_cache, "h": h, "qh": qh, "kh": kh, "vh": vh,
                "attn": attn, "merged": merged, "x_in": cache.get(prefix + "x_in", None),
                "x_mid": x_mid, "ln2": ln2_cache, "h2": h2, "pre": pre, "act": act,
                "gelu_tanh": gelu_tanh,
            }
            cache[prefix]["x_in"] = x_mid - attn_out

        hf, ln_f_cache = layernorm(x, p["ln_f_g"], p["ln_f_b"])
        logits = hf @ p["wte"].T
        cache["hf"] = hf
        cache["ln_f"] = ln_f_cache
        cache["x_final"] = x
        return logits, cache

    # -- loss --------------------------------------------------------------------

    @staticmethod
    def token_losses(logits, input_ids, loss_mask):
        """Per-position cross-entropy for next-token prediction.

        `logits[:, i]` predicts `input_ids[:, i + 1]`, so the loss for the token at
        position i is produced at index i - 1. The returned array is aligned with
        `input_ids`: entry i is the loss of predicting token i, and is zero where the mask
        says that position is not a scored target.
        """
        b, t, _ = logits.shape
        predictions = logits[:, :-1, :]
        targets = input_ids[:, 1:]
        mask = loss_mask[:, 1:].astype(logits.dtype)

        shifted = predictions - predictions.max(axis=-1, keepdims=True)
        logsumexp = np.log(np.exp(shifted).sum(axis=-1))
        picked = np.take_along_axis(shifted, targets[:, :, None], axis=-1)[:, :, 0]
        losses = (logsumexp - picked) * mask

        out = np.zeros((b, t), dtype=logits.dtype)
        out[:, 1:] = losses
        return out, mask

    def forward_backward(self, input_ids, position_ids, attn_mask, loss_mask):
        """Returns (mean loss, per-token losses, gradients)."""
        cfg = self.config
        p = self.params
        logits, cache = self.forward(input_ids, position_ids, attn_mask)
        b, t = input_ids.shape

        # The loss and its gradient share the same softmax over the vocabulary. Computing
        # it once and deriving both is the single largest saving in this function: the
        # logit tensor is (b, T, vocab) and dominates everything else here.
        predictions = logits[:, :-1, :]
        targets = input_ids[:, 1:]
        mask = loss_mask[:, 1:].astype(logits.dtype)

        shifted = predictions - predictions.max(axis=-1, keepdims=True)
        exponentiated = np.exp(shifted)
        total = exponentiated.sum(axis=-1, keepdims=True)
        probs = exponentiated / total
        picked = np.take_along_axis(shifted, targets[:, :, None], axis=-1)[:, :, 0]
        losses = (np.log(total[:, :, 0]) - picked) * mask

        per_token = np.zeros((b, t), dtype=logits.dtype)
        per_token[:, 1:] = losses

        scored = float(mask.sum())
        denom = scored if scored > 0 else 1.0
        loss = float(per_token.sum() / denom)

        d_pred = probs
        np.put_along_axis(
            d_pred, targets[:, :, None], np.take_along_axis(d_pred, targets[:, :, None], -1) - 1.0, -1
        )
        d_pred *= (mask / denom)[:, :, None]
        d_logits = np.zeros_like(logits)
        d_logits[:, :-1, :] = d_pred

        grads = {name: np.zeros_like(value) for name, value in p.items()}

        # Tied output projection: logits = hf @ wte.T
        grads["wte"] += d_logits.reshape(-1, cfg.vocab_size).T @ cache["hf"].reshape(-1, cfg.d_model)
        d_hf = d_logits @ p["wte"]

        dx, d_g, d_b = layernorm_grad(d_hf, cache["ln_f"])
        grads["ln_f_g"] += d_g
        grads["ln_f_b"] += d_b

        nh, hd = cfg.n_head, cfg.head_dim
        for layer in reversed(range(cfg.n_layer)):
            prefix = f"h{layer}_"
            c = cache[prefix]

            # --- feed-forward block
            d_ff_out = dx
            grads[prefix + "b2"] += d_ff_out.sum(axis=(0, 1))
            grads[prefix + "w2"] += c["act"].reshape(-1, cfg.d_ff).T @ d_ff_out.reshape(-1, cfg.d_model)
            d_act = d_ff_out @ p[prefix + "w2"].T
            d_pre = d_act * gelu_grad(c["pre"], c["gelu_tanh"])
            grads[prefix + "b1"] += d_pre.sum(axis=(0, 1))
            grads[prefix + "w1"] += c["h2"].reshape(-1, cfg.d_model).T @ d_pre.reshape(-1, cfg.d_ff)
            d_h2 = d_pre @ p[prefix + "w1"].T

            d_ln2, d_g2, d_b2 = layernorm_grad(d_h2, c["ln2"])
            grads[prefix + "ln2_g"] += d_g2
            grads[prefix + "ln2_b"] += d_b2
            d_x_mid = dx + d_ln2

            # --- attention block
            d_attn_out = d_x_mid
            grads[prefix + "wo"] += c["merged"].reshape(-1, cfg.d_model).T @ d_attn_out.reshape(-1, cfg.d_model)
            d_merged = d_attn_out @ p[prefix + "wo"].T
            d_context = d_merged.reshape(b, t, nh, hd).transpose(0, 2, 1, 3)

            d_attn = d_context @ c["vh"].transpose(0, 1, 3, 2)
            d_vh = c["attn"].transpose(0, 1, 3, 2) @ d_context

            # softmax jacobian
            d_scores = c["attn"] * (d_attn - (d_attn * c["attn"]).sum(axis=-1, keepdims=True))
            d_scores = np.where(cache["mask"][:, None, :, :], d_scores, 0.0) / np.sqrt(hd)

            d_qh = d_scores @ c["kh"]
            d_kh = d_scores.transpose(0, 1, 3, 2) @ c["qh"]

            def merge(z):
                return z.transpose(0, 2, 1, 3).reshape(b, t, cfg.d_model)

            d_q, d_k, d_v = merge(d_qh), merge(d_kh), merge(d_vh)
            h_flat = c["h"].reshape(-1, cfg.d_model)
            grads[prefix + "wq"] += h_flat.T @ d_q.reshape(-1, cfg.d_model)
            grads[prefix + "wk"] += h_flat.T @ d_k.reshape(-1, cfg.d_model)
            grads[prefix + "wv"] += h_flat.T @ d_v.reshape(-1, cfg.d_model)
            d_h = (
                d_q @ p[prefix + "wq"].T
                + d_k @ p[prefix + "wk"].T
                + d_v @ p[prefix + "wv"].T
            )

            d_ln1, d_g1, d_b1 = layernorm_grad(d_h, c["ln1"])
            grads[prefix + "ln1_g"] += d_g1
            grads[prefix + "ln1_b"] += d_b1
            dx = d_x_mid + d_ln1

        # --- embeddings
        np.add.at(grads["wte"], cache["input_ids"], dx)
        np.add.at(grads["wpe"], np.clip(cache["position_ids"], 0, cfg.context - 1), dx)

        return loss, per_token, grads


class Adam:
    """Adam with decoupled gradient clipping. Deterministic; no state outside these arrays."""

    def __init__(self, params: dict[str, np.ndarray], lr: float, betas=(0.9, 0.95), eps=1e-8):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = {name: np.zeros_like(value) for name, value in params.items()}
        self.v = {name: np.zeros_like(value) for name, value in params.items()}

    def step(self, params, grads, lr_scale: float = 1.0, clip: float | None = None):
        if clip is not None:
            norm = np.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads.values()))
            if norm > clip:
                scale = clip / (norm + 1e-12)
                for name in grads:
                    grads[name] = grads[name] * scale
        else:
            norm = np.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads.values()))

        self.t += 1
        lr = self.lr * lr_scale
        bias1 = 1 - self.b1**self.t
        bias2 = 1 - self.b2**self.t
        for name in sorted(params):
            g = grads[name]
            self.m[name] = self.b1 * self.m[name] + (1 - self.b1) * g
            self.v[name] = self.b2 * self.v[name] + (1 - self.b2) * (g * g)
            m_hat = self.m[name] / bias1
            v_hat = self.v[name] / bias2
            params[name] -= (lr * m_hat / (np.sqrt(v_hat) + self.eps)).astype(params[name].dtype)
        return float(norm)

    def state(self) -> dict:
        state = {"__t": np.asarray([self.t], dtype=np.int64)}
        for name in sorted(self.m):
            state[f"m::{name}"] = self.m[name]
            state[f"v::{name}"] = self.v[name]
        return state

    def load_state(self, state: dict) -> None:
        self.t = int(state["__t"][0])
        for key, value in state.items():
            if key.startswith("m::"):
                self.m[key[3:]] = np.asarray(value)
            elif key.startswith("v::"):
                self.v[key[3:]] = np.asarray(value)


def build_model(config, vocab_size: int) -> tuple[TinyTransformer, ModelConfig]:
    model_config = ModelConfig(
        vocab_size=vocab_size,
        context=config.require("batch.sequence_length"),
        n_layer=config.require("model.n_layer"),
        n_head=config.require("model.n_head"),
        d_model=config.require("model.d_model"),
        d_ff=config.require("model.d_ff"),
    )
    return TinyTransformer(model_config, seed=config.require("run.seed")), model_config

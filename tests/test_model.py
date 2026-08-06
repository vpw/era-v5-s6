"""Gradient check.

Hand-written backprop that has never been checked against finite differences is a guess,
and every loss number in the learning ledger depends on it being right.

Two details make this check meaningful rather than decorative:

* it runs in **float64**. At float32, a true gradient of 1e-6 moves the loss by ~1e-12,
  which the format cannot represent against a loss of order 1 -- the check would be
  measuring rounding, not correctness.
* it uses the standard **absolute-or-relative** criterion. The finite-difference roundoff
  floor at eps=1e-6 in float64 is ~1e-10 absolute, so parameters with genuinely tiny
  gradients can only be checked absolutely; demanding a small *relative* error there would
  fail correct code.
"""

from __future__ import annotations

import numpy as np
import pytest

from tds.masks import attention_mask
from tds.model import Adam, ModelConfig, TinyTransformer

EPS = 1e-6
ATOL = 1e-8
RTOL = 1e-6


def _tiny_batch():
    rng = np.random.default_rng(0)
    b, t, vocab = 2, 12, 37
    input_ids = rng.integers(0, vocab, size=(b, t)).astype(np.int32)

    # Two packed documents in row 0, one document plus padding in row 1 -- so the check
    # covers segmented attention and padding, not just the easy case.
    segment_ids = np.zeros((b, t), dtype=np.int32)
    segment_ids[0, 7:] = 1
    segment_ids[1, 9:] = -1

    position_ids = np.zeros((b, t), dtype=np.int32)
    for row in range(b):
        start = 0
        for i in range(t):
            if i > 0 and segment_ids[row, i] != segment_ids[row, i - 1]:
                start = i
            position_ids[row, i] = i - start

    loss_mask = (segment_ids >= 0).astype(np.uint8)
    loss_mask[position_ids == 0] = 0
    masks = np.stack([attention_mask(segment_ids[i]) for i in range(b)])
    return input_ids, position_ids, masks, loss_mask, vocab


def test_gradients_match_finite_differences():
    input_ids, position_ids, masks, loss_mask, vocab = _tiny_batch()
    config = ModelConfig(vocab_size=vocab, context=12, n_layer=2, n_head=2,
                         d_model=8, d_ff=16, dtype=np.float64)
    model = TinyTransformer(config, seed=7)

    _, _, grads = model.forward_backward(input_ids, position_ids, masks, loss_mask)

    rng = np.random.default_rng(0)
    failures = []
    checked = 0
    for name in model.param_names:
        tensor = model.params[name]
        for _ in range(4):
            index = tuple(int(rng.integers(0, size)) for size in tensor.shape)
            original = float(tensor[index])

            tensor[index] = original + EPS
            plus, _, _ = model.forward_backward(input_ids, position_ids, masks, loss_mask)
            tensor[index] = original - EPS
            minus, _, _ = model.forward_backward(input_ids, position_ids, masks, loss_mask)
            tensor[index] = original

            numerical = (plus - minus) / (2 * EPS)
            analytic = float(grads[name][index])
            absolute = abs(numerical - analytic)
            relative = absolute / max(abs(numerical) + abs(analytic), 1e-30)
            checked += 1
            if not (absolute < ATOL or relative < RTOL):
                failures.append((name, index, numerical, analytic, absolute, relative))

    assert checked == 4 * len(model.param_names)
    assert not failures, f"gradient mismatches: {failures[:4]}"


def test_every_parameter_receives_gradient():
    """A parameter that never gets a gradient is dead weight, or a wiring bug."""
    input_ids, position_ids, masks, loss_mask, vocab = _tiny_batch()
    config = ModelConfig(vocab_size=vocab, context=12, n_layer=2, n_head=2,
                         d_model=8, d_ff=16, dtype=np.float64)
    model = TinyTransformer(config, seed=7)
    _, _, grads = model.forward_backward(input_ids, position_ids, masks, loss_mask)

    dead = [name for name in model.param_names if not np.any(grads[name])]
    assert dead == [], f"parameters with all-zero gradients: {dead}"


def test_forward_is_deterministic():
    """Replay depends on this: identical inputs must give bit-identical outputs."""
    input_ids, position_ids, masks, loss_mask, vocab = _tiny_batch()
    config = ModelConfig(vocab_size=vocab, context=12, n_layer=2, n_head=2, d_model=8, d_ff=16)
    a = TinyTransformer(config, seed=7)
    b = TinyTransformer(config, seed=7)

    assert a.state_hash() == b.state_hash()
    loss_a, per_a, _ = a.forward_backward(input_ids, position_ids, masks, loss_mask)
    loss_b, per_b, _ = b.forward_backward(input_ids, position_ids, masks, loss_mask)
    assert loss_a == loss_b
    np.testing.assert_array_equal(per_a, per_b)


def test_padding_and_unscored_positions_carry_no_loss():
    input_ids, position_ids, masks, loss_mask, vocab = _tiny_batch()
    config = ModelConfig(vocab_size=vocab, context=12, n_layer=2, n_head=2, d_model=8, d_ff=16)
    model = TinyTransformer(config, seed=7)
    _, per_token, _ = model.forward_backward(input_ids, position_ids, masks, loss_mask)

    assert np.all(per_token[loss_mask == 0] == 0.0)
    assert np.any(per_token[loss_mask == 1] > 0.0)


def test_optimizer_state_round_trips():
    config = ModelConfig(vocab_size=17, context=4, n_layer=1, n_head=1, d_model=4, d_ff=8)
    model = TinyTransformer(config, seed=3)
    optimizer = Adam(model.params, lr=0.01)
    grads = {name: np.ones_like(v) for name, v in model.params.items()}
    optimizer.step(model.params, grads, 1.0, 1.0)

    restored = Adam(model.params, lr=0.01)
    restored.load_state(optimizer.state())
    assert restored.t == optimizer.t
    for name in optimizer.m:
        np.testing.assert_array_equal(restored.m[name], optimizer.m[name])
        np.testing.assert_array_equal(restored.v[name], optimizer.v[name])

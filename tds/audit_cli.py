"""Run the audit on its own: `python -m tds.audit_cli`.

Useful for checking a submission_artifacts/ tree without re-running the demo. It reads
only what is on disk, so it can be pointed at artifacts produced by a different machine.
"""

from __future__ import annotations

import json
import sys

from .audit import run_audit, write
from .config import Config


def main() -> int:
    config = Config.load(sys.argv[1] if len(sys.argv) > 1 else None)
    report = run_audit(config, log=lambda message: print(message))
    path = write(config.artifacts_dir / "reports" / "audit_report.json", report)
    for check in report["checks"]:
        print(f"{check['result']:5s} {check['check']:26s} {check['title']}")
    print(f"\n{report['checks_passed']}/{report['checks_total']} checks passed -> {path}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

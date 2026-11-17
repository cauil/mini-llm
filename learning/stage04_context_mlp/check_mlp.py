"""单独运行核心正确性检查，不训练或覆盖正式结果。"""

import json
import torch
from .checks import run_checks


def main():
    torch.set_num_threads(1)
    report = run_checks().result()
    print(json.dumps({key: report[key] for key in ("count", "passed", "failed")}, indent=2))
    if not report["all_passed"]:
        failures = [item for item in report["checks"] if not item["passed"]]
        raise RuntimeError(json.dumps(failures, ensure_ascii=False))


if __name__ == "__main__":
    main()

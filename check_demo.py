"""check_demo.py - proves the demo gives the same decisions as the evaluation.

Run from the folder that has demo.py, project.py and evaluate.py:

    python check_demo.py

For every window (1-4 s, all participants, all targets) it compares
  A) the demo pipeline  : project.py (sklearn CCA) + Welch from demo.py
  B) the evaluation     : evaluate.py (exact CCA + Welch)
and prints the accuracies and how many decisions differ (should be 0).
"""
import sys
import numpy as np

import demo
import evaluate as ev

ev.REFERENCE_MODE = "sincos"


def main():
    if len(sys.argv) > 1:
        demo.DATA_DIR = sys.argv[1]
    fs = demo.FS
    preprocessing_ok = True
    total_diff = 0
    print(f"{'Win':>4} {'N':>5} | {'Welch':>7} {'CCA':>7} (demo) | {'Welch':>7} {'CCA':>7} (evaluate) "
          f"| {'differ':>6} | {'only Welch right':>16} {'only CCA right':>15}")
    for L in (1, 2, 3, 4):
        n = L * fs
        dw = dc = ew = ec = 0
        only_w = only_c = tot = diff = 0
        for s in range(1, demo.N_SUBJECTS + 1):
            for tgt in range(4):
                rec = demo.load_recording(s, tgt)
                raw = demo._raw_cache[s][:, tgt]
                raw = raw[~np.isnan(raw)]
                if not np.allclose(rec, ev.preprocess(raw)):
                    preprocessing_ok = False
                for k in range(len(rec) // n):
                    w = rec[k * n:(k + 1) * n]
                    _, _, wi = demo.welch_analysis(w)
                    _, ci = demo.cca_analysis(w)
                    ewi = int(np.argmax(ev.welch_scores(w)))
                    eci = int(np.argmax(ev.cca_scores(w)))
                    tot += 1
                    dw += wi == tgt
                    dc += ci == tgt
                    ew += ewi == tgt
                    ec += eci == tgt
                    diff += (wi != ewi) + (ci != eci)
                    only_w += (wi == tgt) and (ci != tgt)
                    only_c += (ci == tgt) and (wi != tgt)
        total_diff += diff
        print(f"{L:>3}s {tot:>5} | {100 * dw / tot:>6.1f}% {100 * dc / tot:>6.1f}%        | "
              f"{100 * ew / tot:>6.1f}% {100 * ec / tot:>6.1f}%          | {diff:>6} | "
              f"{100 * only_w / tot:>15.1f}% {100 * only_c / tot:>14.1f}%")
    print()
    print("Preprocessing identical to evaluate.py:", "YES" if preprocessing_ok else "NO  <-- problem")
    print("Decisions that differ between demo and evaluation:", total_diff)
    print("RESULT:", "PASS - the demo matches the evaluation" if (preprocessing_ok and total_diff == 0)
          else "FAIL - send me this output")
    print("\nFor comparison, evaluate.py reported at 4 s: Welch 56.8%, CCA 65.3%.")


if __name__ == "__main__":
    main()

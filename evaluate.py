"""evaluate.py - Welch (PSD) vs CCA comparison for the single-channel SSVEP project.

What it does
------------
1. Loads subject1.csv ... subject11.csv (4 columns F1..F4 = 4 stimulus frequencies).
2. Preprocesses each recording exactly like project.py (detrend + 4th-order
   Butterworth 7.5-30 Hz, zero-phase).
3. Splits each recording into non-overlapping windows of 1, 2, 3, 4 s.
4. Runs BOTH methods on the SAME windows: Welch peak-power and CCA max-correlation.
5. Reports accuracy, ITR, per-participant accuracy, confusion matrices and
   statistical tests, and saves figures (PNG) + tables (CSV).

Run
---
    pip install matplotlib
    python evaluate.py
or with custom folders:
    python evaluate.py --data-dir F:\\Python\\data --out-dir F:\\Python\\Project\\results
"""

import argparse
import csv
import os

import numpy as np
from scipy import signal, stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Settings (change here only)
# ---------------------------------------------------------------------------
FS = 256                                   # Hz (confirm in the dataset paper)
FREQS = np.array([8.57, 10.0, 12.0, 15.0])  # F1, F2, F3, F4 in this order
N_SUBJECTS = 11
WINDOWS_S = [1, 2, 3, 4]                   # analysis window lengths (s)
FOCUS_WINDOW_S = 4                         # window used for confusion matrices / per-participant bars
REFERENCE_MODE = "sincos"                  # "sincos" = sin+cos with harmonics (research plan)
                                           # "sine"   = sine only (old project.py)
N_HARMONICS = 2                            # used when REFERENCE_MODE == "sincos"

COLOR_CCA = "#1B5E7B"
COLOR_WELCH = "#E08A1E"
N_TARGETS = len(FREQS)


# ---------------------------------------------------------------------------
# Loading and preprocessing
# ---------------------------------------------------------------------------
def load_subject(data_dir, s):
    path = os.path.join(data_dir, f"subject{s}.csv")
    data = np.genfromtxt(path, delimiter=";", skip_header=1)
    cols = []
    for c in range(N_TARGETS):
        col = data[:, c]
        cols.append(col[~np.isnan(col)])
    return cols


def preprocess(x):
    """Same pipeline as project.py: linear detrend + Butterworth 4th order, zero-phase."""
    x = signal.detrend(x, type="linear")
    sos = signal.butter(4, (7.5, 30.0), btype="bandpass", output="sos", fs=FS)
    return signal.sosfiltfilt(sos, x, padtype="odd")


# ---------------------------------------------------------------------------
# Method 1: Welch baseline
# ---------------------------------------------------------------------------
def welch_scores(w):
    """Power at the bin nearest to each stimulus frequency (same rule as project.py).

    nperseg is capped at the window length (1024 does not fit in a 1-3 s window);
    nfft stays 1024 so the bin spacing is the same for all window lengths.
    """
    nper = min(1024, len(w))
    f, p = signal.welch(w, fs=FS, window="hann", nperseg=nper, noverlap=nper // 2,
                        nfft=1024, detrend="constant", scaling="spectrum")
    return np.array([p[np.argmin(np.abs(f - t))] for t in FREQS])


# ---------------------------------------------------------------------------
# Method 2: CCA
# ---------------------------------------------------------------------------
def _reference(freq, n):
    t = np.arange(n) / FS
    cols = []
    if REFERENCE_MODE == "sine":
        cols.append(np.sin(2 * np.pi * freq * t))
    else:
        for h in range(1, N_HARMONICS + 1):
            cols.append(np.sin(2 * np.pi * h * freq * t))
            cols.append(np.cos(2 * np.pi * h * freq * t))
    return np.column_stack(cols)


_Q_CACHE = {}


def _basis(freq, n):
    key = (float(freq), n)
    if key not in _Q_CACHE:
        r = _reference(freq, n)
        r = r - r.mean(axis=0)
        q, _ = np.linalg.qr(r)
        _Q_CACHE[key] = q
    return _Q_CACHE[key]


def cca_score(w, freq):
    """First canonical correlation between one EEG channel and the references of `freq`.

    With a single EEG channel this equals the multiple correlation
    ||Q^T y|| / ||y|| (Q = orthonormal basis of the references), which is
    identical to what sklearn's CCA(n_components=1) returns, but exact and fast.
    """
    y = w - w.mean()
    q = _basis(freq, len(w))
    return float(np.linalg.norm(q.T @ y) / np.linalg.norm(y))


def cca_scores(w):
    return np.array([cca_score(w, f) for f in FREQS])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def itr_bits_per_min(p, window_s, n=N_TARGETS):
    """Wolpaw ITR. T = window length only (no gaze-shift time)."""
    if p <= 1.0 / n:
        return 0.0
    bits = np.log2(n)
    if p < 1.0:
        bits += p * np.log2(p) + (1 - p) * np.log2((1 - p) / (n - 1))
    return bits * 60.0 / window_s


def mean_ci95(values):
    v = np.asarray(values, dtype=float)
    m = v.mean()
    if len(v) < 2:
        return m, 0.0, 0.0
    sd = v.std(ddof=1)
    half = stats.t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v))
    return m, sd, half


def exact_mcnemar(correct_a, correct_b):
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(stats.binomtest(min(b, c), b + c, 0.5).pvalue)


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=r"F:\Python\data")
    ap.add_argument("--out-dir", default=r"F:\Python\Project\results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Calibri", "DejaVu Sans"],
        "font.size": 18,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    print(f"Reference mode: {REFERENCE_MODE}"
          + (f" ({N_HARMONICS} harmonics)" if REFERENCE_MODE == "sincos" else ""))

    # ---- load + preprocess ------------------------------------------------
    recordings = {}   # (subject, col) -> filtered signal
    for s in range(1, N_SUBJECTS + 1):
        cols = load_subject(args.data_dir, s)
        for c, x in enumerate(cols):
            if len(x) != 16 * FS:
                print(f"  WARNING: subject{s} F{c + 1} has {len(x)} samples (expected {16 * FS})")
            recordings[(s, c)] = preprocess(x)

    # ---- sanity check: does column order match the frequency order? -------
    check = np.zeros((N_TARGETS, N_TARGETS), dtype=int)
    for (s, c), x in recordings.items():
        check[c, int(np.argmax(welch_scores(x)))] += 1
    print("\nColumn check (Welch on the full 16 s recording; rows = column F1..F4, "
          "cols = detected frequency):")
    print("          " + "  ".join(f"{f:>6}" for f in FREQS))
    for c in range(N_TARGETS):
        print(f"  F{c + 1}      " + "  ".join(f"{v:>6}" for v in check[c]))
    print("  -> a strong diagonal means the column order is correct.\n")

    # ---- windows + both methods on the SAME windows -----------------------
    subj, win_s, true_idx, welch_pred, cca_pred = [], [], [], [], []
    feat_rows = []
    for (s, c), x in recordings.items():
        for L in WINDOWS_S:
            n = L * FS
            for k in range(len(x) // n):
                w = x[k * n:(k + 1) * n]
                ws = welch_scores(w)
                cs = cca_scores(w)
                subj.append(s)
                win_s.append(L)
                true_idx.append(c)
                welch_pred.append(int(np.argmax(ws)))
                cca_pred.append(int(np.argmax(cs)))
                feat_rows.append([s, L, FREQS[c], k] + [f"{v:.6g}" for v in ws]
                                 + [f"{v:.6g}" for v in cs])
    subj = np.array(subj)
    win_s = np.array(win_s)
    true_idx = np.array(true_idx)
    welch_pred = np.array(welch_pred)
    cca_pred = np.array(cca_pred)
    welch_ok = welch_pred == true_idx
    cca_ok = cca_pred == true_idx

    # saved so the ML step can reuse the same features later
    write_csv(os.path.join(args.out_dir, "window_features.csv"),
              ["subject", "window_s", "true_freq_hz", "window_idx"]
              + [f"welch_{f}" for f in FREQS] + [f"cca_{f}" for f in FREQS], feat_rows)

    methods = {"Welch": welch_ok, "CCA": cca_ok}

    # ---- per-participant accuracy + ITR -----------------------------------
    per_part_rows = []
    acc = {m: {L: [] for L in WINDOWS_S} for m in methods}
    itr = {m: {L: [] for L in WINDOWS_S} for m in methods}
    for m, ok in methods.items():
        for L in WINDOWS_S:
            for s in range(1, N_SUBJECTS + 1):
                sel = (win_s == L) & (subj == s)
                a = float(ok[sel].mean())
                acc[m][L].append(a)
                itr[m][L].append(itr_bits_per_min(a, L))
                per_part_rows.append([m, L, s, f"{100 * a:.2f}", f"{itr[m][L][-1]:.2f}"])
    write_csv(os.path.join(args.out_dir, "per_participant.csv"),
              ["method", "window_s", "subject", "accuracy_pct", "itr_bits_per_min"],
              per_part_rows)

    # ---- summary table -----------------------------------------------------
    summary_rows = []
    print("Accuracy and ITR (mean +/- SD across the 11 participants)")
    print(f"{'Window':>7} {'Method':>6} {'Accuracy %':>16} {'95% CI':>8} {'ITR bits/min':>16}")
    for L in WINDOWS_S:
        for m in methods:
            am, asd, ah = mean_ci95(acc[m][L])
            im, isd, _ = mean_ci95(itr[m][L])
            summary_rows.append([L, m, f"{100 * am:.2f}", f"{100 * asd:.2f}",
                                 f"{100 * ah:.2f}", f"{im:.2f}", f"{isd:.2f}"])
            print(f"{L:>6}s {m:>6} {100 * am:>9.1f} +/- {100 * asd:<4.1f} "
                  f"+/-{100 * ah:<5.1f} {im:>9.1f} +/- {isd:<4.1f}")
    write_csv(os.path.join(args.out_dir, "summary.csv"),
              ["window_s", "method", "accuracy_mean_pct", "accuracy_sd_pct",
               "accuracy_ci95_halfwidth_pct", "itr_mean", "itr_sd"], summary_rows)

    # ---- statistics ---------------------------------------------------------
    stat_rows = []
    print("\nCCA vs Welch (same windows)")
    print(f"{'Window':>7} {'diff (CCA-Welch) %':>20} {'Wilcoxon p (n=11)':>19} "
          f"{'McNemar b/c':>13} {'McNemar p':>10}")
    for L in WINDOWS_S:
        a_cca = np.array(acc["CCA"][L])
        a_wel = np.array(acc["Welch"][L])
        diff = 100 * float(np.mean(a_cca - a_wel))
        if np.allclose(a_cca, a_wel):
            p_w = 1.0          # identical accuracies for every participant
        else:
            try:
                p_w = float(stats.wilcoxon(a_cca, a_wel).pvalue)
            except ValueError:
                p_w = float("nan")
        sel = win_s == L
        b, c, p_m = exact_mcnemar(welch_ok[sel], cca_ok[sel])
        stat_rows.append([L, f"{diff:.2f}", f"{p_w:.4g}", b, c, f"{p_m:.4g}"])
        print(f"{L:>6}s {diff:>20.1f} {p_w:>19.4g} {str(b) + '/' + str(c):>13} {p_m:>10.4g}")
    write_csv(os.path.join(args.out_dir, "stats.csv"),
              ["window_s", "mean_diff_cca_minus_welch_pct", "wilcoxon_p_participants",
               "mcnemar_b_welch_only_correct", "mcnemar_c_cca_only_correct",
               "mcnemar_exact_p_pooled_windows"], stat_rows)
    print("  Wilcoxon (per-participant accuracies) is the main test. McNemar is on pooled windows,\n"
          "  and windows from the same recording are correlated, so treat its p-value as optimistic.")

    # ---- Figure 1: accuracy vs window --------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    for m, col in (("CCA", COLOR_CCA), ("Welch", COLOR_WELCH)):
        means = [100 * np.mean(acc[m][L]) for L in WINDOWS_S]
        sds = [100 * np.std(acc[m][L], ddof=1) for L in WINDOWS_S]
        ax.errorbar(WINDOWS_S, means, yerr=sds, marker="o", markersize=9, linewidth=3,
                    capsize=6, color=col, label=m)
    ax.axhline(100 / N_TARGETS, linestyle="--", color="gray", linewidth=2)
    ax.text(WINDOWS_S[0], 100 / N_TARGETS + 2, "chance (25%)", color="gray", fontsize=15)
    ax.set_xticks(WINDOWS_S)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Window length (s)")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(os.path.join(args.out_dir, "accuracy_vs_window.png"), dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- Figure 2: ITR vs window -------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    for m, col in (("CCA", COLOR_CCA), ("Welch", COLOR_WELCH)):
        means = [np.mean(itr[m][L]) for L in WINDOWS_S]
        sds = [np.std(itr[m][L], ddof=1) for L in WINDOWS_S]
        ax.errorbar(WINDOWS_S, means, yerr=sds, marker="o", markersize=9, linewidth=3,
                    capsize=6, color=col, label=m)
    ax.set_xticks(WINDOWS_S)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Window length (s)")
    ax.set_ylabel("ITR (bits/min)")
    ax.legend(loc="upper right", frameon=False)
    fig.savefig(os.path.join(args.out_dir, "itr_vs_window.png"), dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- Figure 3: confusion matrices at the focus window -------------------
    labels = [f"{f:g}" for f in FREQS]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (m, pred) in zip(axes, (("Welch", welch_pred), ("CCA", cca_pred))):
        sel = win_s == FOCUS_WINDOW_S
        cm = np.zeros((N_TARGETS, N_TARGETS))
        for t, p in zip(true_idx[sel], pred[sel]):
            cm[t, p] += 1
        cm_pct = 100 * cm / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(N_TARGETS):
            for j in range(N_TARGETS):
                ax.text(j, i, f"{cm_pct[i, j]:.0f}", ha="center", va="center", fontsize=18,
                        color="white" if cm_pct[i, j] > 55 else "black")
        ax.set_xticks(range(N_TARGETS))
        ax.set_yticks(range(N_TARGETS))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Detected frequency (Hz)")
        ax.set_ylabel("True frequency (Hz)")
        ax.set_title(f"{m} ({FOCUS_WINDOW_S} s window, % of trials)", fontsize=17)
        for sp in ax.spines.values():
            sp.set_visible(True)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "confusion_matrices.png"), dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- Figure 4: per-participant accuracy at the focus window -------------
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(N_SUBJECTS)
    wbar = 0.38
    ax.bar(x - wbar / 2, [100 * a for a in acc["Welch"][FOCUS_WINDOW_S]], wbar,
           color=COLOR_WELCH, label="Welch")
    ax.bar(x + wbar / 2, [100 * a for a in acc["CCA"][FOCUS_WINDOW_S]], wbar,
           color=COLOR_CCA, label="CCA")
    ax.axhline(100 / N_TARGETS, linestyle="--", color="gray", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{s}" for s in range(1, N_SUBJECTS + 1)])
    ax.set_ylim(0, 105)
    ax.set_xlabel(f"Participant ({FOCUS_WINDOW_S} s window)")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=2)
    fig.savefig(os.path.join(args.out_dir, "per_participant_accuracy.png"), dpi=300,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    n_win = {L: int(np.sum(win_s == L)) for L in WINDOWS_S}
    print(f"\nWindows analysed per window length: {n_win}")
    print(f"Saved figures and tables in: {args.out_dir}")


if __name__ == "__main__":
    main()
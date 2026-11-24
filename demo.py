"""demo.py - NeuroVocal live demo (offline replay of the public SSVEP dataset).

Put this file in the same folder as project.py and run:

    python demo.py

What the audience sees
----------------------
1. Choose a participant, the target they were looking at, and a window length.
2. Press "Analyze" (or Space/Enter): the recorded EEG window streams in real time.
3. When the window is complete, the system decides with BOTH methods
   (Welch spectrum and CCA) and the CCA result is spoken aloud (offline Piper TTS).
4. The session tally shows how often each method was right, including the misses.

Stimulus modes (STIMULUS box)
-----------------------------
* Static (off)     - the target cards do not flicker (safe default).
* Flicker all      - every card has a stimulus square flickering at its own frequency
                     (8.57 / 10 / 12 / 15 Hz) while the EEG window is recorded, exactly
                     like a real SSVEP screen.
* Flicker target   - only the card the participant is looking at flickers (teaching mode).
WAVEFORM box: Square (on/off, 50 % duty cycle) or Sinusoidal (smooth brightness).

The EEG is a replay of recordings from Acampora et al. (2021), not live brain activity,
so the flicker you see is a visual illustration of the stimulus that evoked the recorded
response. WARNING: flicker modes may trigger seizures in photosensitive people, which is
why the demo starts in Static mode.
"""

import math
import os
import random
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
from scipy import signal
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import project  # your final project.py (same folder)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DATA_DIR = r"F:\Python\data"                                  # subject1.csv ... subject11.csv
MODEL_PATH = r"F:\Python\Project\en_US-lessac-medium.onnx"    # Piper voice (same as project.py)
N_SUBJECTS = 11
WINDOW_CHOICES = [1, 2, 3, 4]
DEFAULT_WINDOW_S = 4
VOICE_VOLUME = 0.8        # project.py uses 0.5; a bit louder for a noisy hall

FS = project.SAMPLING_RATE
FREQS = project.STIMULUS_FREQUENCIES
PHRASES = project.FREQUENCY_TO_COMMAND

FONT = "Segoe UI"          # falls back automatically if the font is missing

INK = "#14213D"
MUTED = "#6B7A90"
BG = "#EEF2F7"
CARD = "#FFFFFF"
LINE = "#DDE3EA"
TEAL = "#1B5E7B"
TEAL_LIGHT = "#E3F0F5"
ORANGE = "#E08A1E"
GREEN = "#1E8E4E"
GREEN_BG = "#E4F5EA"
RED = "#C8372D"
RED_BG = "#FBE7E5"

# visual states of a target card
CARD_STATES = {
    "idle":    dict(bg=CARD,       strip=LINE,  tag=MUTED),
    "looking": dict(bg=TEAL_LIGHT, strip=TEAL,  tag=TEAL),
    "correct": dict(bg=GREEN_BG,   strip=GREEN, tag=GREEN),
    "wrong":   dict(bg=RED_BG,     strip=RED,   tag=RED),
}

# flicker stimulus
STIM_STATIC, STIM_ALL, STIM_TARGET = "Static (off)", "Flicker all", "Flicker target"
STIM_CHOICES = [STIM_STATIC, STIM_ALL, STIM_TARGET]
WAVE_SQUARE, WAVE_SINE = "Square", "Sinusoidal"
WAVE_CHOICES = [WAVE_SQUARE, WAVE_SINE]
PATCH_IDLE = "#C9D2DD"     # resting colour of a stimulus square
PATCH_SIZE = 72
FLICKER_TICK_MS = 4

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT, "Arial", "DejaVu Sans"],
    "axes.edgecolor": LINE,
    "axes.labelcolor": MUTED,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": CARD,
    "axes.facecolor": CARD,
})


# ---------------------------------------------------------------------------
# Data + analysis (no GUI code here)
# ---------------------------------------------------------------------------
_raw_cache = {}
_filtered_cache = {}


def load_recording(subject, target_idx):
    """Filtered 16 s recording of one participant looking at one target.

    Filtering uses project.filter_signal on the whole recording first and the
    windows are cut afterwards, exactly like the evaluation (evaluate.py).
    """
    key = (subject, target_idx)
    if key not in _filtered_cache:
        if subject not in _raw_cache:
            path = os.path.join(DATA_DIR, f"subject{subject}.csv")
            _raw_cache[subject] = np.genfromtxt(path, delimiter=";", skip_header=1)
        col = _raw_cache[subject][:, target_idx]
        col = col[~np.isnan(col)]
        _filtered_cache[key] = project.filter_signal(col)
    return _filtered_cache[key]


def welch_analysis(w):
    """Welch baseline (same settings as evaluate.py). Returns f, p, index of winner."""
    nper = min(1024, len(w))
    f, p = signal.welch(w, fs=FS, window="hann", nperseg=nper, noverlap=nper // 2,
                        nfft=1024, detrend="constant", scaling="spectrum")
    powers = np.array([p[np.argmin(np.abs(f - t))] for t in FREQS])
    return f, p, int(np.argmax(powers))


def cca_analysis(w):
    """CCA through project.py. Returns the 4 correlations and the index of the winner."""
    waves = project.hertz_to_waves(FREQS, len(w))
    x_c, y_c = project.cca(waves, w)
    corr = np.array([abs(np.corrcoef(x.ravel(), y.ravel())[0, 1])
                     for x, y in zip(x_c, y_c)])
    detected = project.detect_frequency(x_c, y_c)          # the decision of project.py
    idx = int(np.argmin(np.abs(FREQS - detected)))
    assert idx == int(np.argmax(corr)), "display values disagree with project.detect_frequency"
    return corr, idx


def stimulus_gray(freq, t, waveform):
    """Brightness (0..255) of a flickering square at time t (s) for a stimulus at freq (Hz).

    Square: fully on for the first half of every cycle, fully off for the second half.
    Sinusoidal: brightness follows a raised sine between 0 and 255.
    """
    phase = (t * freq) % 1.0
    if waveform == WAVE_SINE:
        lum = 0.5 * (1.0 - math.cos(2.0 * math.pi * phase))
    else:
        lum = 1.0 if phase < 0.5 else 0.0
    return int(round(lum * 255))


# ---------------------------------------------------------------------------
# Offline speech (Piper), loaded once and played in a background thread
# ---------------------------------------------------------------------------
class Speaker:
    def __init__(self):
        self.voice = None
        self.cfg = None
        self.state = "Voice: loading..."
        self._lock = threading.Lock()

    def _load(self):
        if self.voice is None:
            self.voice = project.PiperVoice.load(MODEL_PATH)
            self.cfg = project.SynthesisConfig(
                volume=VOICE_VOLUME, length_scale=1.5,
                noise_scale=1.0, noise_w_scale=1.0, normalize_audio=True)

    def preload(self):
        with self._lock:
            try:
                self._load()
                self.state = "Voice: ready (offline)"
            except Exception as exc:                        # demo keeps working without sound
                self.state = f"Voice unavailable: {exc}"

    def say(self, text):
        threading.Thread(target=self._speak, args=(text,), daemon=True).start()

    def _speak(self, text):
        with self._lock:
            try:
                self._load()
                for chunk in self.voice.synthesize(text, syn_config=self.cfg):
                    audio = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
                    project.sd.play(audio, chunk.sample_rate)
                    project.sd.wait()
                self.state = "Voice: ready (offline)"
            except Exception as exc:
                self.state = f"Voice unavailable: {exc}"


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
def make_card(parent, **kw):
    """White panel with a thin border (flat, modern look)."""
    return tk.Frame(parent, bg=CARD, highlightthickness=1, highlightbackground=LINE,
                    highlightcolor=LINE, **kw)


def style_axes(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(LINE)
    ax.tick_params(length=0)
    ax.grid(axis="y", color="#EDF1F5", linewidth=1)
    ax.set_axisbelow(True)


class DemoApp:
    def __init__(self, root):
        self.root = root
        self.speaker = Speaker()
        threading.Thread(target=self.speaker.preload, daemon=True).start()
        self.busy = False
        self.queues = {}
        self.tally = {"CCA": [0, 0], "Welch": [0, 0]}   # [correct, total]
        self.t = np.array([])
        self.cards = []
        # flicker state
        self._flicker_running = False
        self._flicker_t0 = 0.0
        self._flicker_tgt = 0
        self._flicker_target_only = False
        self._flicker_wave = WAVE_SQUARE
        self._patch_state = [None] * len(FREQS)
        # blitting background of the EEG axes (keeps the animation cheap so the
        # flicker timing is not disturbed by matplotlib redraws)
        self._eeg_bg = None

        root.title("NeuroVocal - single-channel SSVEP communication (offline replay)")
        root.configure(bg=BG)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w, h = min(1320, sw - 40), min(860, sh - 90)
        root.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+10")
        root.minsize(1000, 640)

        self._setup_styles()
        self._build_header()
        self._build_controls()
        self._build_footer()      # packed at the bottom first, so it never gets cut off
        self._build_body()

        root.bind("<Return>", lambda e: self.on_analyze())
        root.bind("<space>", lambda e: self.on_analyze())
        self.eeg_canvas.mpl_connect("draw_event", self._on_eeg_draw)
        self._poll_voice_state()
        self._init_eeg_axes(DEFAULT_WINDOW_S, None)
        self._reset_cards()
        self._set_banner("neutral")
        self.status_cca.config(text="Press Analyze to start", fg=INK)
        self.status_welch.config(text="Choose a participant and a target, then press Space or Enter.",
                                 fg=MUTED)
        self._show_placeholder("The Welch spectrum and CCA correlations appear here "
                               "after each window")

    # ----- styles ---------------------------------------------------------
    def _setup_styles(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TCombobox", fieldbackground="white", background="white",
                        foreground=INK, bordercolor=LINE, lightcolor=LINE, darkcolor=LINE,
                        arrowcolor=TEAL, padding=6)
        style.map("TCombobox",
                  fieldbackground=[("readonly", "white")],
                  background=[("readonly", "white")],
                  foreground=[("readonly", INK)],
                  selectbackground=[("readonly", "white")],
                  selectforeground=[("readonly", INK)])
        self.root.option_add("*TCombobox*Listbox.font", (FONT, 12))
        self.root.option_add("*TCombobox*Listbox.selectBackground", TEAL)

    @staticmethod
    def _hover(btn, normal, hover):
        btn.bind("<Enter>", lambda e: btn.config(bg=hover) if str(btn["state"]) == "normal" else None)
        btn.bind("<Leave>", lambda e: btn.config(bg=normal) if str(btn["state"]) == "normal" else None)

    # ----- layout ---------------------------------------------------------
    def _build_header(self):
        bar = tk.Frame(self.root, bg=INK)
        bar.pack(fill="x")
        left = tk.Frame(bar, bg=INK)
        left.pack(side="left", padx=24, pady=8)
        tk.Label(left, text="NeuroVocal", font=(FONT, 22, "bold"), bg=INK, fg="white"
                 ).pack(anchor="w")
        tk.Label(left, text="Single-channel EEG   \u2192   CCA   \u2192   offline speech",
                 font=(FONT, 11), bg=INK, fg="#A9B8CC").pack(anchor="w")
        tk.Label(bar, text="OFFLINE REPLAY", font=(FONT, 10, "bold"), bg=TEAL, fg="white",
                 padx=14, pady=6).pack(side="right", padx=24)

    def _build_controls(self):
        wrap = tk.Frame(self.root, bg=BG)
        wrap.pack(fill="x", padx=20, pady=(14, 6))
        bar = make_card(wrap)
        bar.pack(fill="x")
        inner = tk.Frame(bar, bg=CARD)
        inner.pack(fill="x", padx=18, pady=12)

        def field(col, title, var, values, width, on_select=None):
            fr = tk.Frame(inner, bg=CARD)
            fr.grid(row=0, column=col, padx=(0, 16), sticky="w")
            tk.Label(fr, text=title, font=(FONT, 9, "bold"), bg=CARD, fg=MUTED).pack(anchor="w")
            box = ttk.Combobox(fr, textvariable=var, width=width, state="readonly",
                               values=values, font=(FONT, 13))
            box.pack(anchor="w", pady=(3, 0))

            def selected(e):
                e.widget.selection_clear()
                self.root.focus_set()
                if on_select:
                    on_select()
            box.bind("<<ComboboxSelected>>", selected)

        self.sub_var = tk.StringVar(value="P9")
        self.tgt_names = [f"{f:g} Hz" for f in FREQS]
        self.tgt_var = tk.StringVar(value=self.tgt_names[1])
        self.win_var = tk.StringVar(value=f"{DEFAULT_WINDOW_S} s")
        self.stim_var = tk.StringVar(value=STIM_STATIC)         # safe default: no flicker
        self.wave_var = tk.StringVar(value=WAVE_SQUARE)
        field(0, "PARTICIPANT", self.sub_var, [f"P{i}" for i in range(1, N_SUBJECTS + 1)], 6)
        field(1, "LOOKING AT", self.tgt_var, self.tgt_names, 8)
        field(2, "WINDOW LENGTH", self.win_var, [f"{w} s" for w in WINDOW_CHOICES], 6)
        field(3, "TARGET STIMULUS", self.stim_var, STIM_CHOICES, 13, self._on_stim_change)
        field(4, "WAVEFORM", self.wave_var, WAVE_CHOICES, 10)

        self.rt_var = tk.BooleanVar(value=True)
        self.voice_var = tk.BooleanVar(value=True)
        toggles = tk.Frame(inner, bg=CARD)
        toggles.grid(row=0, column=5, padx=(0, 16))
        for text, var in (("Real-time replay", self.rt_var), ("Speak result", self.voice_var)):
            tk.Checkbutton(toggles, text=text, variable=var, bg=CARD, fg=INK,
                           activebackground=CARD, selectcolor="white", font=(FONT, 12),
                           anchor="w", takefocus=0).pack(anchor="w")

        inner.grid_columnconfigure(6, weight=1)       # pushes the buttons to the right
        tk.Button(inner, text="Reset tally", font=(FONT, 11), bg="#E8EDF3", fg=INK, bd=0,
                  relief="flat", activebackground="#D8E0EA", padx=16, pady=10, takefocus=0,
                  cursor="hand2", command=self.on_reset).grid(row=0, column=7, padx=(0, 12))
        self.btn = tk.Button(inner, text="Analyze   (Space)", font=(FONT, 15, "bold"),
                             bg=TEAL, fg="white", bd=0, relief="flat", activebackground=INK,
                             activeforeground="white", disabledforeground="#D6E6EC",
                             padx=26, pady=10, takefocus=0, cursor="hand2",
                             command=self.on_analyze)
        self.btn.grid(row=0, column=8)
        self._hover(self.btn, TEAL, INK)

    def _build_body(self):
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=20, pady=6)

        left = tk.Frame(body, bg=BG, width=340)
        left.pack(side="left", fill="y", padx=(0, 16))
        left.grid_propagate(False)
        left.grid_columnconfigure(0, weight=1)
        tk.Label(left, text="TARGETS", font=(FONT, 9, "bold"), bg=BG, fg=MUTED
                 ).grid(row=0, column=0, sticky="w")
        for r, f in enumerate(FREQS, start=1):
            left.grid_rowconfigure(r, weight=1, uniform="cards")   # equal share of the height
            outer = make_card(left)
            outer.grid(row=r, column=0, sticky="nsew", pady=4)
            strip = tk.Frame(outer, bg=LINE, width=8)
            strip.pack(side="left", fill="y")
            # stimulus square: created now, packed only when a flicker mode is selected
            patch = tk.Frame(outer, bg=PATCH_IDLE, width=PATCH_SIZE, height=PATCH_SIZE)
            inner = tk.Frame(outer, bg=CARD)
            inner.pack(side="left", fill="x", expand=True, padx=14)   # centred vertically
            top = tk.Frame(inner, bg=CARD)
            top.pack(fill="x")
            t1 = tk.Label(top, text=f"{f:g} Hz", font=(FONT, 20, "bold"), bg=CARD, fg=INK)
            t3 = tk.Label(top, text=" ", font=(FONT, 9, "bold"), bg=CARD, fg=MUTED)
            t2 = tk.Label(inner, text=PHRASES[f], font=(FONT, 13), bg=CARD, fg=MUTED,
                          wraplength=260 - PATCH_SIZE, justify="left")
            t1.pack(side="left")
            t3.pack(side="right")
            t2.pack(anchor="w")
            self.cards.append(dict(outer=outer, strip=strip, inner=inner, top=top,
                                   labels=[t1, t2, t3], patch=patch))

        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(0, weight=2)
        right.grid_rowconfigure(1, weight=3)

        eeg_card = make_card(right)
        eeg_card.grid(row=0, column=0, sticky="nsew")
        self.eeg_fig = Figure(figsize=(7.4, 2.1), dpi=100, facecolor=CARD)
        self.eeg_ax = self.eeg_fig.add_subplot(111)
        self.eeg_canvas = FigureCanvasTkAgg(self.eeg_fig, master=eeg_card)
        self.eeg_canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

        res_card = make_card(right)
        res_card.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        self.res_fig = Figure(figsize=(7.4, 3.0), dpi=100, facecolor=CARD)
        self.res_canvas = FigureCanvasTkAgg(self.res_fig, master=res_card)
        self.res_canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

    def _build_footer(self):
        footer = tk.Frame(self.root, bg=BG)
        footer.pack(side="bottom", fill="x", padx=20, pady=(4, 10))

        row = tk.Frame(footer, bg=BG)
        row.pack(fill="x")
        self.banner = make_card(row)
        self.banner.pack(side="left", fill="both", expand=True)
        self.status_cca = tk.Label(self.banner, text="", font=(FONT, 20, "bold"), bg=CARD,
                                   fg=INK, anchor="w")
        self.status_cca.pack(fill="x", padx=22, pady=(8, 0))
        self.status_welch = tk.Label(self.banner, text="", font=(FONT, 12), bg=CARD,
                                     fg=MUTED, anchor="w")
        self.status_welch.pack(fill="x", padx=22, pady=(0, 8))

        self.chip_cca = self._make_chip(row, "CCA", TEAL)
        self.chip_welch = self._make_chip(row, "WELCH", ORANGE)

        self.voice_lbl = tk.Label(footer, text=self.speaker.state, font=(FONT, 9), bg=BG,
                                  fg=MUTED, anchor="w")
        self.voice_lbl.pack(fill="x", pady=(6, 0))
        self.warn_lbl = tk.Label(footer, text="", font=(FONT, 10, "bold"), bg=BG,
                                 fg=RED, anchor="w")
        self.warn_lbl.pack(fill="x")
        tk.Label(footer, text="Offline replay of recordings from Acampora et al. (2021) - not "
                              "live brain activity. Single dry-electrode channel, 11 healthy "
                              "volunteers. Chance level = 25%.",
                 font=(FONT, 9), bg=BG, fg=MUTED, anchor="w").pack(fill="x")
        self._update_tally()

    @staticmethod
    def _make_chip(parent, name, color):
        chip = make_card(parent)
        chip.pack(side="left", fill="y", padx=(12, 0))
        tk.Label(chip, text=name, font=(FONT, 9, "bold"), bg=CARD, fg=color
                 ).pack(anchor="w", padx=18, pady=(6, 0))
        pct = tk.Label(chip, text="-", font=(FONT, 22, "bold"), bg=CARD, fg=INK, width=5, anchor="w")
        pct.pack(anchor="w", padx=18)
        cnt = tk.Label(chip, text="0 / 0 windows", font=(FONT, 10), bg=CARD, fg=MUTED)
        cnt.pack(anchor="w", padx=18, pady=(0, 6))
        return pct, cnt

    # ----- helpers ---------------------------------------------------------
    def _poll_voice_state(self):
        self.voice_lbl.config(text=self.speaker.state)
        self.root.after(500, self._poll_voice_state)

    def _set_card(self, i, state="idle", tag=" "):
        card = self.cards[i]
        s = CARD_STATES[state]
        card["outer"].config(bg=s["bg"])
        card["inner"].config(bg=s["bg"])
        card["top"].config(bg=s["bg"])
        card["strip"].config(bg=s["strip"])
        for lb in card["labels"]:
            lb.config(bg=s["bg"])
        card["labels"][2].config(text=tag, fg=s["tag"])

    def _show_placeholder(self, text):
        self.res_fig.clear()
        self.res_fig.text(0.5, 0.5, text, ha="center", va="center", fontsize=13, color=MUTED)
        self.res_canvas.draw()

    def _reset_cards(self):
        for i in range(len(FREQS)):
            self._set_card(i)

    def _set_banner(self, state):
        bg = {"neutral": CARD, "correct": GREEN_BG, "wrong": RED_BG}[state]
        self.banner.config(bg=bg)
        self.status_cca.config(bg=bg)
        self.status_welch.config(bg=bg)

    def _update_tally(self):
        for name, (pct, cnt) in (("CCA", self.chip_cca), ("Welch", self.chip_welch)):
            c, n = self.tally[name]
            pct.config(text=f"{100 * c / n:.0f}%" if n else "-")
            cnt.config(text=f"{c} / {n} windows")

    def _on_eeg_draw(self, _event):
        """Keep a copy of the EEG axes without the moving line (for fast blitting)."""
        self._eeg_bg = self.eeg_canvas.copy_from_bbox(self.eeg_ax.bbox)

    def _init_eeg_axes(self, L, w):
        ax = self.eeg_ax
        ax.clear()
        style_axes(ax)
        ax.set_xlim(0, L)
        if w is not None:
            pad = 0.1 * (w.max() - w.min() + 1e-9)
            ax.set_ylim(w.min() - pad, w.max() + pad)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("EEG (a.u.)")
        ax.set_title(f"Single-channel EEG  \u00b7  {L} s window", loc="left", fontsize=12,
                     fontweight="bold")
        (self.eeg_line,) = ax.plot([], [], color=TEAL, lw=1.4, animated=True)
        self.eeg_fig.tight_layout()
        self.eeg_canvas.draw()          # also stores the blit background (draw_event)

    def _next_window(self, key, n_win):
        q = self.queues.get(key)
        if not q:
            q = list(range(n_win))
            random.shuffle(q)
            self.queues[key] = q
        return q.pop()

    # ----- flicker stimulus --------------------------------------------------
    def _on_stim_change(self):
        """Show the stimulus squares only when a flicker mode is selected."""
        mode = self.stim_var.get()
        for card in self.cards:
            if mode == STIM_STATIC:
                card["patch"].pack_forget()
            else:
                card["patch"].pack(side="right", padx=14, before=card["inner"])
        self._set_patches_idle()
        if mode == STIM_STATIC:
            self.warn_lbl.config(text="")
        else:
            self.warn_lbl.config(text="\u26a0 Flicker mode ON - may trigger seizures in "
                                      "photosensitive people.")

    def _set_patches_idle(self):
        for i, card in enumerate(self.cards):
            card["patch"].config(bg=PATCH_IDLE)
            self._patch_state[i] = None

    def _start_flicker(self, tgt):
        self._flicker_tgt = tgt
        self._flicker_target_only = self.stim_var.get() == STIM_TARGET
        self._flicker_wave = self.wave_var.get()
        self._patch_state = [None] * len(FREQS)
        self._flicker_running = True
        self._flicker_t0 = time.perf_counter()
        self._flicker_tick()

    def _stop_flicker(self):
        self._flicker_running = False
        self._set_patches_idle()

    def _flicker_tick(self):
        """Update every stimulus square from the real clock (not a frame counter), so each
        square keeps its true frequency even if one Tk update arrives late."""
        if not self._flicker_running:
            return
        t = time.perf_counter() - self._flicker_t0
        for i, f in enumerate(FREQS):
            if self._flicker_target_only and i != self._flicker_tgt:
                continue                                     # non-targets stay at rest
            g = stimulus_gray(f, t, self._flicker_wave)
            if g != self._patch_state[i]:                    # touch the widget only on change
                self._patch_state[i] = g
                self.cards[i]["patch"].config(bg=f"#{g:02x}{g:02x}{g:02x}")
        self.root.after(FLICKER_TICK_MS, self._flicker_tick)

    # ----- actions ---------------------------------------------------------
    def on_reset(self):
        self.tally = {"CCA": [0, 0], "Welch": [0, 0]}
        self._update_tally()

    def on_analyze(self):
        if self.busy:
            return
        sub = int(self.sub_var.get()[1:])
        tgt = self.tgt_names.index(self.tgt_var.get())
        L = int(self.win_var.get().split()[0])
        try:
            rec = load_recording(sub, tgt)
        except Exception as exc:
            self._set_banner("wrong")
            self.status_cca.config(text=f"Could not load data: {exc}", fg=RED)
            return
        n = L * FS
        n_win = len(rec) // n
        k = self._next_window((sub, tgt, L), n_win)
        w = rec[k * n:(k + 1) * n]

        self.busy = True
        self.btn.config(state="disabled", bg="#7FA8B8")
        self._reset_cards()
        self._set_card(tgt, "looking", "LOOKING HERE")
        self._show_placeholder("Decision appears when the window is complete")
        self._set_banner("neutral")
        self.status_cca.config(text=f"Recording window {k + 1} of {n_win} ...", fg=INK)
        self.status_welch.config(text="Waiting for the full window before deciding", fg=MUTED)
        self.t = np.arange(n) / FS
        self._init_eeg_axes(L, w)
        self.eeg_line.set_data([], [])

        flicker = self.stim_var.get() != STIM_STATIC
        # the flicker only makes sense while the window "plays", so it implies real-time replay
        if self.rt_var.get() or flicker:
            t0 = time.perf_counter()
            if flicker:
                self._start_flicker(tgt)
            self._animate(w, tgt, t0)
        else:
            self._finish(w, tgt)

    def _animate(self, w, tgt, t0):
        n = len(w)
        k = min(n, int((time.perf_counter() - t0) * FS))
        self.eeg_line.set_data(self.t[:k], w[:k])
        if self._eeg_bg is not None:
            # fast path: redraw only the moving line (keeps the flicker timing steady)
            self.eeg_canvas.restore_region(self._eeg_bg)
            self.eeg_ax.draw_artist(self.eeg_line)
            self.eeg_canvas.blit(self.eeg_ax.bbox)
        else:
            self.eeg_canvas.draw_idle()
        if k < n:
            self.root.after(40, lambda: self._animate(w, tgt, t0))
        else:
            self._finish(w, tgt)

    def _finish(self, w, tgt):
        try:
            self._stop_flicker()
            self.eeg_line.set_animated(False)      # draw the complete trace normally
            self.eeg_line.set_data(self.t, w)
            self.eeg_canvas.draw_idle()
            f, p, w_idx = welch_analysis(w)
            corr, c_idx = cca_analysis(w)
            c_ok, w_ok = c_idx == tgt, w_idx == tgt

            self.tally["CCA"][0] += int(c_ok)
            self.tally["CCA"][1] += 1
            self.tally["Welch"][0] += int(w_ok)
            self.tally["Welch"][1] += 1
            self._update_tally()

            if c_ok:
                self._set_card(c_idx, "correct", "CORRECT")
            else:
                self._set_card(c_idx, "wrong", "CCA: WRONG")
                self._set_card(tgt, "looking", "LOOKING HERE")
            self._draw_results(f, p, w_idx, corr, c_idx, tgt)

            phrase = PHRASES[FREQS[c_idx]]
            self._set_banner("correct" if c_ok else "wrong")
            self.status_cca.config(text=f"\u201c{phrase}\u201d", fg=GREEN if c_ok else RED)
            self.status_welch.config(
                text=f"CCA detected {FREQS[c_idx]:g} Hz \u2014 {'correct' if c_ok else 'wrong'}"
                     f"     \u00b7     Welch baseline detected {FREQS[w_idx]:g} Hz \u2014 "
                     f"{'correct' if w_ok else 'wrong'}",
                fg=INK)
            if self.voice_var.get():
                self.speaker.say(phrase)
        except Exception as exc:
            self._set_banner("wrong")
            self.status_cca.config(text=f"Error: {exc}", fg=RED)
        finally:
            self.busy = False
            self.btn.config(state="normal", bg=TEAL)

    def _draw_results(self, f, p, w_idx, corr, c_idx, tgt):
        fig = self.res_fig
        fig.clear()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        style_axes(ax1)
        style_axes(ax2)

        m = (f >= 5) & (f <= 35)
        ax1.plot(f[m], p[m], color=ORANGE, lw=2.2)
        ax1.fill_between(f[m], p[m], color=ORANGE, alpha=0.14)
        for fq in FREQS:
            ax1.axvline(fq, ls="--", color="#B8C2CF", lw=1)
        ax1.axvline(FREQS[w_idx], color=ORANGE, lw=6, alpha=0.4)
        ax1.set_title(f"Welch baseline  \u2192  {FREQS[w_idx]:g} Hz", loc="left", fontsize=13,
                      fontweight="bold")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Power")
        ax1.set_yticks([])

        colors = ["#C9DCE5"] * len(FREQS)
        colors[c_idx] = TEAL
        ax2.bar(range(len(FREQS)), corr, width=0.6, color=colors)
        for i, v in enumerate(corr):
            ax2.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=11, color=MUTED)
        ax2.set_xticks(range(len(FREQS)))
        ax2.set_xticklabels([f"{x:g}" for x in FREQS])
        ax2.set_ylim(0, 1.1)
        ax2.set_yticks([0, 0.5, 1.0])
        ax2.set_title(f"CCA  \u2192  {FREQS[c_idx]:g} Hz", loc="left", fontsize=13,
                      fontweight="bold")
        ax2.set_xlabel(f"Frequency (Hz)  \u00b7  looking at {FREQS[tgt]:g}")
        ax2.set_ylabel("Canonical correlation")

        fig.tight_layout()
        self.res_canvas.draw()


def main():
    missing = [s for s in range(1, N_SUBJECTS + 1)
               if not os.path.exists(os.path.join(DATA_DIR, f"subject{s}.csv"))]
    if missing:
        sys.exit(f"Missing data files in {DATA_DIR}: subject{missing}.csv - "
                 f"check DATA_DIR at the top of demo.py")
    try:                                    # sharper text on Windows high-DPI screens
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    DemoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
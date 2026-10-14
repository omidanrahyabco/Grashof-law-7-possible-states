"""Render the Grashof 7-states teaching video and export the source data.

Usage:  python3 src/render_video.py
Outputs: output/grashof_7_states.mp4, output/grashof_7_states_poster.png,
         data/cases.csv, data/cases.json, data/kinematics/case_N_*.csv
"""

import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

from fourbar import CASES, classify, collinear_mask, simulate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output")
DATA = os.path.join(ROOT, "data")

FPS = 30
W, H = 12.8, 7.2          # inches at dpi 100 -> 1280 x 720
INTRO_S, CASE_S, OUTRO_S, FADE_S = 3.5, 6.0, 7.0, 0.35

BG = "#F7F6F2"
INK = "#22262E"
MUTED = "#6B7280"
GROUND = "#8A8F98"
COL = {2: "#2F6FDE", 3: "#E8833A", 4: "#1E9E6A"}
FAMILY_COL = {"Grashof": "#2F6FDE", "Non-Grashof": "#C2410C", "Change point": "#7C3AED"}

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})


# ---------------------------------------------------------------- data --------

def case_info(case):
    d = case.slpq()
    motion = classify(case)
    names = {}
    for k, m in motion.items():
        if k == 3:
            if m["full_turn"]:
                names[k] = "Coupler – turns 360°"
            elif m["swing_deg"] < 1:
                names[k] = "Coupler – translates (no rotation)"
            else:
                names[k] = f"Coupler – swings {m['swing_deg']:.0f}°"
        else:
            names[k] = "CRANK – turns 360°" if m["full_turn"] else f"ROCKER – swings {m['swing_deg']:.0f}°"
    return d, motion, names


def export_data(infos):
    os.makedirs(os.path.join(DATA, "kinematics"), exist_ok=True)
    rows = []
    for case, (d, motion, _) in zip(CASES, infos):
        rows.append({
            "case": case.number, "name": case.title, "family": case.family,
            "link1_ground": case.ground, "link2": case.link2, "link3_coupler": case.coupler, "link4": case.link4,
            "s": d["s"], "l": d["l"], "p": d["p"], "q": d["q"],
            "l_plus_s": round(d["l"] + d["s"], 6), "p_plus_q": round(d["p"] + d["q"], 6),
            "relation": f"l+s {case.relation} p+q", "shortest_link": case.shortest_where,
            "link2_motion": "crank" if motion[2]["full_turn"] else f"rocker {motion[2]['swing_deg']:.1f} deg",
            "link3_motion": "full turn" if motion[3]["full_turn"] else (
                "translation" if motion[3]["swing_deg"] < 1 else f"swing {motion[3]['swing_deg']:.1f} deg"),
            "link4_motion": "crank" if motion[4]["full_turn"] else f"rocker {motion[4]['swing_deg']:.1f} deg",
            "driven_by": case.driver, "input_mode": case.mode, "lecture_ref": case.lecture_ref,
        })
        sim = simulate(case, CASE_S, FPS)
        path = os.path.join(DATA, "kinematics", f"case_{case.number}_{case.key}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "input_deg", "Ax", "Ay", "Bx", "By"])
            for i in range(len(sim["t"])):
                w.writerow([f"{sim['t'][i]:.4f}", f"{np.degrees(sim['input'][i]):.3f}",
                            *(f"{v:.5f}" for v in (*sim["A"][i], *sim["B"][i]))])
    with open(os.path.join(DATA, "cases.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(DATA, "cases.json"), "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


# ------------------------------------------------------------- drawing --------

def ground_support(ax, p, size, z=2):
    s = size
    ax.add_patch(Polygon([[p[0], p[1]], [p[0] - s, p[1] - 1.5 * s], [p[0] + s, p[1] - 1.5 * s]],
                         closed=True, fc="#D9DBDF", ec=GROUND, lw=1.5, zorder=z))
    y = p[1] - 1.5 * s
    ax.plot([p[0] - 1.6 * s, p[0] + 1.6 * s], [y, y], color=GROUND, lw=1.8, zorder=z)
    for x in np.linspace(p[0] - 1.5 * s, p[0] + 1.3 * s, 6):
        ax.plot([x, x - 0.4 * s], [y, y - 0.5 * s], color=GROUND, lw=1.0, zorder=z)


class Linkage:
    """Matplotlib artists for one animated four-bar."""

    def __init__(self, ax, case, sim, scale=1.0, labels=True, trail_s=0.6):
        self.ax, self.sim, self.labels = ax, sim, labels
        O2, O4 = sim["O2"], sim["O4"]
        L = max(case.lengths.values())
        lw = 9 * scale
        pin = 0.075 * L
        self.trail_n = int(trail_s * FPS)

        # full joint paths (faint) reveal crank (circle) vs rocker (arc)
        ax.plot(*sim["A_full"].T, color=COL[2], lw=1.3, alpha=0.3, zorder=1)
        ax.plot(*sim["B_full"].T, color=COL[4], lw=1.3, alpha=0.3, zorder=1)
        self.trA, = ax.plot([], [], color=COL[2], lw=2.5 * scale, alpha=0.55, zorder=1.5, solid_capstyle="round")
        self.trB, = ax.plot([], [], color=COL[4], lw=2.5 * scale, alpha=0.55, zorder=1.5, solid_capstyle="round")

        ground_support(ax, O2, 0.09 * L)
        ground_support(ax, O4, 0.09 * L)
        ax.plot(*np.c_[O2, O4], color=GROUND, lw=lw * 0.75, solid_capstyle="round", zorder=2.5, alpha=0.9)

        self.l2, = ax.plot([], [], color=COL[2], lw=lw, solid_capstyle="round", zorder=4)
        self.l3, = ax.plot([], [], color=COL[3], lw=lw, solid_capstyle="round", zorder=3.5)
        self.l4, = ax.plot([], [], color=COL[4], lw=lw, solid_capstyle="round", zorder=4)
        for p in (O2, O4):
            ax.add_patch(Circle(p, pin * 0.55, fc="white", ec=INK, lw=1.6 * scale, zorder=6))
        self.pA = Circle((0, 0), pin * 0.55, fc="white", ec=INK, lw=1.6 * scale, zorder=6)
        self.pB = Circle((0, 0), pin * 0.55, fc="white", ec=INK, lw=1.6 * scale, zorder=6)
        ax.add_patch(self.pA)
        ax.add_patch(self.pB)

        if labels:
            box = dict(boxstyle="circle,pad=0.25", fc="white", lw=1.4)
            mid = (O2 + O4) / 2
            ax.text(mid[0], mid[1] - 0.16 * L, "1", ha="center", va="center", fontsize=11,
                    color=GROUND, fontweight="bold", bbox={**box, "ec": GROUND}, zorder=7)
            self.t = {k: ax.text(0, 0, str(k), ha="center", va="center", fontsize=11, fontweight="bold",
                                 color=COL[k], bbox={**box, "ec": COL[k]}, zorder=7) for k in (2, 3, 4)}

    def update(self, i):
        s = self.sim
        A, B, O2, O4 = s["A"][i], s["B"][i], s["O2"], s["O4"]
        self.l2.set_data([O2[0], A[0]], [O2[1], A[1]])
        self.l3.set_data([A[0], B[0]], [A[1], B[1]])
        self.l4.set_data([O4[0], B[0]], [O4[1], B[1]])
        self.pA.center, self.pB.center = A, B
        j = max(0, i - self.trail_n)
        self.trA.set_data(*s["A"][j:i + 1].T)
        self.trB.set_data(*s["B"][j:i + 1].T)
        if self.labels:
            for k, (P, Q) in {2: (O2, A), 3: (A, B), 4: (O4, B)}.items():
                m = (P + Q) / 2
                self.t[k].set_position(m)


def fit_axes(ax, sims, case, pad=0.18):
    pts = np.vstack([sims["A_full"], sims["B_full"], [sims["O2"], sims["O4"]]])
    L = max(case.lengths.values())
    lo, hi = pts.min(0) - pad * L, pts.max(0) + pad * L
    lo[1] -= 0.12 * L                      # room for the ground supports
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")


def run_sim(case, seconds):
    sim = simulate(case, seconds, FPS)
    full = simulate(case, 2 * case.period + 0.1, FPS)   # 2 cycles: closes even when the path needs 2 turns
    sim["A_full"], sim["B_full"] = full["A"], full["B"]
    return sim


def fade_alpha(i, n):
    t_in, t_out = i / FPS, (n - 1 - i) / FPS
    return float(np.clip(1 - min(t_in, t_out) / FADE_S, 0, 1))


def add_fader(fig):
    r = Rectangle((0, 0), 1, 1, transform=fig.transFigure, fc=BG, ec="none", zorder=1000)
    fig.add_artist(r)
    return r


def progress_dots(fig, current):
    for k in range(1, 8):
        x = 0.78 + (k - 1) * 0.028
        on = k == current
        fig.add_artist(Circle((x, 0.955), 0.0075 if on else 0.0055, transform=fig.transFigure,
                              fc=INK if on else "#C9CCD2", ec="none"))
    fig.text(0.78 - 0.02, 0.955, f"Case {current} / 7", ha="right", va="center", fontsize=11, color=MUTED)


# --------------------------------------------------------------- scenes -------

def scene_intro(fig, writer):
    fig.clf()
    fig.patch.set_facecolor(BG)
    fig.text(0.5, 0.74, "Grashof's Law", ha="center", fontsize=44, fontweight="bold")
    fig.text(0.5, 0.645, "The 7 possible states of a four-bar linkage", ha="center", fontsize=22, color=MUTED)
    fig.text(0.5, 0.50, r"$s$ = shortest     $l$ = longest     $p,\ q$ = the other two links",
             ha="center", fontsize=18)
    rows = [(r"$l + s < p + q$", "Grashof: at least one link turns a full 360°", FAMILY_COL["Grashof"]),
            (r"$l + s > p + q$", "Non-Grashof: no link can turn fully (triple rocker)", FAMILY_COL["Non-Grashof"]),
            (r"$l + s = p + q$", "Change point: links can line up and switch branch", FAMILY_COL["Change point"])]
    for j, (eq, txt, c) in enumerate(rows):
        y = 0.38 - j * 0.075
        fig.text(0.30, y, eq, ha="left", va="center", fontsize=19, color=c)
        fig.text(0.47, y, txt, ha="left", va="center", fontsize=15)
    legend = [("1  ground", GROUND), ("2  side link (input)", COL[2]), ("3  coupler", COL[3]), ("4  side link (output)", COL[4])]
    for j, (txt, c) in enumerate(legend):
        x = 0.16 + j * 0.19
        fig.add_artist(Rectangle((x, 0.085), 0.025, 0.012, transform=fig.transFigure, fc=c, ec="none"))
        fig.text(x + 0.032, 0.091, txt, va="center", fontsize=12, color=INK)
    fig.text(0.5, 0.025, "Dynamics of Machines – Lecture 6, slides 2–4", ha="center", fontsize=10, color=MUTED)
    fader = add_fader(fig)
    n = int(INTRO_S * FPS)
    for i in range(n):
        fader.set_alpha(fade_alpha(i, n))
        writer.grab_frame(facecolor=BG)


def scene_case(fig, writer, case, info):
    d, motion, names = info
    fig.clf()
    fig.patch.set_facecolor(BG)
    sim = run_sim(case, CASE_S)
    ax = fig.add_axes([0.02, 0.04, 0.56, 0.86])
    fit_axes(ax, sim, case)
    link = Linkage(ax, case, sim)

    fam_c = FAMILY_COL[case.family]
    fig.text(0.03, 0.955, "GRASHOF'S LAW  ·  FOUR-BAR STATES", fontsize=11, color=MUTED, va="center",
             fontweight="bold")
    progress_dots(fig, case.number)

    x0 = 0.60
    fig.add_artist(FancyBboxPatch((x0, 0.795), 0.13, 0.042, boxstyle="round,pad=0.004,rounding_size=0.01",
                                  transform=fig.transFigure, fc=fam_c, ec="none"))
    fig.text(x0 + 0.065, 0.816, case.family.upper(), ha="center", va="center", fontsize=11, color="white",
             fontweight="bold")
    fig.text(x0, 0.725, case.title, fontsize=26, fontweight="bold", va="center")

    sym = {"<": "<", ">": ">", "=": "="}[case.relation]
    fig.text(x0, 0.635, rf"$l + s\ {sym}\ p + q$", fontsize=24, color=fam_c, va="center")
    nums = f"{d['l']:g} + {d['s']:g} = {d['l'] + d['s']:g}   {sym}   {d['p'] + d['q']:g} = {d['p']:g} + {d['q']:g}"
    fig.text(x0, 0.575, nums, fontsize=15, va="center", color=INK)
    fig.text(x0, 0.52, f"Shortest: {case.shortest_where}", fontsize=12.5, va="center", color=MUTED)

    rows = [(1, f"Ground – fixed   (length {case.ground:g})", GROUND)]
    rows += [(k, f"{names[k]}   (length {case.lengths[k]:g})", COL[k]) for k in (2, 3, 4)]
    for j, (k, txt, c) in enumerate(rows):
        y = 0.43 - j * 0.065
        fig.add_artist(Circle((x0 + 0.012, y), 0.014, transform=fig.transFigure, fc="white", ec=c, lw=2))
        fig.text(x0 + 0.012, y, str(k), ha="center", va="center", fontsize=11, color=c, fontweight="bold")
        bold = "bold" if "360" in txt and k != 3 else "normal"
        fig.text(x0 + 0.035, y, txt, va="center", fontsize=13.5, fontweight=bold)

    fig.text(x0, 0.14, case.note, fontsize=12.5, va="center", color=INK, wrap=True)
    fig.text(x0, 0.075, case.lecture_ref, fontsize=10.5, va="center", color=MUTED)

    L = max(case.lengths.values())
    cp = collinear_mask(sim, 0.03 * L)
    hold = np.convolve(cp.astype(float), np.ones(int(0.5 * FPS)), mode="same") > 0
    banner = ax.text(0.5, 0.97, "CHANGE POINT – all links collinear", transform=ax.transAxes,
                     ha="center", va="top", fontsize=14, fontweight="bold", color="white",
                     bbox=dict(boxstyle="round,pad=0.4", fc=FAMILY_COL["Change point"], ec="none"))
    banner.set_visible(False)

    fader = add_fader(fig)
    n = len(sim["t"])
    for i in range(n):
        link.update(i)
        banner.set_visible(bool(hold[i]))
        fader.set_alpha(fade_alpha(i, n))
        writer.grab_frame(facecolor=BG)


def scene_summary(fig, writer, infos, still_path=None):
    fig.clf()
    fig.patch.set_facecolor(BG)
    fig.text(0.5, 0.955, "Grashof's Law – all 7 states", ha="center", va="center", fontsize=22,
             fontweight="bold")
    links = []
    cw, ch = 0.245, 0.42
    for idx, (case, (d, motion, _)) in enumerate(zip(CASES, infos)):
        r, c = divmod(idx, 4)
        x, y = 0.005 + c * 0.2475, 0.48 - r * 0.455
        ax = fig.add_axes([x, y + 0.03, cw, ch - 0.06])
        sim = run_sim(case, OUTRO_S)
        fit_axes(ax, sim, case, pad=0.12)
        links.append(Linkage(ax, case, sim, scale=0.55, labels=False, trail_s=0.4))
        fam_c = FAMILY_COL[case.family]
        fig.text(x + cw / 2, y + ch - 0.005, f"{case.number}. {case.title}", ha="center", va="center",
                 fontsize=12.5, fontweight="bold")
        tags = []
        for k in (2, 4):
            tags.append(f"{k}: {'crank' if motion[k]['full_turn'] else 'rocker'}")
        if motion[3]["full_turn"] and not (motion[2]["full_turn"] and motion[4]["full_turn"]):
            tags.append("3: turns")
        fig.text(x + cw / 2, y + 0.012, rf"$l+s {case.relation} p+q$   " + "  ·  ".join(tags),
                 ha="center", va="center", fontsize=10.5, color=fam_c)
    # legend cell
    x, y = 0.005 + 3 * 0.2475, 0.025
    for j, (fam, txt) in enumerate([("Grashof", "l + s < p + q"), ("Non-Grashof", "l + s > p + q"),
                                    ("Change point", "l + s = p + q")]):
        yy = y + 0.33 - j * 0.07
        fig.add_artist(Rectangle((x + 0.02, yy - 0.012), 0.022, 0.024, transform=fig.transFigure,
                                 fc=FAMILY_COL[fam], ec="none"))
        fig.text(x + 0.05, yy, f"{fam}:  {txt}", va="center", fontsize=11.5)
    fig.text(x + 0.02, y + 0.10, "Faint curves = paths of the moving\njoints: full circle = crank,\narc = rocker",
             va="center", fontsize=10.5, color=MUTED, linespacing=1.4)

    fader = add_fader(fig)
    n = int(OUTRO_S * FPS)
    for i in range(n):
        for lk in links:
            lk.update(i)
        a = fade_alpha(i, n) if i < n / 2 else float(np.clip(1 - (n - 1 - i) / (2 * FADE_S * FPS), 0, 1))
        fader.set_alpha(a)
        if still_path and i == int(1.4 * FPS):
            fader.set_alpha(0)
            fig.savefig(still_path, facecolor=BG, dpi=150)
        writer.grab_frame(facecolor=BG)


def main():
    os.makedirs(OUT, exist_ok=True)
    infos = [case_info(c) for c in CASES]
    export_data(infos)

    fig = plt.figure(figsize=(W, H), dpi=100)
    writer = FFMpegWriter(fps=FPS, codec="libx264", bitrate=-1,
                          extra_args=["-crf", "30", "-preset", "slow", "-pix_fmt", "yuv420p",
                                      "-tune", "animation", "-an", "-movflags", "+faststart"])
    video = os.path.join(OUT, "grashof_7_states.mp4")
    with writer.saving(fig, video, dpi=100):
        scene_intro(fig, writer)
        for case, info in zip(CASES, infos):
            scene_case(fig, writer, case, info)
            print("rendered case", case.number, case.title, flush=True)
        scene_summary(fig, writer, infos, still_path=os.path.join(OUT, "grashof_7_states_poster.png"))
    print("wrote", video)


if __name__ == "__main__":
    main()

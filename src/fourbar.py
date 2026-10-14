"""Planar four-bar kinematics and the 7 Grashof cases from DoM Lecture 6 (slides 2-4).

Link numbering follows the lecture:
    1 = ground (fixed, from O2 to O4)
    2 = left side link (pivots at O2, moving joint A)
    3 = coupler (A to B)
    4 = right side link (pivots at O4, moving joint B)

Grashof: s = shortest, l = longest, p, q = the other two.
    l + s < p + q  -> Grashof (at least one link turns 360 deg)
    l + s > p + q  -> non-Grashof (triple rocker)
    l + s = p + q  -> special Grashof / change point
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Case:
    number: int
    key: str
    title: str
    family: str          # "Grashof", "Non-Grashof", "Change point"
    relation: str        # "<", ">", "="
    ground: float        # link 1
    link2: float         # link 2 (O2-A)
    coupler: float       # link 3 (A-B)
    link4: float         # link 4 (O4-B)
    driver: str          # "link2" or "coupler": the angle that is prescribed
    mode: str            # "rotate" (full turn) or "rock" (oscillate between toggle limits)
    period: float        # seconds per input cycle
    start_deg: float     # starting input angle
    shortest_where: str
    note: str
    lecture_ref: str
    lengths: dict = field(init=False)

    def __post_init__(self):
        self.lengths = {1: self.ground, 2: self.link2, 3: self.coupler, 4: self.link4}

    def slpq(self):
        v = sorted(self.lengths.values())
        return {"s": v[0], "p": v[1], "q": v[2], "l": v[3]}


CASES = [
    Case(1, "triple_rocker", "Triple Rocker", "Non-Grashof", ">",
         ground=4.5, link2=2.0, coupler=2.6, link4=2.4,
         driver="link2", mode="rock", period=3.6, start_deg=0.0,
         shortest_where="any link (no link can turn fully)",
         note="No link can make a full turn: every moving link only rocks.",
         lecture_ref="Slide 3: l + s > p + q"),
    Case(2, "double_crank", "Double Crank", "Grashof", "<",
         ground=1.6, link2=3.2, coupler=3.8, link4=4.0,
         driver="link2", mode="rotate", period=3.0, start_deg=60.0,
         shortest_where="link 1 (the ground)",
         note="Shortest link is the ground, so both side links turn 360°.",
         lecture_ref="Slide 3: s = link 1"),
    Case(3, "double_rocker", "Double Rocker", "Grashof", "<",
         ground=4.5, link2=3.0, coupler=1.4, link4=3.5,
         driver="coupler", mode="rotate", period=3.0, start_deg=0.0,
         shortest_where="link 3 (coupler, opposite the ground)",
         note="Shortest link is the coupler: it turns 360°, the side links rock.",
         lecture_ref="Slide 3: s = link 3"),
    Case(4, "crank_rocker", "Crank–Rocker", "Grashof", "<",
         ground=4.0, link2=1.4, coupler=4.2, link4=3.2,
         driver="link2", mode="rotate", period=3.0, start_deg=60.0,
         shortest_where="link 2 (adjacent to the ground)",
         note="Shortest link is next to the ground: it is the crank, link 4 rocks.",
         lecture_ref="Slide 3: s = link 2 or 4"),
    Case(5, "parallelogram", "Parallelogram", "Change point", "=",
         ground=4.0, link2=2.0, coupler=4.0, link4=2.0,
         driver="link2", mode="rotate", period=3.0, start_deg=60.0,
         shortest_where="links 2 and 4 (equal, opposite each other)",
         note="Equal opposite links. At the change point it could flip to an anti-parallelogram.",
         lecture_ref="Slide 4: parallelogram"),
    Case(6, "deltoid", "Deltoid (Kite)", "Change point", "=",
         ground=4.0, link2=1.8, coupler=1.8, link4=4.0,
         driver="link2", mode="rotate", period=3.0, start_deg=60.0,
         shortest_where="links 2 and 3 (equal, adjacent)",
         note="Equal adjacent links. At the change point the motion can switch branch.",
         lecture_ref="Slide 4: deltoid"),
    Case(7, "change_point", "General Change Point", "Change point", "=",
         ground=3.0, link2=1.5, coupler=3.5, link4=2.0,
         driver="link2", mode="rotate", period=3.0, start_deg=60.0,
         shortest_where="link 2 (adjacent to the ground)",
         note="Lecture example 3.5 + 1.5 = 2 + 3: all links line up once per turn.",
         lecture_ref="Slide 4: 3.5, 1.5, 2, 3"),
]


def _unit(a):
    return np.array([np.cos(a), np.sin(a)])


def circle_intersections(p0, r0, p1, r1):
    """Both intersections of two circles (tangent/near-miss is clamped to the touching point)."""
    d_vec = p1 - p0
    d = np.hypot(*d_vec)
    u = d_vec / d
    along = (r0 * r0 - r1 * r1 + d * d) / (2 * d)
    h = np.sqrt(max(r0 * r0 - along * along, 0.0))
    base = p0 + along * u
    n = np.array([-u[1], u[0]])
    return base + h * n, base - h * n


def rock_limits(case):
    """Input (link 2) angle range for a rocking input: limits where links 3 and 4 are collinear."""
    a, g, c, b = case.link2, case.ground, case.coupler, case.link4
    lo = (a * a + g * g - (c + b) ** 2) / (2 * a * g)
    hi = (a * a + g * g - (c - b) ** 2) / (2 * a * g)
    if hi >= 1 and lo > -1:            # range centred on 0 rad
        m = np.arccos(lo)
        return 0.0, m
    if lo <= -1 and hi < 1:            # range centred on pi rad
        m = np.pi - np.arccos(hi)
        return np.pi, m
    raise ValueError("input link has two separate ranges; not handled")


def simulate(case, duration, fps, substeps=40):
    """Positions of the moving joints A and B for every video frame.

    The input is advanced in small sub-steps and, at each step, the branch
    (assembly) is chosen by continuity: the candidate closest to a linear
    extrapolation of the two previous states. This carries the linkage
    smoothly through toggle and change-point positions.
    """
    O2 = np.zeros(2)
    O4 = np.array([case.ground, 0.0])
    n_frames = int(round(duration * fps))
    n_steps = n_frames * substeps
    t = np.arange(n_steps + 1) / (fps * substeps)

    if case.mode == "rotate":
        q = np.radians(case.start_deg) + 2 * np.pi * t / case.period
    else:
        centre, amp = rock_limits(case)
        amp *= 1 - 1e-12
        q = centre + amp * np.sin(2 * np.pi * t / case.period + np.radians(case.start_deg))

    def candidates(qi):
        if case.driver == "link2":
            A = O2 + case.link2 * _unit(qi)
            B1, B2 = circle_intersections(A, case.coupler, O4, case.link4)
            return [np.r_[A, B1], np.r_[A, B2]]
        e = _unit(qi)
        A1, A2 = circle_intersections(O2, case.link2, O4 - case.coupler * e, case.link4)
        return [np.r_[A1, A1 + case.coupler * e], np.r_[A2, A2 + case.coupler * e]]

    # start in the "open" assembly: B (or A) above the ground line
    first = max(candidates(q[0]), key=lambda s: s[3] + s[1])
    states = [first]
    prev2, prev1 = first, first
    for qi in q[1:]:
        pred = 2 * prev1 - prev2
        best = min(candidates(qi), key=lambda s: np.sum((s - pred) ** 2))
        states.append(best)
        prev2, prev1 = prev1, best
    states = np.array(states)[::substeps][:n_frames]
    return {
        "t": np.arange(n_frames) / fps,
        "input": q[::substeps][:n_frames],
        "A": states[:, 0:2],
        "B": states[:, 2:4],
        "O2": O2,
        "O4": O4,
    }


def link_angles(sim):
    A, B, O4 = sim["A"], sim["B"], sim["O4"]
    th2 = np.unwrap(np.arctan2(A[:, 1], A[:, 0]))
    th3 = np.unwrap(np.arctan2(B[:, 1] - A[:, 1], B[:, 0] - A[:, 0]))
    th4 = np.unwrap(np.arctan2(B[:, 1] - O4[1], B[:, 0] - O4[0]))
    return {2: th2, 3: th3, 4: th4}


def classify(case, fps=60):
    """Motion of each moving link, measured from the simulation (not assumed)."""
    sim = simulate(case, duration=4 * case.period, fps=fps)
    out = {}
    for k, th in link_angles(sim).items():
        span = np.degrees(th.max() - th.min())
        out[k] = {"full_turn": bool(span >= 359.0), "swing_deg": float(min(span, 360.0))}
    return out


def collinear_mask(sim, tol):
    """Frames where all four joints lie on the ground line (change point)."""
    return (np.abs(sim["A"][:, 1]) < tol) & (np.abs(sim["B"][:, 1]) < tol)

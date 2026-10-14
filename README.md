# Grashof's Law: the 7 states of a four-bar linkage

An animated teaching simulation of the seven four-bar cases from *Dynamics of Machines, Lecture 6* (slides 2–4).

**Video:** [`output/grashof_7_states.mp4`](output/grashof_7_states.mp4). It runs 52 s at 1280×720 and 30 fps, is 1.4 MB, and has no audio.
**Poster (all 7 on one page):** [`output/grashof_7_states_poster.png`](output/grashof_7_states_poster.png)

## The rule

`s` = shortest link, `l` = longest, `p`, `q` = the other two.

| Condition | Meaning |
|---|---|
| `l + s < p + q` | Grashof: at least one link can turn a full 360° |
| `l + s > p + q` | Non-Grashof: no link turns fully (triple rocker) |
| `l + s = p + q` | Special case / change point: the links can line up and switch branch |

Link numbering follows the lecture: **1** ground, **2** left side link (input), **3** coupler, **4** right side link.

## The 7 cases in the video

| # | Case | Lengths (1, 2, 3, 4) | l + s vs p + q | Link 2 | Coupler 3 | Link 4 |
|---|---|---|---|---|---|---|
| 1 | Triple rocker | 4.5, 2, 2.6, 2.4 | 6.5 > 5 | rocker 185° | swings 142° | rocker 154° |
| 2 | Double crank (s = ground) | 1.6, 3.2, 3.8, 4 | 5.6 < 7 | crank | full turn | crank |
| 3 | Double rocker (s = coupler) | 4.5, 3, 1.4, 3.5 | 5.9 < 6.5 | rocker 56° | **full turn** | rocker 47° |
| 4 | Crank–rocker (s next to ground) | 4, 1.4, 4.2, 3.2 | 5.6 < 7.2 | crank | swings 43° | rocker 57° |
| 5 | Parallelogram | 4, 2, 4, 2 | 6 = 6 | crank | translates | crank |
| 6 | Deltoid (kite) | 4, 1.8, 1.8, 4 | 5.8 = 5.8 | crank | full turn | rocker 107° |
| 7 | General change point (lecture's 3.5, 1.5, 2, 3) | 3, 1.5, 3.5, 2 | 5 = 5 | crank | swings 129° | rocker 277° |

The crank and rocker labels are not typed in by hand. They come from the simulation, by measuring how far each link actually turns. Cases 5–7 show a **CHANGE POINT** banner when all four links lie on one line.

> **Note on slide 3:** for *s = link 2 or 4* (shortest link next to the ground), the slide's row reads "②④ Double Crank, ③ Rocker". The standard result, and what the simulation shows, is a **crank–rocker**: the shortest side link is the crank and the opposite side link rocks. The video uses the crank–rocker result.

## Files

```
src/fourbar.py        kinematics (loop-closure solver, branch tracking) + the 7 case definitions
src/render_video.py   renders the video/poster and exports the data below
data/cases.csv|json   lengths, l+s vs p+q, measured motion of every link, per case
data/kinematics/      per-frame joint coordinates (t, input angle, A, B) for each case at 30 fps
output/               the video and poster
```

Ground pivots are at O2 = (0, 0) and O4 = (link 1, 0). A is the joint between links 2 and 3, and B is the joint between links 3 and 4.

## How it works

For each step of the input angle (link 2, or the coupler in the double-rocker case), the moving joint is found by intersecting two circles. That gives two possible assemblies. The solver keeps the one closest to a linear extrapolation of the previous steps, which carries the linkage smoothly through toggle and change-point positions. It uses 40 sub-steps per frame. Link lengths hold to about 1e-15 in every frame.

## Rebuild

```bash
pip install -r requirements.txt   # numpy, matplotlib; ffmpeg must be on PATH
python3 src/render_video.py       # about 2 minutes
```

To change a case, edit its lengths or timing in `CASES` in `src/fourbar.py` and re-run. The labels and the data files update automatically.

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import halfspace_intersection, ConvexHull


# ---- exact geometry of the L-corridor complement ---------------------------
# Base corridor (a=0) = union of two width-1 rectangles:
#   H = {0<=u<=1, -3<=v<=1}  and  V = {-3<=u<=1, 0<=v<=1}
# Its complement (the 4 outer-corner regions) is a union of 4 closed half-planes.
# A sofa point is inside the corridor iff it is NOT in any of these 4 half-planes.
# Therefore the sofa = bounding box  ∩  (complement of all 4 half-planes of every
# placement)  =  a convex polygon.  We build it exactly with halfspace_intersection.

# Each row: [A, B, c, lo, hi]  means  A*u + B*v <= c  AND  lo <= u <= hi.
# The sofa is the intersection of the negations of these (plus the global box).
_BASE_OUTER = np.array([
    [1.0,  1.0,  1.0, -3.0, 1.0],   # u>=1 and v>=1
    [-1.0, 1.0, -1.0, -3.0, 1.0],   # u<=-1 and v>=1
    [-1.0, -1.0, 1.0, -3.0, 1.0],   # u<=-1 and v<=-1
    [1.0, -1.0, 1.0, -3.0, 1.0],    # u>=1 and v<=-1
])
_BOX = np.array([
    [-1.0,  0.0,  1.5, -3.0, 1.5],  # u <= 1.5
    [ 1.0,  0.0,  3.0, -3.0, 1.5],  # u >= -3
    [ 0.0, -1.0,  1.5, -3.0, 1.5],  # v <= 1.5
    [ 0.0,  1.0,  1.5, -3.0, 1.5],  # v >= -1.5
])


def _placement_rows(x, y, a_deg):
    """Return the 4 outer-corner half-plane rows for a placement at (x,y,a)."""
    a = np.deg2rad(a_deg)
    c, s = np.cos(a), np.sin(a)
    # R(-a) applied to the base outer-corner normals/offsets.
    A = np.array([
        [c + s,  s - c, x + y - c - s],
        [-c - s, s - c, -x - y + c + s],
        [-c + s, -s - c, -x + y - c + s],
        [c - s,  -s - c, x - y + c - s],
    ])
    return np.hstack([A, np.tile(_BASE_OUTER[:, 3:5], (4, 1))])


def _sofa_area(rows):
    """Exact area of the convex polygon = box ∩ (complement of all `rows`)."""
    H = np.vstack([_BOX, rows])
    H = np.hstack([H, np.ones((H.shape[0], 1))])
    try:
        poly = halfspace_intersection(H)
    except Exception:
        return 0.0
    if poly is None or len(poly) < 3:
        return 0.0
    poly = np.asarray(poly)
    if not np.all(np.isfinite(poly)):
        return 0.0
    # shoelace
    x = poly[:, 0]
    y = poly[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(area)


def _full_rows(angles, x=None, y=None):
    """Assemble all placement rows for the given angle list (and optional coords)."""
    if x is None:
        x = np.full_like(angles, -0.5)
    if y is None:
        y = np.full_like(angles, 0.0)
    rows = [_placement_rows(0.0, 0.0, 0.0)]       # start (0,0,0)
    for xi, yi, ai in zip(x, y, angles):
        rows.append(_placement_rows(xi, yi, ai))
    rows.append(_placement_rows(-2.0, 0.0, 90.0))  # end (-2,0,90)
    return np.vstack(rows)


def _score_angles(angles, x=None, y=None):
    return _sofa_area(_full_rows(angles, x, y))


# ---- the baseline fixed path (score 1.0 reference) -------------------------
_BASELINE_ANGLES = np.array([15.0, 20.0, 34.0, 40.0, 50.0, 55.0, 75.0, 86.0])
_BASELINE_X = np.array([-0.1, -0.2, -0.3, -0.3, -0.4, -0.5, -0.6, -0.7])
_BASELINE_Y = np.array([0.1, 0.2, 0.3, 0.3, 0.3, 0.2, 0.2, 0.3])


def _build_path_from_angles(angles, x=None, y=None):
    if x is None:
        x = np.full_like(angles, -0.5)
    if y is None:
        y = np.full_like(angles, 0.0)
    return np.column_stack([x, y, angles]).astype(np.float64)


def moving_sofa_2d() -> np.ndarray:
    """
    Move a rigid shape through a width-1 L-shaped corridor with one 90° turn.

    The sofa is the intersection of all corridor placements (the evaluator adds the
    start (0,0,0) and end (-2,0,90)).  We express that intersection exactly as a convex
    polygon (bounding box ∩ the complement of each placement's 4 outer-corner
    half-planes) and maximize its area by optimizing the rotation schedule of the
    intermediate placements.  A smoother rotation through the turn widens the
    intersection, so we search the angle sequence with a deterministic hill-climb.
    """
    t_start = time.time()
    deadline = t_start + 600.0
    safety = 10.0  # return this many seconds before the hard limit

    # start from the known-good baseline
    best_angles = _BASELINE_ANGLES.copy()
    best_x = _BASELINE_X.copy()
    best_y = _BASELINE_Y.copy()
    best_score = _score_angles(best_angles, best_x, best_y)

    def time_ok():
        return (time.time() + safety) < deadline

    # ---- coordinate fine-tuning (fixed angles) -----------------------------
    # The intersection is symmetric about y=0 for symmetric x, so y=0 is optimal;
    # we only nudge x.  Try a coarse sweep of a constant x offset.
    for dx in np.arange(-0.6, 0.61, 0.1):
        if not time_ok():
            break
        cand = best_x + dx
        s = _score_angles(best_angles, cand, best_y)
        if s > best_score + 1e-12:
            best_score, best_x = s, cand

    # ---- optimize the rotation schedule -------------------------------------
    # The dominant effect is how the angle ramps from 0 to 90.  Model the n angles
    # as a smooth ramp: angle_i = 90 * f(t_i) with t_i = (i+1)/(n+1).  We optimize
    # the shape of f over a small parameter family, then do local coordinate-wise
    # hill-climbing on the actual angle list.

    def ramp_alpha(alpha):
        t = (np.arange(1, len(best_angles) + 1) / (len(best_angles) + 1))
        # f(t) = t^alpha  (alpha>1 -> lingers near 0 then accelerates; alpha<1 ->
        # accelerates early).  Keep within (0,90).
        return 90.0 * np.power(t, alpha)

    # coarse search over ramp exponent
    for alpha in np.arange(0.4, 2.21, 0.1):
        if not time_ok():
            break
        cand = ramp_alpha(alpha)
        s = _score_angles(cand)
        if s > best_score + 1e-12:
            best_score, best_angles = s, cand

    # coordinate-wise hill-climb on the angle list (and a light x sweep per step)
    n = len(best_angles)
    step = 2.0
    improved = True
    while improved and time_ok():
        improved = False
        for i in range(n):
            for d in (-step, step):
                if not time_ok():
                    break
                cand = best_angles.copy()
                cand[i] = np.clip(best_angles[i] + d, 2.0, 88.0)
                s = _score_angles(cand)
                if s > best_score + 1e-12:
                    best_score, best_angles = s, cand
                    improved = True
        # halve step once a full pass made no progress (anneal)
        if not improved:
            step *= 0.6
            if step < 0.05:
                break

    # final light coordinate polish around the best angles
    for dx in np.arange(-0.3, 0.31, 0.05):
        if not time_ok():
            break
        cand = best_x + dx
        s = _score_angles(best_angles, cand, best_y)
        if s > best_score + 1e-12:
            best_score, best_x = s, cand

    return _build_path_from_angles(best_angles, best_x, best_y)


# EVOLVE-BLOCK-END

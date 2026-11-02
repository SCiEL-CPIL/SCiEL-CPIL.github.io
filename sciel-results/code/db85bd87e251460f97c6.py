"""Circle packing in a perimeter-4 rectangle: maximize sum of radii for 21 disjoint circles."""
import time
import numpy as np
from scipy.spatial import Delaunay

N = 21
DEADLINE = 110.0  # safety margin under 120s limit

# ---------- geometry helpers ----------

def _compute_penalty(cx, cy, r, W, H, logT, lam):
    """Return (penalty, score) for the current state.

    score = sum(r) - penalty.  penalty encodes overlap and wall violations
    with a soft (log-sum-exp) surrogate so the objective is smooth.
    """
    n = len(r)
    # wall violations: circle must lie inside [0,W]x[0,H]
    wall = np.maximum(
        np.maximum(r - cx, r - cy),
        np.maximum(r - (W - cx), r - (H - cy)),
    )
    # pairwise overlap violations
    d = np.sqrt((cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2)
    np.fill_diagonal(d, np.inf)
    ov = (r[:, None] + r[None, :]) - d
    ov = ov[ov > 0]

    # soft-min over violations (0 if none)
    def softmin(v):
        if v.size == 0:
            return 0.0
        m = np.min(v)
        return m + logT * np.log(np.sum(np.exp((v - m) / logT)))

    pw = softmin(wall)
    po = softmin(ov)
    penalty = lam * (pw + po)
    score = np.sum(r) - penalty
    return penalty, score


def _is_valid(cx, cy, r, W, H, tol=1e-6):
    """Hard validity check: all circles inside rectangle, no overlaps."""
    if np.any(cx < -tol) or np.any(cy < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    d = np.sqrt((cx[:, None] - cx[None, :]) ** 2 + (cy[:, None] - cy[None, :]) ** 2)
    np.fill_diagonal(d, np.inf)
    if np.any((r[:, None] + r[None, :]) > d + tol):
        return False
    return True


def _initial_state(rng):
    """Random feasible-ish starting state."""
    a = rng.uniform(0.8, 1.25)          # aspect ratio W/H
    H = 2.0 / (1.0 + a)
    W = a * H
    cx = rng.uniform(0.05, W - 0.05, N)
    cy = rng.uniform(0.05, H - 0.05, N)
    r = rng.uniform(0.04, 0.12, N)
    return cx, cy, r, W, H


def _anneal(cx, cy, r, W, H, T0, T1, iters, rng, lam, logT, deadline):
    """Run simulated annealing. Returns best (cx,cy,r,W,H,score) found."""
    best = None
    best_score = -np.inf

    for it in range(iters):
        if time.time() > deadline:
            break
        # --- propose a move ---
        mtype = rng.random()
        if mtype < 0.25:
            i = rng.integers(N)
            cx[i] += rng.normal(0, 0.05)
        elif mtype < 0.50:
            i = rng.integers(N)
            cy[i] += rng.normal(0, 0.05)
        elif mtype < 0.70:
            i = rng.integers(N)
            r[i] = max(0.02, r[i] * (1.0 + rng.normal(0, 0.08)))
        elif mtype < 0.85:
            # adjust aspect ratio
            a = W / H
            a *= (1.0 + rng.normal(0, 0.03))
            a = min(max(a, 0.6), 1.4)
            H = 2.0 / (1.0 + a)
            W = a * H
            # rescale positions to stay in box
            cx *= (W / (W))  # keep; clip below
        else:
            # global jitter of one circle
            i = rng.integers(N)
            cx[i] = rng.uniform(0.05, W - 0.05)
            cy[i] = rng.uniform(0.05, H - 0.05)

        # clip positions into box
        cx = np.clip(cx, 0.0, W)
        cy = np.clip(cy, 0.0, H)
        r = np.clip(r, 0.02, min(W, H) / 2)

        # --- evaluate ---
        pen, score = _compute_penalty(cx, cy, r, W, H, logT, lam)

        if score > best_score:
            best_score = score
            best = (cx.copy(), cy.copy(), r.copy(), W, H, score)

        # --- accept/reject ---
        # compare to current tracked score
        if it == 0:
            cur_score = score
        else:
            cur_score = last_score
        dE = score - cur_score
        if dE >= 0 or rng.random() < np.exp(dE / T):
            last_score = score
        else:
            # revert: recompute from best is expensive; just continue (state already mutated)
            # To keep it simple, we accept the mutation always after first eval (annealing on score)
            last_score = score

        # temperature schedule
        T = T0 * (T1 / T0) ** (it / max(iters, 1))

    return best


def _polish(cx, cy, r, W, H, rng, deadline):
    """Local refinement: reduce overlaps by nudging circles apart."""
    cur_pen, cur_score = _compute_penalty(cx, cy, r, W, H, 1e-3, 1.0)
    for _ in range(20000):
        if time.time() > deadline:
            break
        i = rng.integers(N)
        cx[i] += rng.normal(0, 0.02)
        cy[i] += rng.normal(0, 0.02)
        cx = np.clip(cx, 0.0, W)
        cy = np.clip(cy, 0.0, H)
        pen, score = _compute_penalty(cx, cy, r, W, H, 1e-3, 1.0)
        if score > cur_score:
            cur_score = score
            cur_pen = pen
    return cx, cy, r, W, H, cur_score


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    rng = np.random.default_rng(12345)

    lam = 100.0
    logT = 0.01
    T0 = 0.1
    T1 = 1e-4

    best_overall = None
    best_overall_score = -np.inf

    # multiple restarts with annealing
    restart = 0
    while time.time() < deadline - 20.0:
        restart += 1
        cx, cy, r, W, H = _initial_state(rng)
        iters = 20000 if restart == 1 else 12000
        res = _anneal(cx, cy, r, W, H, T0, T1, iters, rng, lam, logT, deadline)
        if res is not None:
            cx, cy, r, W, H, score = res
            # polish
            cx, cy, r, W, H, score = _polish(cx, cy, r, W, H, rng, deadline)
            if score > best_overall_score:
                best_overall_score = score
                best_overall = (cx, cy, r, W, H, score)
                # try to fix feasibility: shrink radii slightly if invalid
                if not _is_valid(cx, cy, r, W, H):
                    # scale radii down until valid (approx)
                    scale = 0.95
                    for _ in range(20):
                        if _is_valid(cx, cy, r * scale, W, H):
                            r = r * scale
                            break
                        scale *= 0.95
                    r = r * scale

    if best_overall is None:
        # fallback: seed-style grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    cx, cy, r, W, H, score = best_overall

    # Final safety: ensure validity, shrink if needed
    if not _is_valid(cx, cy, r, W, H):
        scale = 1.0
        for _ in range(50):
            if _is_valid(cx, cy, r * scale, W, H):
                break
            scale *= 0.95
        r = r * scale

    out = np.column_stack([cx, cy, r])
    return out

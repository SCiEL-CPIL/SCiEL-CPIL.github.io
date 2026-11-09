# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from scipy.optimize import linprog


def _check_validity(centers, sides, tol=1e-7):
    """Return (ok, max_violation)."""
    n = len(sides)
    if centers.shape != (n, 2):
        return False, np.inf
    if not (np.all(np.isfinite(centers)) and np.all(np.isfinite(sides))):
        return False, np.inf
    if np.any(sides <= 0):
        return False, np.inf
    # inside unit square
    viol = 0.0
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        lo_x = cx - s / 2.0
        hi_x = cx + s / 2.0
        lo_y = cy - s / 2.0
        hi_y = cy + s / 2.0
        if lo_x < -tol:
            viol = max(viol, -tol - lo_x)
        if hi_x > 1.0 + tol:
            viol = max(viol, hi_x - (1.0 + tol))
        if lo_y < -tol:
            viol = max(viol, -tol - lo_y)
        if hi_y > 1.0 + tol:
            viol = max(viol, hi_y - (1.0 + tol))
    # pairwise non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2.0 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            if dx < t and dy < t:
                # overlap; measure how far inside the forbidden region
                # violation ~ max(0, t - dx, t - dy) style; use min margin
                viol = max(viol, min(t - dx, t - dy) if (t - dx) > 0 and (t - dy) > 0 else 0.0)
    return viol < 1e-9, viol


def _solve_lp_for_orientation(n, orient, time_budget=None):
    """
    orient: dict mapping (i,j) with i<j to one of 0..3:
      0: i left of j   (x_i + s_i/2 <= x_j - s_j/2)
      1: j left of i   (x_j + s_j/2 <= x_i - s_i/2)
      2: i below j     (y_i + s_i/2 <= y_j - s_j/2)
      3: j below i     (y_j + s_j/2 <= y_i - s_i/2)
    Returns (sum_s, centers, sides) or None if infeasible.
    Variables: x_i, y_i, s_i for i in 0..n-1. Order: x_0..x_{n-1}, y_0..y_{n-1}, s_0..s_{n-1}.
    """
    m = 3 * n
    # Objective: maximize sum s_i  -> minimize -sum s_i
    c = np.zeros(m)
    for i in range(n):
        c[2 * n + i] = -1.0

    # Bounds
    lb = np.zeros(m)
    ub = np.full(m, np.inf)
    for i in range(n):
        # 0 <= x_i <= 1, 0 <= y_i <= 1, 0 <= s_i <= 1
        ub[i] = 1.0
        ub[n + i] = 1.0
        ub[2 * n + i] = 1.0

    A_ub = []
    b_ub = []

    # Inside unit square: x_i - s_i/2 >= 0  ->  -x_i + s_i/2 <= 0
    for i in range(n):
        row = np.zeros(m)
        row[i] = -1.0
        row[2 * n + i] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)
        row = np.zeros(m)
        row[i] = 1.0
        row[2 * n + i] = 0.5
        A_ub.append(row)
        b_ub.append(1.0)
        row = np.zeros(m)
        row[n + i] = -1.0
        row[2 * n + i] = 0.5
        A_ub.append(row)
        b_ub.append(0.0)
        row = np.zeros(m)
        row[n + i] = 1.0
        row[2 * n + i] = 0.5
        A_ub.append(row)
        b_ub.append(1.0)

    # Orientation constraints
    for (i, j), o in orient.items():
        if o == 0:  # x_i + s_i/2 <= x_j - s_j/2  -> x_i - x_j + s_i/2 + s_j/2 <= 0
            row = np.zeros(m)
            row[i] = 1.0
            row[j] = -1.0
            row[2 * n + i] = 0.5
            row[2 * n + j] = 0.5
            A_ub.append(row)
            b_ub.append(0.0)
        elif o == 1:  # x_j + s_j/2 <= x_i - s_i/2  -> x_j - x_i + s_j/2 + s_i/2 <= 0
            row = np.zeros(m)
            row[i] = -1.0
            row[j] = 1.0
            row[2 * n + i] = 0.5
            row[2 * n + j] = 0.5
            A_ub.append(row)
            b_ub.append(0.0)
        elif o == 2:  # y_i + s_i/2 <= y_j - s_j/2
            row = np.zeros(m)
            row[n + i] = 1.0
            row[n + j] = -1.0
            row[2 * n + i] = 0.5
            row[2 * n + j] = 0.5
            A_ub.append(row)
            b_ub.append(0.0)
        elif o == 3:  # y_j + s_j/2 <= y_i - s_i/2
            row = np.zeros(m)
            row[n + i] = -1.0
            row[n + j] = 1.0
            row[2 * n + i] = 0.5
            row[2 * n + j] = 0.5
            A_ub.append(row)
            b_ub.append(0.0)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=list(zip(lb, ub)),
                      method='highs', options={'presolve': True, 'dual_feasibility_tolerance': 1e-9,
                                                'primal_feasibility_tolerance': 1e-9})
    except Exception:
        return None
    if res is None or not res.success:
        return None
    x = res.x
    centers = np.empty((n, 2))
    sides = np.empty(n)
    for i in range(n):
        centers[i, 0] = x[i]
        centers[i, 1] = x[n + i]
        sides[i] = x[2 * n + i]
    # clamp tiny negatives from solver noise
    sides = np.maximum(sides, 0.0)
    return float(np.sum(sides)), centers, sides


def _orientation_from_topological_order(n, order, rng):
    """
    Given a permutation 'order' of [0..n-1], build a valid (acyclic) orientation.
    For each pair (i,j), i<j: if i appears before j in order, we may orient the
    constraint in a way that respects the order. We choose randomly among the
    4 orientations but bias to keep the graph acyclic by using the order as a
    topological order: an edge a->b is allowed only if pos[a] < pos[b].
    For pair (i,j):
      orientation 0 (i left of j): edge i->j, need pos[i]<pos[j]
      orientation 1 (j left of i): edge j->i, need pos[j]<pos[i]
      orientation 2 (i below j): edge i->j, need pos[i]<pos[j]
      orientation 3 (j below i): edge j->i, need pos[j]<pos[i]
    So we pick orientation consistent with the order, then random axis.
    """
    pos = {v: i for i, v in enumerate(order)}
    orient = {}
    for i in range(n):
        for j in range(i + 1, n):
            if pos[i] < pos[j]:
                # edges i->j allowed: orientation 0 (x) or 2 (y)
                orient[(i, j)] = rng.choice([0, 2])
            else:
                # edges j->i allowed: orientation 1 (x) or 3 (y)
                orient[(i, j)] = rng.choice([1, 3])
    return orient


def squares_in_square10() -> tuple:
    n = 10
    t0 = time.time()
    DEADLINE = 600.0
    SAFETY = 20.0  # seconds to reserve
    rng = random.Random(20240601)

    # Baseline grid (always valid)
    k = int(np.ceil(np.sqrt(n)))
    side0 = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx0 = (cols + 0.5) * side0
    cy0 = (rows + 0.5) * side0
    best_centers = np.stack([cx0, cy0], axis=1).astype(np.float64)
    best_sides = np.full(n, side0, dtype=np.float64)
    best_sum = float(np.sum(best_sides))

    # --- Phase 1: enumerate diverse acyclic orientations, solve LP for each ---
    max_iters = 4000
    it = 0
    while it < max_iters:
        if time.time() - t0 > DEADLINE - SAFETY:
            break
        order = list(range(n))
        rng.shuffle(order)
        orient = _orientation_from_topological_order(n, order, rng)
        out = _solve_lp_for_orientation(n, orient)
        if out is not None:
            ssum, centers, sides = out
            if ssum > best_sum + 1e-12:
                best_sum = ssum
                best_centers = centers.copy()
                best_sides = sides.copy()
        it += 1

    # --- Phase 2: local search / perturbation refinement on best solution ---
    # Use scipy minimize on the sum of side lengths with penalty for violations,
    # starting from best LP solution, to polish (smooth surrogate).
    if time.time() - t0 < DEADLINE - 30:
        from scipy.optimize import minimize

        def penalty_objective(v):
            # v: 3n vector [x,y,s]
            x = v[0:n]
            y = v[n:2*n]
            s = v[2*n:3*n]
            # objective: -sum s (minimize)
            obj = -np.sum(s)
            # penalty for boundary violations
            pen = 0.0
            for i in range(n):
                pen += max(0.0, -(x[i] - s[i] / 2.0)) ** 2
                pen += max(0.0, (x[i] + s[i] / 2.0) - 1.0) ** 2
                pen += max(0.0, -(y[i] - s[i] / 2.0)) ** 2
                pen += max(0.0, (y[i] + s[i] / 2.0) - 1.0) ** 2
                pen += max(0.0, -s[i]) ** 2
            # penalty for overlaps
            for i in range(n):
                for j in range(i + 1, n):
                    t = (s[i] + s[j]) / 2.0 - 1e-7
                    dx = abs(x[i] - x[j])
                    dy = abs(y[i] - y[j])
                    if dx < t and dy < t:
                        pen += (min(t - dx, t - dy)) ** 2 * 100.0
            return obj + 1e4 * pen

        v0 = np.concatenate([best_centers[:, 0], best_centers[:, 1], best_sides])
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0))  # x
        for i in range(n):
            bounds.append((0.0, 1.0))  # y
        for i in range(n):
            bounds.append((0.0, 1.0))  # s

        best_local = None
        best_local_val = best_sum
        # multiple restarts from perturbed best
        n_restarts = 8
        for r in range(n_restarts):
            if time.time() - t0 > DEADLINE - 15:
                break
            if r == 0:
                start = v0.copy()
            else:
                start = v0 + rng.gauss(0, 0.02, size=3 * n)
                start = np.clip(start, 0.0, 1.0)
            try:
                res = minimize(penalty_objective, start, method='L-BFGS-B',
                               bounds=bounds,
                               options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            except Exception:
                continue
            v = res.x
            x = v[0:n]
            y = v[n:2*n]
            s = np.maximum(v[2*n:3*n], 0.0)
            centers = np.stack([x, y], axis=1)
            ok, viol = _check_validity(centers, s)
            if ok:
                ssum = float(np.sum(s))
                if ssum > best_local_val + 1e-12:
                    best_local_val = ssum
                    best_local = (centers, s)
        if best_local is not None and best_local_val > best_sum + 1e-9:
            best_centers, best_sides = best_local
            best_sum = best_local_val

    # Final safety check; fall back to baseline if invalid
    ok, _ = _check_validity(best_centers, best_sides)
    if not ok:
        # recompute baseline
        k = int(np.ceil(np.sqrt(n)))
        side0 = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx0 = (cols + 0.5) * side0
        cy0 = (rows + 0.5) * side0
        best_centers = np.stack([cx0, cy0], axis=1).astype(np.float64)
        best_sides = np.full(n, side0, dtype=np.float64)

    return best_centers.astype(np.float64), best_sides.astype(np.float64)


# EVOLVE-BLOCK-END

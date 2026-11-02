"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles.
Strategy: Delaunay contact-graph constrained optimization with SLSQP,
alternating grow-and-relax phases.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 26
EPS = 1e-9
DEADLINE = 145.0  # safety margin below 150s kill


def _max_radius(pos, r):
    """Maximal radius for each circle given current positions and radii."""
    n = len(r)
    mr = np.empty(n)
    for i in range(n):
        dmin = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        for j in range(n):
            if i != j:
                dmin = min(dmin, np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1]) - r[j])
        mr[i] = max(0.0, dmin)
    return mr


def _valid_check(arr, tol=1e-7):
    """Check all circles are in-square and non-overlapping."""
    for i in range(len(arr)):
        x, y, r = arr[i]
        if r < 0 or x < r + tol or x > 1 - r - tol or y < r + tol or y > 1 - r - tol:
            return False
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            d = np.hypot(arr[i, 0] - arr[j, 0], arr[i, 1] - arr[j, 1])
            if d < arr[i, 2] + arr[j, 2] - tol:
                return False
    return True


def _delaunay_edges(pos):
    """Return set of (i,j) pairs that are Delaunay neighbors."""
    try:
        tri = Delaunay(pos)
        edges = set()
        for simplex in tri.simplices:
            for a in range(3):
                for b in range(a + 1, 3):
                    i, j = simplex[a], simplex[b]
                    edges.add((min(i, j), max(i, j)))
        return edges
    except Exception:
        # Fallback: all pairs
        return [(i, j) for i in range(N) for j in range(i + 1, N)]


def _build_constraints(pos):
    """Build constraint functions for SLSQP based on Delaunay edges + walls."""
    edges = _delaunay_edges(pos)
    cons = []
    # Wall constraints: r_i - x_i >= 0, x_i + r_i - 1 <= 0, etc.
    # We'll encode as: r_i - x_i >= 0  =>  r_i - x_i
    #                  x_i + r_i - 1 <= 0  =>  -(x_i + r_i - 1) = 1 - x_i - r_i
    #                  r_i - y_i >= 0
    #                  1 - y_i - r_i >= 0
    for i in range(N):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*N + i] - x[2*i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*N + i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*N + i] - x[2*i+1]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*N + i]})
    # Delaunay edge separation: r_i + r_j - dist(i,j) <= 0  =>  dist - r_i - r_j >= 0
    for (i, j) in edges:
        cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j:
                     np.hypot(x[2*i] - x[2*j], x[2*i+1] - x[2*j+1]) - x[2*N+i] - x[2*N+j]})
    return cons, edges


def _slsqp_step(x0, maxiter=200):
    """One SLSQP optimization step with Delaunay-based constraints."""
    pos0 = x0[:2*N].reshape(N, 2)
    cons, edges = _build_constraints(pos0)

    def obj(x):
        return -np.sum(x[2*N:])

    def jac(x):
        g = np.zeros_like(x)
        g[2*N:] = -1.0
        return g

    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
        bounds.append((0.0, 1.0))  # y
    for i in range(N):
        bounds.append((0.0, 0.5))  # radius

    try:
        res = minimize(obj, x0, jac=jac, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': maxiter, 'ftol': 1e-9})
        return res.x
    except Exception:
        return x0


def _grow_and_relax(pos, rad, time_budget=5.0):
    """Iterate: maximize radii, re-triangulate, optimize positions with SLSQP."""
    t0 = time.time()
    for it in range(50):
        if time.time() - t0 > time_budget:
            break
        # Grow: maximize radii given fixed positions
        rad = _max_radius(pos, rad)
        # Build x vector
        x = np.concatenate([pos.reshape(-1), rad])
        # Relax: SLSQP with Delaunay constraints
        x_new = _slsqp_step(x, maxiter=100)
        pos_new = x_new[:2*N].reshape(N, 2)
        rad_new = np.maximum(x_new[2*N:], 0.0)
        # Check if improved
        if np.sum(rad_new) > np.sum(rad) - 1e-10:
            pos, rad = pos_new, rad_new
        else:
            # Try smaller step: just re-compute radii
            rad = _max_radius(pos, rad)
    return pos, rad


def _polish(pos, rad):
    """Final polish: alternate radius growth and position optimization."""
    t_start = time.time()
    for it in range(40):
        if time.time() > DEADLINE - 3:
            break
        rad = _max_radius(pos, rad)
        # SLSQP step
        x = np.concatenate([pos.reshape(-1), rad])
        x_new = _slsqp_step(x, maxiter=150)
        pos_new = x_new[:2*N].reshape(N, 2)
        rad_new = np.maximum(x_new[2*N:], 0.0)
        if _valid_check(np.column_stack([pos_new, rad_new])):
            pos, rad = pos_new, rad_new
        rad = _max_radius(pos, rad)
    rad = _max_radius(pos, rad)
    return pos, rad


def construct():
    t_start = time.time()
    rng = np.random.default_rng(42)
    best = None
    best_score = -1.0

    def consider(pos, rad):
        nonlocal best, best_score
        arr = np.column_stack([pos, rad])
        if not _valid_check(arr):
            # Try to fix: reduce radii slightly
            rad = np.maximum(rad * 0.999, 0)
            arr = np.column_stack([pos, rad])
            if not _valid_check(arr):
                return
        s = float(arr[:, 2].sum())
        if s > best_score:
            best_score = s
            best = arr

    # ---- Phase A: Generate diverse initializations ----
    starts = []

    # Grid initializations
    for side in range(3, 9):
        if time.time() > DEADLINE - 30:
            break
        cells = []
        for i in range(side):
            for j in range(side):
                cells.append(((i + 0.5) / side, (j + 0.5) / side))
        cells = cells[:N]
        pos = np.array(cells, dtype=float)
        rad = np.full(N, 0.5 / side - 1e-4)
        for _ in range(10):
            rad = _max_radius(pos, rad)
        starts.append((pos.copy(), rad.copy()))

        # Perturbed grids
        for k in range(2):
            if time.time() > DEADLINE - 30:
                break
            p0 = pos + 0.05 * (rng.random((N, 2)) * 2 - 1)
            p0 = np.clip(p0, 0.05, 0.95)
            r0 = np.full(N, 0.5 / side)
            r0 = _max_radius(p0, r0)
            starts.append((p0, r0))

    # Hexagonal packing initialization
    if time.time() <= DEADLINE - 30:
        hex_pos = []
        hex_rad = []
        # Hexagonal lattice
        r_hex = 0.12
        dx = 2 * r_hex
        dy = r_hex * np.sqrt(3)
        for row in range(8):
            for col in range(8):
                x = col * dx + (row % 2) * dx / 2 + 0.15
                y = row * dy + 0.15
                if 0.05 < x < 0.95 and 0.05 < y < 0.95:
                    hex_pos.append((x, y))
        hex_pos = hex_pos[:N]
        if len(hex_pos) >= N:
            hp = np.array(hex_pos, dtype=float)
            hr = _max_radius(hp, np.full(N, 0.1))
            starts.append((hp, hr))

    # Random initializations
    for k in range(8):
        if time.time() > DEADLINE - 30:
            break
        p0 = rng.random((N, 2)) * 0.8 + 0.1
        r0 = np.full(N, 0.05)
        r0 = _max_radius(p0, r0)
        starts.append((p0, r0))

    # ---- Phase B: Grow-and-relax from each start ----
    for idx, (pos0, rad0) in enumerate(starts):
        if time.time() > DEADLINE - 20:
            break
        budget = min(8.0, max(2.0, (DEADLINE - 20 - time.time()) / max(1, len(starts) - idx)))
        pos_r, rad_r = _grow_and_relax(pos0, rad0, time_budget=budget)
        consider(pos_r, rad_r)

    # ---- Phase C: Polish the best solution found ----
    if best is not None:
        pos_b = best[:, :2].copy()
        rad_b = best[:, 2].copy()
        pos_p, rad_p = _polish(pos_b, rad_b)
        consider(pos_p, rad_p)

    # ---- Phase D: Local search perturbation on best ----
    if best is not None and time.time() < DEADLINE - 10:
        for k in range(10):
            if time.time() > DEADLINE - 5:
                break
            pos_p = best[:, :2].copy()
            rad_p = best[:, 2].copy()
            # Perturb a random subset
            n_perturb = rng.integers(3, 8)
            idx_perturb = rng.choice(N, size=n_perturb, replace=False)
            pos_p[idx_perturb] += 0.02 * (rng.random((n_perturb, 2)) * 2 - 1)
            pos_p = np.clip(pos_p, 0.05, 0.95)
            rad_p = _max_radius(pos_p, rad_p)
            pos_f, rad_f = _polish(pos_p, rad_p)
            consider(pos_f, rad_f)

    if best is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    return best


if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)

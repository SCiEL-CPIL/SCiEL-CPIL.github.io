"""Circle packing in unit square: 26 circles, maximize sum of radii.
Combined strategy:
  - Analytic gradient penalty optimization (from inspiration)
  - Delaunay contact-graph pruning for speed (from current)
  - Voronoi/Hex/Grid diverse initializations
  - Iterative shake-and-refine to escape local minima
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy.spatial import Voronoi

N = 26
DEADLINE = 145.0  # safety margin below 150s kill
EPS = 1e-8


def _valid_check(arr, tol=1e-7):
    """Check all circles are in-square and non-overlapping."""
    n = len(arr)
    for i in range(n):
        x, y, r = arr[i]
        if r < -tol:
            return False
        if x < r + tol or x > 1 - r - tol or y < r + tol or y > 1 - r - tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
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
        return list(edges)
    except Exception:
        # Fallback to all pairs if Delaunay fails (e.g., coincident points)
        return [(i, j) for i in range(N) for j in range(i + 1, N)]


def _objective_and_grad(params, penalty=1000.0, edges=None):
    """
    Penalty objective and analytic gradient.
    params: shape (3*N,) -> [x0, y0, r0, x1, y1, r1, ...]
    edges: list of (i,j) pairs to check for overlap. If None, checks all pairs.
    """
    p = params.reshape(N, 3)
    x = p[:, 0]
    y = p[:, 1]
    r = p[:, 2]

    # Objective: -sum(r) + penalty * violations
    score = -np.sum(r)

    # Wall boundary violations (always active)
    # x < r -> r - x > 0
    # x > 1-r -> x + r - 1 > 0
    wx_neg = np.maximum(0.0, r - x)
    wx_pos = np.maximum(0.0, x + r - 1.0)
    wy_neg = np.maximum(0.0, r - y)
    wy_pos = np.maximum(0.0, y + r - 1.0)
    wall_pen = np.sum(wx_neg) + np.sum(wx_pos) + np.sum(wy_neg) + np.sum(wy_pos)
    
    score += penalty * wall_pen

    # Gradient
    g = np.zeros_like(params)
    # d(-sum r)/dr = -1
    g[2::3] = -1.0

    # Wall gradients
    for i in range(N):
        if r[i] - x[i] > 0:
            g[3*i] += penalty * (-1.0)  # d(r-x)/dx
            g[3*i+2] += penalty * (+1.0) # d(r-x)/dr
        if x[i] + r[i] - 1.0 > 0:
            g[3*i] += penalty * (+1.0)   # d(x+r-1)/dx
            g[3*i+2] += penalty * (+1.0)
        if r[i] - y[i] > 0:
            g[3*i+1] += penalty * (-1.0)
            g[3*i+2] += penalty * (+1.0)
        if y[i] + r[i] - 1.0 > 0:
            g[3*i+1] += penalty * (+1.0)
            g[3*i+2] += penalty * (+1.0)

    # Pairwise overlap penalties
    if edges is None:
        iu = np.triu_indices(N, k=1)
    else:
        iu = np.array(edges).T  # shape (2, num_edges)
        # Ensure i < j for consistency, though math is symmetric
        iu = iu[np.argsort(iu, axis=1).T] 
        # Unique them just in case Delaunay returned duplicates (it shouldn't)
        iu = np.unique(iu, axis=1)
        if iu.shape[1] == 0:
            return score, g
        iu = (iu[0], iu[1])

    if len(iu[0]) > 0:
        dx = x[iu[0]] - x[iu[1]]
        dy = y[iu[0]] - y[iu[1]]
        dist = np.sqrt(dx * dx + dy * dy)
        dist_safe = np.maximum(dist, 1e-10)
        
        # Overlap amount: (r_i + r_j) - dist
        ov = (r[iu[0]] + r[iu[1]]) - dist
        active = ov > 0
        
        if np.any(active):
            ax = iu[0][active]
            ay = iu[1][active]
            adx = dx[active] / dist_safe[active]
            ady = dy[active] / dist_safe[active]
            
            # Add to objective
            score += penalty * np.sum(ov[active])
            
            # Add to gradient
            # d(ov)/dx_i = -adx, d(ov)/dx_j = +adx
            # d(ov)/dr_i = +1, d(ov)/dr_j = +1
            for k in range(len(ax)):
                i = ax[k]
                j = ay[k]
                g[3*i] += penalty * (-adx[k])
                g[3*j] += penalty * (+adx[k])
                g[3*i+1] += penalty * (-ady[k])
                g[3*j+1] += penalty * (+ady[k])
                g[3*i+2] += penalty * (+1.0)
                g[3*j+2] += penalty * (+1.0)

    return score, g


def _enforce_validity(params):
    """Shrink radii and clip to make a configuration valid."""
    p = params.reshape(N, 3).copy()
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    p[:, 2] = np.clip(p[:, 2], 1e-6, 0.5)
    
    # Clamp to walls
    wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                      np.minimum(p[:, 1], 1.0 - p[:, 1]))
    p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
    
    # Iteratively resolve overlaps
    for _ in range(50):
        iu = np.triu_indices(N, k=1)
        dx = p[iu[0], 0] - p[iu[1], 0]
        dy = p[iu[0], 1] - p[iu[1], 1]
        dist = np.sqrt(dx * dx + dy * dy)
        overlap = (p[iu[0], 2] + p[iu[1], 2]) - dist
        active = overlap > 0
        if not np.any(active):
            break
        for k in range(len(active)):
            if not active[k]:
                continue
            i, j = iu[0][k], iu[1][k]
            ov = overlap[k]
            if p[i, 2] <= p[j, 2]:
                sh = min(ov, p[i, 2] - 1e-6)
                p[i, 2] -= sh
                rem = ov - sh
                if rem > 0:
                    p[j, 2] -= min(rem, p[j, 2] - 1e-6)
            else:
                sh = min(ov, p[j, 2] - 1e-6)
                p[j, 2] -= sh
                rem = ov - sh
                if rem > 0:
                    p[i, 2] -= min(rem, p[i, 2] - 1e-6)
        # Re-clamp to walls
        wall = np.minimum(np.minimum(p[:, 0], 1.0 - p[:, 0]),
                          np.minimum(p[:, 1], 1.0 - p[:, 1]))
        p[:, 2] = np.minimum(p[:, 2], np.maximum(wall, 1e-6))
    return p


def _initial_grid(seed):
    rng = np.random.RandomState(seed)
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            cells.append(((i + 0.5) / side, (j + 0.5) / side))
    cells = cells[:N]
    pos = np.array(cells, dtype=float)
    # Small jitter
    pos += rng.uniform(-0.01, 0.01, size=pos.shape)
    pos = np.clip(pos, 0.05, 0.95)
    # Initial radii: half min dist to neighbor or wall
    radii = np.zeros(N)
    for i in range(N):
        dmin = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
        for j in range(N):
            if i != j:
                dmin = min(dmin, np.hypot(pos[i,0]-pos[j,0], pos[i,1]-pos[j,1]))
        radii[i] = 0.4 * dmin
    return np.column_stack([pos, radii])

def _initial_hex(seed):
    rng = np.random.RandomState(seed)
    points = []
    # Generate a hex grid larger than needed, then crop
    for i in range(8):
        for j in range(8):
            x = i * 0.25 + (0.125 if i % 2 else 0)
            y = j * 0.22
            points.append((x, y))
    points = np.array(points)
    # Scale to fit in [0,1]
    points -= points.min(axis=0)
    points /= points.max(axis=0)
    points *= 0.9
    points += 0.05
    
    # Select N closest to center
    d_center = np.sum((points - 0.5)**2, axis=1)
    idx = np.argsort(d_center)[:N]
    pos = points[idx]
    pos += rng.uniform(-0.01, 0.01, size=pos.shape)
    pos = np.clip(pos, 0.05, 0.95)
    
    radii = np.zeros(N)
    for i in range(N):
        dmin = min(pos[i,0], 1-pos[i,0], pos[i,1], 1-pos[i,1])
        for j in range(N):
            if i != j:
                dmin = min(dmin, np.hypot(pos[i,0]-pos[j,0], pos[i,1]-pos[j,1]))
        radii[i] = 0.4 * dmin
    return np.column_stack([pos, radii])

def _initial_random(seed):
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, size=(N, 2))
    radii = np.full(N, 0.02)
    return np.column_stack([pos, radii])

def _initial_voronoi(seed):
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, size=(N, 2))
    try:
        vor = Voronoi(pos)
        radii = np.zeros(N)
        for i, p in enumerate(pos):
            # Distance to walls
            wall_dist = min(p[0], 1-p[0], p[1], 1-p[1])
            # Distance to nearest neighbor
            dists = np.linalg.norm(pos - p, axis=1)
            dists[i] = np.inf
            min_dist_neighbor = np.min(dists)
            radii[i] = 0.4 * min(wall_dist, min_dist_neighbor)
    except Exception:
        radii = np.full(N, 0.02)
    return np.column_stack([pos, radii])


def _refine(params, time_budget=5.0):
    """
    Refine a configuration using L-BFGS-B with analytic gradients.
    Dynamically updates Delaunay edges.
    """
    t0 = time.time()
    best_params = params.copy()
    best_score = -np.sum(params[2::3]) # We want to maximize sum(r), minimize -sum(r)
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((1e-6, 0.5)) # r

    # Escalating penalty weights
    penalties = [100.0, 1000.0, 10000.0]
    
    current_params = params.copy()
    
    for pen in penalties:
        if time.time() - t0 > time_budget:
            break
            
        # Compute edges for current best
        pos = current_params.reshape(N, 3)[:, :2]
        edges = _delaunay_edges(pos)
        
        # Define objective with closure for edges and penalty
        def obj_grad(p):
            val, grad = _objective_and_grad(p, penalty=pen, edges=edges)
            return val, grad
            
        try:
            res = minimize(obj_grad, current_params, jac=True, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success:
                # Check if it's valid
                arr = res.x.reshape(N, 3)
                if _valid_check(arr):
                    s = float(np.sum(arr[:, 2]))
                    if s > best_score:
                        best_score = s
                        best_params = res.x.copy()
                        current_params = res.x.copy()
                else:
                    # If not valid, try to enforce validity and see if it's better
                    valid_arr = _enforce_validity(res.x)
                    if _valid_check(valid_arr):
                        s = float(np.sum(valid_arr[:, 2]))
                        if s > best_score:
                            best_score = s
                            best_params = valid_arr.flatten().copy()
                            current_params = valid_arr.flatten().copy()
        except Exception:
            pass

    return best_params


def construct():
    t_start = time.time()
    rng = np.random.default_rng(42)
    best = None
    best_score = -1.0

    def consider(arr):
        nonlocal best, best_score
        if not _valid_check(arr):
            arr = _enforce_validity(arr)
            if not _valid_check(arr):
                return
        s = float(arr[:, 2].sum())
        if s > best_score:
            best_score = s
            best = arr.copy()

    # Generate initial candidates
    inits = []
    # Grid
    for k in range(3):
        inits.append(_initial_grid(42 + k))
    # Hex
    for k in range(3):
        inits.append(_initial_hex(42 + k))
    # Random
    for k in range(5):
        inits.append(_initial_random(42 + k))
    # Voronoi
    for k in range(3):
        inits.append(_initial_voronoi(42 + k))

    # Phase 1: Refine each initial guess
    for idx, init in enumerate(inits):
        if time.time() > DEADLINE - 20:
            break
        # Allocate time based on remaining budget
        remaining = DEADLINE - 20 - time.time()
        num_left = len(inits) - idx
        budget = max(1.0, remaining / num_left)
        
        refined = _refine(init.flatten(), time_budget=budget)
        consider(refined.reshape(N, 3))

    # Phase 2: Local search / Shake and Refine on best
    if best is not None:
        # Try to perturb the best solution and refine
        for k in range(10):
            if time.time() > DEADLINE - 5:
                break
            
            # Perturb positions slightly
            perturbed = best.copy()
            # Randomly select a few circles to move
            n_move = rng.integers(3, 8)
            idx_move = rng.choice(N, size=n_move, replace=False)
            perturbed[idx_move, :2] += 0.01 * (rng.random((n_move, 2)) * 2 - 1)
            perturbed = np.clip(perturbed, 0.0, 1.0)
            
            # Re-estimate radii roughly to ensure they are feasible starting points
            # This helps the optimizer not start in a deeply infeasible state
            for i in range(N):
                dmin = min(perturbed[i,0], 1-perturbed[i,0], perturbed[i,1], 1-perturbed[i,1])
                for j in range(N):
                    if i != j:
                        dmin = min(dmin, np.hypot(perturbed[i,0]-perturbed[j,0], perturbed[i,1]-perturbed[j,1]))
                # Keep current radius if it fits, otherwise shrink
                if perturbed[i,2] > dmin:
                    perturbed[i,2] = max(1e-6, dmin * 0.9)
            
            remaining = DEADLINE - 5 - time.time()
            budget = max(1.0, remaining / 10)
            
            refined = _refine(perturbed.flatten(), time_budget=budget)
            consider(refined.reshape(N, 3))

    if best is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                cells.append(((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-4))
        return np.array(cells[:N])
        
    return best

if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)

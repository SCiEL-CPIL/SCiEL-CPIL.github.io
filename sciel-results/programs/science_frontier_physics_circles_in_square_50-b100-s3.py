"""Fifty equal circles in a square – combined epigraph + active-set SLSQP optimizer."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import cKDTree

N = 50
HALF = 0.5
DEADLINE = 145.0  # Safety margin before 150s kill


def _score_from_centres(c):
    """Compute the exact radius score for given centres (N,2)."""
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    if np.any(c < -HALF + 1e-12) or np.any(c > HALF - 1e-12):
        return 0.0
    d = pdist(c)
    min_pair = d.min() if d.size > 0 else 0.0
    dist_walls = np.min(np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ]), axis=0)
    min_wall = dist_walls.min()
    r = min(min_pair / 2.0, min_wall)
    return r


def _active_constraints(c, r, tol=1e-7):
    """
    Identify active constraints at radius r:
    - Pairs with distance/2 <= r + tol
    - Walls with distance <= r + tol
    Returns (pair_list, wall_list) where pair_list is list of (i,j)
    and wall_list is list of (circle_idx, wall_idx).
    """
    D = cdist(c, c)
    np.fill_diagonal(D, np.inf)
    half_D = D / 2.0
    iu = np.triu_indices(N, k=1)
    pair_dists = half_D[iu]
    active_pairs = [(int(iu[0][k]), int(iu[1][k])) for k in range(len(iu[0]))
                    if pair_dists[k] <= r + tol]
    
    wall_dists = np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ], axis=1)
    active_walls = []
    for i in range(N):
        for w in range(4):
            if wall_dists[i, w] <= r + tol:
                active_walls.append((i, w))
    
    return active_pairs, active_walls


def _active_set_slsqp(c, r_target, max_iter=200):
    """
    SLSQP on the active constraint subset: maximize t subject to
    active pairwise and wall constraints >= t.
    This is a smaller problem than full SLSQP and can refine precisely.
    """
    active_pairs, active_walls = _active_constraints(c, r_target, tol=1e-6)
    
    if len(active_pairs) == 0 and len(active_walls) == 0:
        return c  # No active constraints; nothing to refine
    
    # Build constraint function for active set only
    def cons(x):
        cc = x[:2*N].reshape(N, 2)
        t = x[2*N]
        vals = []
        for (i, j) in active_pairs:
            vals.append(np.linalg.norm(cc[i] - cc[j]) / 2.0 - t)
        for (i, w) in active_walls:
            if w == 0:
                vals.append(cc[i, 0] + HALF - t)
            elif w == 1:
                vals.append(HALF - cc[i, 0] - t)
            elif w == 2:
                vals.append(cc[i, 1] + HALF - t)
            else:
                vals.append(HALF - cc[i, 1] - t)
        return np.array(vals)
    
    def obj(x):
        return -x[2*N]
    
    x0 = np.concatenate([c.ravel(), [r_target]])
    bounds = [(-HALF + 1e-10, HALF - 1e-10)] * (2*N) + [(0.0, HALF)]
    
    try:
        res = minimize(
            obj, x0, method='SLSQP', bounds=bounds,
            constraints=[{'type': 'ineq', 'fun': cons}],
            options={'maxiter': max_iter, 'ftol': 1e-14, 'iprint': 0}
        )
        if res.success:
            cc = np.clip(res.x[:2*N].reshape(N, 2), -HALF + 1e-10, HALF - 1e-10)
            return cc
    except Exception:
        pass
    return c


def _epigraph_objective(x, beta=100.0):
    """Epigraph: minimize -t + penalty for violations."""
    c = x[:2*N].reshape(N, 2)
    t = x[2*N]
    c = np.clip(c, -HALF + 1e-9, HALF - 1e-9)
    d = pdist(c) / 2.0
    wall_dists = np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ], axis=1).flatten()
    all_dists = np.concatenate([d, wall_dists])
    violations = np.maximum(0.0, t - all_dists)
    if violations.size > 0 and violations.max() > 1e-15:
        penalty = np.exp(-beta * violations).sum() / beta
    else:
        penalty = 0.0
    return -t + penalty


def _negative_smooth_score(c_flat, beta=100.0):
    """Soft-min objective for L-BFGS-B."""
    c = c_flat.reshape(N, 2)
    c = np.clip(c, -HALF + 1e-9, HALF - 1e-9)
    d = pdist(c) / 2.0
    wall_dists = np.stack([
        c[:, 0] + HALF,
        HALF - c[:, 0],
        c[:, 1] + HALF,
        HALF - c[:, 1]
    ], axis=1).flatten()
    all_dists = np.concatenate([d, wall_dists])
    min_d = all_dists.min()
    exp_d = np.exp(-beta * (all_dists - min_d))
    soft_min = min_d - np.log(exp_d.sum()) / beta
    return -soft_min


def _grid_init():
    """Grid initial layout."""
    best_r, best_c = 7, 8
    for r in range(5, 10):
        for c in range(5, 10):
            if r * c >= N and abs(r - c) < abs(best_r - best_c):
                best_r, best_c = r, c
    x = np.linspace(-HALF, HALF, best_c)
    y = np.linspace(-HALF, HALF, best_r)
    full_grid = np.array([(xi, yi) for yi in y for xi in x])
    dists = np.sum(full_grid**2, axis=1)
    sorted_idx = np.argsort(dists)
    return full_grid[sorted_idx[:N]]


def _hex_init():
    """Hexagonal packing initial layout."""
    points = []
    r_init = 0.07
    dx = 2 * r_init
    dy = r_init * np.sqrt(3)
    max_x = HALF - r_init
    max_y = HALF - r_init
    row = 0
    y = -max_y
    while y <= max_y:
        x_start = -max_x
        if row % 2 == 1:
            x_start += dx / 2
        x = x_start
        while x <= max_x:
            points.append([x, y])
            x += dx
        y += dy
        row += 1
        if len(points) > N * 2:
            break
    pts = np.array(points)
    dists = np.sum(pts**2, axis=1)
    sorted_idx = np.argsort(dists)
    return pts[sorted_idx[:N]]


def _random_init(rng):
    return rng.uniform(-HALF, HALF, size=(N, 2))


def _optimize_epigraph(x0, beta_start=50.0, beta_end=500.0, n_stages=3):
    """Multi-stage epigraph optimization."""
    c0 = x0.reshape(N, 2)
    r0 = _score_from_centres(c0)
    t0_init = max(r0, 0.01)
    x = np.concatenate([c0.ravel(), [t0_init]])
    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (2 * N) + [(0.0, HALF)]
    betas = np.linspace(beta_start, beta_end, n_stages)
    for beta in betas:
        try:
            res = minimize(
                lambda x_: _epigraph_objective(x_, beta=beta),
                x, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'iprint': 0}
            )
            x = res.x
        except Exception:
            pass
    c = np.clip(x[:2*N].reshape(N, 2), -HALF + 1e-9, HALF - 1e-9)
    return c


def _polish_smooth(c, max_iter=500):
    """High-precision smooth polish with increasing beta."""
    x0 = c.ravel()
    bounds = [(-HALF + 1e-9, HALF - 1e-9)] * (2 * N)
    for beta in [1000.0, 2000.0, 5000.0, 10000.0]:
        try:
            res = minimize(
                lambda cf: _negative_smooth_score(cf, beta=beta),
                x0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-16, 'gtol': 1e-16, 'iprint': 0}
            )
            x0 = res.x
        except Exception:
            pass
    return np.clip(x0.reshape(N, 2), -HALF + 1e-9, HALF - 1e-9)


def _polish_powell(c):
    """Powell polish for non-smooth refinement."""
    def objective(x):
        cl = x.reshape(N, 2)
        cl = np.clip(cl, -HALF + 1e-12, HALF - 1e-12)
        d = pdist(cl)
        min_pair = d.min() if d.size > 0 else 0.0
        dw = np.min(np.stack([
            cl[:, 0] + HALF, HALF - cl[:, 0],
            cl[:, 1] + HALF, HALF - cl[:, 1]
        ]), axis=0)
        min_wall = dw.min()
        return -min(min_pair / 2.0, min_wall)
    
    try:
        res = minimize(
            objective, c.ravel(), method='Powell',
            options={'maxiter': 500, 'xtol': 1e-12, 'ftol': 1e-12, 'disp': False}
        )
        c_opt = np.clip(res.x.reshape(N, 2), -HALF + 1e-12, HALF - 1e-12)
        if _score_from_centres(c_opt) > _score_from_centres(c):
            return c_opt
    except Exception:
        pass
    return c


def _kdtree_coord_descent_polish(c, max_iters=3000, tol=1e-14):
    """
    Fast coordinate-descent polish using a KD-tree to find nearest neighbors.
    For each circle, find its k nearest neighbors (k=6) and walls, then
    perform a small local optimization that only considers those constraints.
    """
    c = c.copy()
    r_current = _score_from_centres(c)
    
    for it in range(max_iters):
        if time.time() - _kdtree_coord_descent_polish._t0 > DEADLINE - 5:
            break
        
        improved = False
        for i in range(N):
            # Build KD-tree excluding point i for neighbor search
            # Use all points; we'll exclude i from results
            tree = cKDTree(c)
            # Find 6 nearest neighbors (excluding self)
            d, idx = tree.query(c[i], k=7)
            # Exclude self (should be idx[0])
            neighbor_idx = [j for j in idx[1:] if j != i]
            neighbor_dists = d[1:]
            
            # Wall distances for point i
            wall_dists = np.array([
                c[i, 0] + HALF,   # left wall
                HALF - c[i, 0],   # right wall
                c[i, 1] + HALF,   # bottom wall
                HALF - c[i, 1],   # top wall
            ])
            
            # Current minimum constraint for this point
            min_constraint = min(
                neighbor_dists.min() / 2.0,
                wall_dists.min()
            )
            
            # If this point's constraint is not the global minimum, skip
            if min_constraint > r_current + 1e-10:
                continue
            
            # Try to improve by moving point i
            # Compute gradient of the soft-min of constraints involving i
            best_pos = c[i].copy()
            best_local_r = r_current
            
            # Small local search: try perturbations
            step_sizes = [1e-4, 1e-5, 1e-6]
            for ss in step_sizes:
                for dx in [-ss, 0.0, ss]:
                    for dy in [-ss, 0.0, ss]:
                        if dx == 0.0 and dy == 0.0:
                            continue
                        new_pos = c[i] + np.array([dx, dy])
                        new_pos = np.clip(new_pos, -HALF + 1e-12, HALF - 1e-12)
                        
                        # Compute new minimum constraint for this point
                        new_neighbor_dists = np.sqrt(np.sum((c[neighbor_idx] - new_pos)**2, axis=1))
                        new_wall_dists = np.array([
                            new_pos[0] + HALF,
                            HALF - new_pos[0],
                            new_pos[1] + HALF,
                            HALF - new_pos[1],
                        ])
                        new_min = min(new_neighbor_dists.min() / 2.0, new_wall_dists.min())
                        
                        # Check if this improves the global score
                        # The global min is min(all pair dists/2, all wall dists)
                        # Moving point i can only affect constraints involving i
                        # New global min = min(old global min excluding i's constraints, new constraints involving i)
                        # Simplification: just compute full score for candidate
                        c[i] = new_pos
                        new_score = _score_from_centres(c)
                        c[i] = best_pos
                        
                        if new_score > best_local_r + 1e-15:
                            best_local_r = new_score
                            best_pos = new_pos
                            improved = True
                # Only use the smallest step size that works
                if improved:
                    break
            
            c[i] = best_pos
        
        new_r = _score_from_centres(c)
        if new_r > r_current + tol:
            r_current = new_r
        elif not improved:
            # No improvement in this full pass
            # Try a smaller step
            break
        if new_r - r_current < tol:
            break
    
    return c


def _refine_active(c, r):
    """Iterative active-set SLSQP refinement with vectorized Jacobian."""
    for _ in range(5):
        c_new = _active_set_slsqp_fast(c, r, max_iter=300)
        r_new = _score_from_centres(c_new)
        if r_new <= r + 1e-12:
            break
        c = c_new
        r = r_new
    return c, r


def _active_set_slsqp_fast(c, r_target, max_iter=300):
    """
    Fast active-set SLSQP with explicit vectorized Jacobian.
    """
    active_pairs, active_walls = _active_constraints(c, r_target, tol=1e-6)
    
    if len(active_pairs) == 0 and len(active_walls) == 0:
        return c
    
    n_pairs = len(active_pairs)
    n_walls = len(active_walls)
    n_cons = n_pairs + n_walls
    if n_cons == 0:
        return c
    
    # Precompute pair indices as arrays
    if n_pairs > 0:
        pair_i = np.array([p[0] for p in active_pairs])
        pair_j = np.array([p[1] for p in active_pairs])
    else:
        pair_i = np.array([], dtype=int)
        pair_j = np.array([], dtype=int)
    
    if n_walls > 0:
        wall_circle = np.array([w[0] for w in active_walls])
        wall_wall = np.array([w[1] for w in active_walls])
    else:
        wall_circle = np.array([], dtype=int)
        wall_wall = np.array([], dtype=int)
    
    def cons(x):
        cc = x[:2*N].reshape(N, 2)
        t = x[2*N]
        vals = []
        if n_pairs > 0:
            diffs = cc[pair_i] - cc[pair_j]
            dists = np.sqrt(np.sum(diffs**2, axis=1)) / 2.0
            vals.append(dists - t)
        if n_walls > 0:
            for w_idx in range(n_walls):
                ci = wall_circle[w_idx]
                w = wall_wall[w_idx]
                if w == 0:
                    vals.append(np.array([cc[ci, 0] + HALF - t]))
                elif w == 1:
                    vals.append(np.array([HALF - cc[ci, 0] - t]))
                elif w == 2:
                    vals.append(np.array([cc[ci, 1] + HALF - t]))
                else:
                    vals.append(np.array([HALF - cc[ci, 1] - t]))
        return np.concatenate(vals)
    
    def cons_jac(x):
        cc = x[:2*N].reshape(N, 2)
        t = x[2*N]
        jac = np.zeros((n_cons, 2*N + 1))
        
        if n_pairs > 0:
            diffs = cc[pair_i] - cc[pair_j]
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            # Avoid division by zero
            safe_dists = np.where(dists > 1e-15, dists, 1.0)
            unit = diffs / safe_dists[:, np.newaxis]
            for k in range(n_pairs):
                i = pair_i[k]
                j = pair_j[k]
                jac[k, 2*i] = unit[k, 0] / 2.0
                jac[k, 2*i+1] = unit[k, 1] / 2.0
                jac[k, 2*j] = -unit[k, 0] / 2.0
                jac[k, 2*j+1] = -unit[k, 1] / 2.0
                jac[k, 2*N] = -1.0
        
        if n_walls > 0:
            for w_idx in range(n_walls):
                ci = wall_circle[w_idx]
                w = wall_wall[w_idx]
                row = n_pairs + w_idx
                if w == 0:
                    jac[row, 2*ci] = 1.0
                elif w == 1:
                    jac[row, 2*ci] = -1.0
                elif w == 2:
                    jac[row, 2*ci+1] = 1.0
                else:
                    jac[row, 2*ci+1] = -1.0
                jac[row, 2*N] = -1.0
        
        return jac
    
    def obj(x):
        return -x[2*N]
    
    def obj_grad(x):
        g = np.zeros(2*N + 1)
        g[2*N] = -1.0
        return g
    
    x0 = np.concatenate([c.ravel(), [r_target]])
    bounds = [(-HALF + 1e-10, HALF - 1e-10)] * (2*N) + [(0.0, HALF)]
    
    try:
        res = minimize(
            obj, x0, method='SLSQP', bounds=bounds,
            jac=obj_grad,
            constraints=[{'type': 'ineq', 'fun': cons, 'jac': cons_jac}],
            options={'maxiter': max_iter, 'ftol': 1e-14, 'iprint': 0}
        )
        if res.success:
            cc = np.clip(res.x[:2*N].reshape(N, 2), -HALF + 1e-10, HALF - 1e-10)
            return cc
    except Exception:
        pass
    return c


def construct():
    t0 = time.time()
    _kdtree_coord_descent_polish._t0 = t0
    rng = np.random.default_rng(42)
    
    inits = [_grid_init(), _hex_init(), _random_init(rng)]
    best_centres = inits[0]
    best_score = _score_from_centres(best_centres)
    
    n_starts = 30
    for i in range(n_starts):
        if time.time() - t0 > DEADLINE - 25:
            break
        
        if i < len(inits):
            x0 = inits[i].ravel()
        else:
            if i % 3 == 0:
                x0 = best_centres.ravel() + rng.normal(0, 0.01, size=2*N)
            elif i % 3 == 1:
                x0 = best_centres.ravel() + rng.normal(0, 0.03, size=2*N)
            else:
                x0 = _random_init(rng).ravel()
        
        x0 = np.clip(x0, -HALF + 1e-9, HALF - 1e-9)
        
        try:
            c_opt = _optimize_epigraph(x0)
            
            if time.time() - t0 < DEADLINE - 15:
                c_opt = _polish_smooth(c_opt, max_iter=400)
            
            s = _score_from_centres(c_opt)
            if s > best_score:
                best_score = s
                best_centres = c_opt
                
        except Exception:
            pass
    
    # Final refinement pipeline on best result
    if time.time() - t0 < DEADLINE - 12:
        try:
            best_centres = _polish_smooth(best_centres, max_iter=500)
            fs = _score_from_centres(best_centres)
            if fs > best_score:
                best_score = fs
        except Exception:
            pass
    
    # KD-tree coordinate descent polish (fast, uses only local neighbors)
    if time.time() - t0 < DEADLINE - 10:
        try:
            best_centres = _kdtree_coord_descent_polish(best_centres, max_iters=2000)
            fs = _score_from_centres(best_centres)
            if fs > best_score:
                best_score = fs
        except Exception:
            pass
    
    if time.time() - t0 < DEADLINE - 8:
        try:
            best_centres, best_score = _refine_active(best_centres, best_score)
        except Exception:
            pass
    
    if time.time() - t0 < DEADLINE - 6:
        try:
            best_centres = _polish_powell(best_centres)
            fs = _score_from_centres(best_centres)
            if fs > best_score:
                best_score = fs
        except Exception:
            pass
    
    # Final active-set refinement
    if time.time() - t0 < DEADLINE - 3:
        try:
            best_centres, best_score = _refine_active(best_centres, best_score)
        except Exception:
            pass
    
    # One more KD-tree polish at the end
    if time.time() - t0 < DEADLINE - 2:
        try:
            best_centres = _kdtree_coord_descent_polish(best_centres, max_iters=1000)
            fs = _score_from_centres(best_centres)
            if fs > best_score:
                best_score = fs
        except Exception:
            pass
    
    best_centres = np.clip(best_centres, -HALF + 1e-9, HALF - 1e-9)
    return best_centres

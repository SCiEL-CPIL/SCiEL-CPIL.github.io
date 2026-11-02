"""
Packing of 50 equal circles in a unit square.
Structural change: Delaunay-triangulation-based combinatorial search with
graph-theoretic heuristics, combined with force-directed relaxation and
high-precision polishing.
"""
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import Delaunay, distance_matrix
import sys

N = 50
DEADLINE = 144.0

_IU = np.triu_indices(N, k=1)
_NPAIRS = len(_IU[0])

def _compute_radius(centers):
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2) or not np.all(np.isfinite(centers)):
        return 0.0
    wx = 0.5 - np.abs(centers[:, 0])
    wy = 0.5 - np.abs(centers[:, 1])
    min_wall = np.min(np.minimum(wx, wy))
    if min_wall < -1e-12:
        return 0.0
    diff = centers[:, None, :] - centers[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_pair = np.sqrt(np.min(dist_sq)) / 2.0
    return min(min_pair, min_wall)

def _soft_min_obj(x, beta):
    c = x.reshape(-1, 2)
    wx = 0.5 - np.abs(c[:, 0])
    wy = 0.5 - np.abs(c[:, 1])
    walls = np.minimum(wx, wy)
    ci = c[_IU[0]]; cj = c[_IU[1]]
    diff = ci - cj
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    half_d = dists / 2.0
    vals = np.concatenate([walls, half_d])
    z = np.clip(-beta * vals, -50, 50)
    z_max = np.max(z)
    lse = z_max + np.log(np.sum(np.exp(z - z_max)))
    return float(lse / beta)

def _soft_min_grad(x, beta):
    c = x.reshape(-1, 2)
    g = np.zeros_like(x)
    wx = 0.5 - np.abs(c[:, 0])
    wy = 0.5 - np.abs(c[:, 1])
    walls = np.minimum(wx, wy)
    ci = c[_IU[0]]; cj = c[_IU[1]]
    diff = ci - cj
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    half_d = dists / 2.0
    vals = np.concatenate([walls, half_d])
    z = np.clip(-beta * vals, -50, 50)
    z_max = np.max(z)
    exp_z = np.exp(z - z_max)
    sum_e = np.sum(exp_z)
    w = exp_z / sum_e
    n_w = N
    ww = w[:n_w]
    wx_lim = wx < wy; wy_lim = wy < wx; tie = ~(wx_lim | wy_lim)
    if np.any(wx_lim):
        idx = np.nonzero(wx_lim)[0]
        g[2*idx] -= ww[wx_lim] * np.sign(c[idx, 0])
    if np.any(wy_lim):
        idx = np.nonzero(wy_lim)[0]
        g[2*idx+1] -= ww[wy_lim] * np.sign(c[idx, 1])
    if np.any(tie):
        idx = np.nonzero(tie)[0]
        g[2*idx] -= 0.5 * ww[tie] * np.sign(c[idx, 0])
        g[2*idx+1] -= 0.5 * ww[tie] * np.sign(c[idx, 1])
    pw = w[n_w:]
    ds = np.maximum(dists, 1e-12)
    f = (pw / ds)[:, None] * diff
    np.add.at(g, 2*_IU[0], f[:, 0])
    np.add.at(g, 2*_IU[0]+1, f[:, 1])
    np.add.at(g, 2*_IU[1], -f[:, 0])
    np.add.at(g, 2*_IU[1]+1, -f[:, 1])
    return g

def _soft_violation_obj(x, r_target, beta):
    c = x.reshape(-1, 2)
    wx = 0.5 - np.abs(c[:, 0])
    wy = 0.5 - np.abs(c[:, 1])
    walls = np.minimum(wx, wy)
    wv = r_target - walls
    ci = c[_IU[0]]; cj = c[_IU[1]]
    diff = ci - cj
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    pv = 2.0 * r_target - dists
    zw = np.clip(beta * wv, -50, 50)
    zp = np.clip(beta * pv, -50, 50)
    return float(np.sum(np.log1p(np.exp(zw))) / beta + np.sum(np.log1p(np.exp(zp))) / beta)

def _soft_violation_grad(x, r_target, beta):
    c = x.reshape(-1, 2)
    g = np.zeros_like(x)
    wx = 0.5 - np.abs(c[:, 0])
    wy = 0.5 - np.abs(c[:, 1])
    walls = np.minimum(wx, wy)
    wv = r_target - walls
    zw = np.clip(beta * wv, -50, 50)
    sw = 1.0 / (1.0 + np.exp(-zw))
    wx_lim = wx < wy; wy_lim = wy < wx; tie = ~(wx_lim | wy_lim)
    if np.any(wx_lim):
        g[2*np.nonzero(wx_lim)[0]] += sw[wx_lim] * np.sign(c[wx_lim, 0])
    if np.any(wy_lim):
        g[2*np.nonzero(wy_lim)[0]+1] += sw[wy_lim] * np.sign(c[wy_lim, 1])
    if np.any(tie):
        idx = np.nonzero(tie)[0]
        g[2*idx] += 0.5 * sw[tie] * np.sign(c[idx, 0])
        g[2*idx+1] += 0.5 * sw[tie] * np.sign(c[idx, 1])
    ci = c[_IU[0]]; cj = c[_IU[1]]
    diff = ci - cj
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    ds = np.maximum(dists, 1e-12)
    pv = 2.0 * r_target - dists
    zp = np.clip(beta * pv, -50, 50)
    sp = 1.0 / (1.0 + np.exp(-zp))
    f = (sp / ds)[:, None] * diff
    np.add.at(g, 2*_IU[0], -f[:, 0])
    np.add.at(g, 2*_IU[0]+1, -f[:, 1])
    np.add.at(g, 2*_IU[1], f[:, 0])
    np.add.at(g, 2*_IU[1]+1, f[:, 1])
    return g

def _delaunay_edges(centers):
    """Compute Delaunay triangulation and return edge list."""
    try:
        tri = Delaunay(centers)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    if a > b: a, b = b, a
                    edges.add((a, b))
        return list(edges)
    except Exception:
        return []

def _force_directed_relax(centers, edges, n_iter=200, lr=0.01, time_budget=10.0):
    """
    Force-directed relaxation: repulsive forces between all pairs,
    attractive forces along Delaunay edges, wall repulsion.
    """
    start = time.time()
    cur = centers.copy()
    best = cur.copy()
    best_r = _compute_radius(cur)
    
    edge_arr = np.array(edges) if edges else np.zeros((0,2), dtype=int)
    
    for it in range(n_iter):
        if time.time() - start > time_budget:
            break
        
        forces = np.zeros_like(cur)
        
        # Repulsive forces between all pairs
        diff = cur[:, None, :] - cur[None, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        dist = np.sqrt(dist_sq)
        
        # Soft repulsion: F = k / d^2 * direction
        k_rep = 0.001
        rep_force = k_rep / (dist_sq + 1e-8)
        np.fill_diagonal(rep_force, 0)
        forces += np.sum(rep_force[:, :, None] * diff / (dist + 1e-8)[:, :, None], axis=1)
        
        # Attractive forces along Delaunay edges
        if len(edge_arr) > 0:
            k_att = 0.0005
            for e in edge_arr:
                i, j = int(e[0]), int(e[1])
                d = cur[i] - cur[j]
                dist = np.linalg.norm(d)
                if dist > 1e-8:
                    f = k_att * d / dist
                    forces[i] += f
                    forces[j] -= f
        
        # Wall repulsion
        k_wall = 0.01
        for i in range(N):
            for coord in range(2):
                if cur[i, coord] > 0:
                    dist_to_wall = 0.5 - cur[i, coord]
                    if dist_to_wall < 0.1:
                        forces[i, coord] -= k_wall * (0.1 - dist_to_wall)
                else:
                    dist_to_wall = 0.5 + cur[i, coord]
                    if dist_to_wall < 0.1:
                        forces[i, coord] += k_wall * (0.1 - dist_to_wall)
        
        # Update positions
        cur += lr * forces
        cur = np.clip(cur, -0.49, 0.49)
        
        r = _compute_radius(cur)
        if r > best_r:
            best_r = r
            best = cur.copy()
    
    return best, best_r

def _comb_search(centers, n_trials=50, time_budget=30.0):
    """
    Combinatorial search over Delaunay triangulations.
    Mutate edges, relax, evaluate.
    """
    start = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    rng = np.random.default_rng(12345)
    
    for trial in range(n_trials):
        if time.time() - start > time_budget:
            break
        
        # Get current Delaunay edges
        edges = _delaunay_edges(best)
        if not edges:
            break
        
        # Randomly mutate: remove a random edge, add a random non-edge
        edge_set = set(edges)
        non_edges = []
        for i in range(N):
            for j in range(i+1, N):
                if (i, j) not in edge_set:
                    non_edges.append((i, j))
        
        if not non_edges:
            continue
        
        new_edges = list(edge_set)
        # Remove a random edge
        rem_idx = rng.integers(0, len(new_edges))
        new_edges.pop(rem_idx)
        # Add a random non-edge
        add_edge = non_edges[rng.integers(0, len(non_edges))]
        new_edges.append(add_edge)
        
        # Relax with new edge set
        relaxed, r = _force_directed_relax(best, new_edges, n_iter=100, lr=0.008, time_budget=0.5)
        
        if r > best_r:
            best_r = r
            best = relaxed
    
    return best, best_r

def _hex_init():
    """Hexagonal lattice initialization."""
    best_grid = None; best_r = 0
    rng = np.random.default_rng(42)
    for attempt in range(15):
        rows = rng.integers(5, 9)
        cols = int(np.ceil(N / rows))
        sw = 1.0 / (cols + 0.5)
        sh = 1.0 / (rows * 0.866 + 0.5)
        s = min(sw, sh)
        centers = []
        for j in range(rows):
            for i in range(cols):
                x = (i - (cols-1)/2) * s
                y = (j - (rows-1)/2) * s * 0.866
                if j % 2 == 1: x += s/2
                centers.append([x, y])
        centers = np.array(centers[:N])
        if len(centers) < N:
            while len(centers) < N:
                centers = np.vstack([centers, rng.uniform(-0.45, 0.45, 2)])
        centers = np.clip(centers, -0.49, 0.49)
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r; best_grid = centers.copy()
    return best_grid if best_grid is not None else _initial_grid()

def _initial_grid():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])[:N].copy()
    rng = np.random.default_rng(12345)
    grid += rng.normal(0, 0.005, grid.shape)
    return np.clip(grid, -0.49, 0.49)

def _lbfgs_soft_min_polish(x0, beta, time_budget):
    start = time.time()
    def og(x):
        if time.time() - start > time_budget: return 0.0, np.zeros_like(x)
        return _soft_min_obj(x, beta), _soft_min_grad(x, beta)
    try:
        res = opt.minimize(og, x0, jac=True, method='L-BFGS-B',
                           bounds=[(-0.5, 0.5)]*(2*N),
                           options={'maxiter': 3000, 'ftol': 1e-16, 'gtol': 1e-13,
                                    'maxfun': 15000, 'disp': False})
        return np.clip(res.x.reshape(-1, 2), -0.5, 0.5)
    except Exception:
        return x0.reshape(-1, 2)

def _lbfgs_feasibility_polish(x0, r_target, beta, time_budget):
    start = time.time()
    def og(x):
        if time.time() - start > time_budget: return 0.0, np.zeros_like(x)
        return _soft_violation_obj(x, r_target, beta), _soft_violation_grad(x, r_target, beta)
    try:
        res = opt.minimize(og, x0, jac=True, method='L-BFGS-B',
                           bounds=[(-0.5, 0.5)]*(2*N),
                           options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-14,
                                    'maxfun': 25000, 'disp': False, 'maxls': 50})
        return np.clip(res.x.reshape(-1, 2), -0.5, 0.5)
    except Exception:
        return x0.reshape(-1, 2)

def _refine_radius(centers, time_budget):
    start = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    lo = best_r
    hi = min(0.08, best_r * 1.05)
    for i in range(22):
        if time.time() - start > time_budget: break
        mid = (lo + hi) / 2
        progress = i / 21
        beta = 200.0 + 800.0 * progress
        remaining = time_budget - (time.time() - start)
        if remaining < 5: break
        lb = min(18.0, remaining * 0.85)
        trial = _lbfgs_feasibility_polish(best.flatten(), mid, beta, lb)
        trial_r = _compute_radius(trial)
        if trial_r >= mid - 1e-8:
            lo = mid; best = trial; best_r = trial_r
        else:
            hi = mid
    return best, best_r

def _high_precision_polish(centers, time_budget):
    start = time.time()
    best = centers.copy()
    best_r = _compute_radius(best)
    
    # Phase 1: Progressive sharpening
    for beta in [500, 1000, 2000, 4000, 8000, 12000]:
        if time.time() - start > time_budget: break
        remaining = time_budget - (time.time() - start)
        if remaining < 4: break
        lb = min(14.0, remaining * 0.85)
        trial = _lbfgs_feasibility_polish(best.flatten(), best_r, beta, lb)
        trial_r = _compute_radius(trial)
        if trial_r > best_r: best_r = trial_r; best = trial
    
    # Phase 2: Soft-min
    if time.time() - start < time_budget - 8:
        for beta in [200, 500, 1000, 2000, 4000]:
            if time.time() - start > time_budget - 4: break
            remaining = time_budget - (time.time() - start)
            if remaining < 4: break
            lb = min(14.0, remaining * 0.85)
            trial = _lbfgs_soft_min_polish(best.flatten(), beta, lb)
            trial_r = _compute_radius(trial)
            if trial_r > best_r: best_r = trial_r; best = trial
    
    # Phase 3: Push
    if time.time() - start < time_budget - 4:
        for pf in [1.001, 1.003, 1.005, 1.008, 1.012, 1.018]:
            if time.time() - start > time_budget - 3: break
            rp = best_r * pf
            if rp > 0.08: break
            remaining = time_budget - (time.time() - start)
            if remaining < 3: break
            lb = min(12.0, remaining * 0.85)
            trial = _lbfgs_feasibility_polish(best.flatten(), rp, 4000.0, lb)
            trial_r = _compute_radius(trial)
            if trial_r > best_r: best_r = trial_r; best = trial
    
    # Phase 4: Ultra-tight
    if time.time() - start < time_budget - 2:
        for beta in [8000, 15000, 25000]:
            if time.time() - start > time_budget - 1.5: break
            remaining = time_budget - (time.time() - start)
            if remaining < 2: break
            lb = min(15.0, remaining * 0.9)
            trial = _lbfgs_feasibility_polish(best.flatten(), best_r, beta, lb)
            trial_r = _compute_radius(trial)
            if trial_r > best_r: best_r = trial_r; best = trial
    
    return best, best_r

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    rng = np.random.default_rng(777)
    
    # Generate multiple initial configurations
    inits = [_initial_grid(), _hex_init()]
    rng_init = np.random.default_rng(999)
    inits.append(rng_init.uniform(-0.45, 0.45, (N, 2)))
    rng_init2 = np.random.default_rng(4242)
    inits.append(rng_init2.uniform(-0.45, 0.45, (N, 2)))
    
    best_centers = None; best_radius = 0
    for init in inits:
        r = _compute_radius(init)
        if r > best_radius:
            best_radius = r; best_centers = init.copy()
    if best_centers is None:
        best_centers = _initial_grid(); best_radius = _compute_radius(best_centers)
    
    # Phase 1: Force-directed relaxation with Delaunay structure
    if time.time() < deadline - 60:
        fd_budget = min(25.0, (deadline - time.time()) * 0.35)
        if fd_budget > 5:
            edges = _delaunay_edges(best_centers)
            fd_result, fd_r = _force_directed_relax(best_centers, edges, n_iter=300, lr=0.01, time_budget=fd_budget)
            if fd_r > best_radius:
                best_radius = fd_r; best_centers = fd_result
    
    # Phase 2: Combinatorial search over triangulations
    if time.time() < deadline - 45:
        comb_budget = min(20.0, (deadline - time.time()) * 0.35)
        if comb_budget > 5:
            comb_result, comb_r = _comb_search(best_centers, n_trials=40, time_budget=comb_budget)
            if comb_r > best_radius:
                best_radius = comb_r; best_centers = comb_result
    
    # Phase 3: Refine radius via binary search
    if time.time() < deadline - 25:
        rb = min(45.0, deadline - time.time() - 12)
        best_centers, best_radius = _refine_radius(best_centers, rb)
    
    # Phase 4: High-precision polishing
    if time.time() < deadline - 6:
        pb = min(30.0, deadline - time.time() - 2)
        best_centers, best_radius = _high_precision_polish(best_centers, pb)
    
    # Final check
    if time.time() < deadline - 2:
        fb = min(15.0, deadline - time.time() - 1)
        if fb > 3:
            cr = _compute_radius(best_centers)
            trial = _lbfgs_feasibility_polish(best_centers.flatten(), cr, 20000.0, fb)
            trial_r = _compute_radius(trial)
            if trial_r > best_radius:
                best_radius = trial_r; best_centers = trial
    
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 2):
        return _initial_grid()
    return best_centers

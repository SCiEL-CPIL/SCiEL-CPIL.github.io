"""
Optimized Sphere Packing: 100 equal spheres in unit cube.
Smooth log-sum-exp objective + L-BFGS-B + multi-start + FCC/BCC inits + high-alpha polish.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 100
DEADLINE = 148.0  # Safety margin from 150s limit

def _all_dists(c):
    """Return all pairwise and wall distances (N*(N-1)/2 + 6N values)."""
    d = cdist(c, c)
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = d[i_idx, j_idx] / 2.0
    wall_dists = np.concatenate([
        0.5 - c[:, 0], 0.5 + c[:, 0],
        0.5 - c[:, 1], 0.5 + c[:, 1],
        0.5 - c[:, 2], 0.5 + c[:, 2]
    ])
    return pair_dists, wall_dists, i_idx, j_idx

def _compute_smooth_obj_grad(c, alpha):
    """
    Smooth log-sum-exp objective: f(c) = (1/alpha) * log(sum exp(-alpha * s_k))
    Minimizing f maximizes the minimum distance s.
    """
    pair_dists, wall_dists, i_idx, j_idx = _all_dists(c)
    all_dists = np.concatenate([pair_dists, wall_dists])
    n_pairs = len(pair_dists)
    
    # Numerical stability: shift by min
    m = np.min(all_dists)
    shifted = all_dists - m
    exp_terms = np.exp(-alpha * shifted)
    Z = np.sum(exp_terms)
    
    if Z == 0 or not np.isfinite(Z):
        return 0.0, np.zeros_like(c)
    
    obj_val = (1.0 / alpha) * (np.log(Z) - alpha * m)
    
    # Weights: w_k = exp(-alpha * s_k) / Z
    weights = exp_terms / Z
    
    grad_c = np.zeros_like(c)
    
    # Pairwise part
    pair_dists_raw = 2.0 * pair_dists  # actual distances
    pair_dists_raw_safe = np.where(pair_dists_raw < 1e-12, 1e-12, pair_dists_raw)
    
    ci = c[i_idx]
    cj = c[j_idx]
    diff = ci - cj
    norms = pair_dists_raw_safe[:, np.newaxis]
    u = diff / norms  # unit vector from j to i
    
    w_pair = weights[:n_pairs]
    w_pair = w_pair[:, np.newaxis]
    
    # s_ij = 0.5 * ||c_i - c_j||, ds_ij/dc_i = 0.5 * u_i
    # grad contribution: -w * ds/dc
    contrib = (0.5 * w_pair * u)
    
    np.add.at(grad_c, i_idx, -contrib)
    np.add.at(grad_c, j_idx, contrib)
    
    # Wall part
    w_wall = weights[n_pairs:]
    ww = w_wall.reshape(6, N)
    
    # +wall: s = 0.5 - c, ds/dc = -1, contribution = -w*(-1) = +w
    # -wall: s = 0.5 + c, ds/dc = +1, contribution = -w*(+1) = -w
    grad_c[:, 0] += (ww[0] - ww[1])
    grad_c[:, 1] += (ww[2] - ww[3])
    grad_c[:, 2] += (ww[4] - ww[5])
    
    return obj_val, grad_c

def _score(c):
    """Compute the exact packing radius."""
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    if np.any(c > 0.5 + 1e-15) or np.any(c < -0.5 - 1e-15):
        return 0.0
    c = np.clip(c, -0.5, 0.5)
    bound = np.min(0.5 - np.abs(c))
    if bound < 0.0:
        return 0.0
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    pair_min = np.min(d) / 2.0
    return float(min(bound, pair_min))

def _initial_fcc(seed):
    """FCC lattice initialization, cropped to 100 points in cube."""
    rng = np.random.RandomState(seed)
    # FCC lattice points in [0,1]^3 scaled to [-0.5, 0.5]^3
    # FCC: face-centered cubic, points at (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) + integer shifts
    pts = []
    for ix in range(-1, 3):
        for iy in range(-1, 3):
            for iz in range(-1, 3):
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    x = ix + dx
                    y = iy + dy
                    z = iz + dz
                    # Map to [-0.5, 0.5]
                    if -0.48 <= x <= 0.48 and -0.48 <= y <= 0.48 and -0.48 <= z <= 0.48:
                        pts.append([x - 0.5, y - 0.5, z - 0.5])
    pts = np.array(pts)
    if len(pts) >= N:
        # Randomly select N points
        idx = rng.choice(len(pts), N, replace=False)
        c = pts[idx].copy()
    else:
        # Pad with random points
        c = pts.copy()
        while len(c) < N:
            c = np.vstack([c, rng.uniform(-0.45, 0.45, (1, 3))])
        c = c[:N]
    # Add small jitter
    c += rng.uniform(-0.01, 0.01, c.shape)
    c = np.clip(c, -0.48, 0.48)
    return c

def _initial_bcc(seed):
    """BCC lattice initialization."""
    rng = np.random.RandomState(seed)
    pts = []
    for ix in range(-1, 3):
        for iy in range(-1, 3):
            for iz in range(-1, 3):
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0.5)]:
                    x = ix + dx
                    y = iy + dy
                    z = iz + dz
                    if -0.48 <= x <= 0.48 and -0.48 <= y <= 0.48 and -0.48 <= z <= 0.48:
                        pts.append([x - 0.5, y - 0.5, z - 0.5])
    pts = np.array(pts)
    if len(pts) >= N:
        idx = rng.choice(len(pts), N, replace=False)
        c = pts[idx].copy()
    else:
        c = pts.copy()
        while len(c) < N:
            c = np.vstack([c, rng.uniform(-0.45, 0.45, (1, 3))])
        c = c[:N]
    c += rng.uniform(-0.01, 0.01, c.shape)
    c = np.clip(c, -0.48, 0.48)
    return c

def _initial_grid(seed):
    """Jittered 5x5x4 grid."""
    rng = np.random.RandomState(seed)
    xs = np.linspace(-0.4, 0.4, 5)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.linspace(-0.375, 0.375, 4)
    centres = []
    for z in zs:
        for y in ys:
            for x in xs:
                jitter = rng.uniform(-0.02, 0.02, 3)
                centres.append([x, y, z] + jitter)
    return np.array(centres[:N])

def _initial_random(seed):
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, (N, 3))

def _optimize_smooth(c_init, time_limit, start_time, alphas=None):
    """Run smooth log-sum-exp annealing from c_init."""
    if alphas is None:
        alphas = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    
    best_c = c_init.copy()
    best_score = _score(c_init)
    current_c = c_init.copy()
    
    for alpha in alphas:
        if time.time() - start_time > time_limit:
            break
        
        def obj(x):
            c_flat = x.reshape(N, 3)
            val, _ = _compute_smooth_obj_grad(c_flat, alpha)
            return val
        
        def grad(x):
            c_flat = x.reshape(N, 3)
            _, g = _compute_smooth_obj_grad(c_flat, alpha)
            return g.flatten()
        
        x0 = current_c.flatten()
        bounds = [(-0.5, 0.5)] * (3 * N)
        
        try:
            remaining_time = time_limit - (time.time() - start_time)
            if remaining_time < 3:
                break
            
            # Scale maxiter with remaining time
            maxiter = int(min(800, max(50, remaining_time * 15)))
            
            result = minimize(
                obj, x0, jac=grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            c_new = result.x.reshape(N, 3)
            c_new = np.clip(c_new, -0.5, 0.5)
            
            score_new = _score(c_new)
            
            if score_new > best_score:
                best_score = score_new
                best_c = c_new.copy()
                current_c = c_new.copy()
            elif result.fun < obj(x0) - 1e-13:
                current_c = c_new.copy()
            else:
                # Kick: structured perturbation
                rng = np.random.RandomState()
                # Randomly pick a subset of points and shift them
                n_kick = max(5, N // 10)
                idx = rng.choice(N, n_kick, replace=False)
                shift = rng.uniform(-0.02, 0.02, (n_kick, 3))
                c_kicked = current_c.copy()
                c_kicked[idx] += shift
                c_kicked = np.clip(c_kicked, -0.5, 0.5)
                s_kick = _score(c_kicked)
                if s_kick > best_score:
                    best_score = s_kick
                    best_c = c_kicked.copy()
                    current_c = c_kicked.copy()
                elif s_kick > _score(current_c) * 0.95:
                    current_c = c_kicked.copy()
                # If badly stuck, revert to best
                if score_new < best_score * 0.85:
                    current_c = best_c.copy()
        except Exception:
            pass
    
    return best_c, best_score

def _polish(c, time_limit, start_time):
    """High-alpha polish for final refinement."""
    alphas_polish = [5000, 10000, 20000, 50000]
    best_c = c.copy()
    best_score = _score(c)
    
    for alpha in alphas_polish:
        if time.time() - start_time > time_limit:
            break
        
        def obj(x):
            c_flat = x.reshape(N, 3)
            val, _ = _compute_smooth_obj_grad(c_flat, alpha)
            return val
        
        def grad(x):
            c_flat = x.reshape(N, 3)
            _, g = _compute_smooth_obj_grad(c_flat, alpha)
            return g.flatten()
        
        try:
            remaining_time = time_limit - (time.time() - start_time)
            if remaining_time < 2:
                break
            
            maxiter = int(min(500, max(30, remaining_time * 10)))
            
            result = minimize(
                obj, best_c.flatten(), jac=grad,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (3 * N),
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            
            c_new = result.x.reshape(N, 3)
            c_new = np.clip(c_new, -0.5, 0.5)
            score_new = _score(c_new)
            
            if score_new > best_score:
                best_score = score_new
                best_c = c_new.copy()
        except Exception:
            pass
    
    return best_c, best_score

def construct():
    start_time = time.time()
    
    best_c = None
    best_score = 0.0
    
    # Generate diverse initializations
    inits = [
        ("fcc1", _initial_fcc(42)),
        ("fcc2", _initial_fcc(123)),
        ("bcc1", _initial_bcc(7)),
        ("grid1", _initial_grid(1)),
        ("grid2", _initial_grid(99)),
        ("rand1", _initial_random(55)),
        ("rand2", _initial_random(777)),
    ]
    
    # Allocate time: first 70% for multi-start, last 30% for polish
    multi_start_limit = start_time + (DEADLINE - start_time) * 0.7
    polish_limit = DEADLINE - 2.0
    
    # Run multi-start optimization
    for name, c_init in inits:
        if time.time() > multi_start_limit:
            break
        
        # Give each start a portion of the multi-start time
        # But stop early if we're close to the limit
        per_start_limit = min(multi_start_limit, time.time() + 15)
        
        c_opt, s_opt = _optimize_smooth(c_init, per_start_limit, start_time)
        
        if s_opt > best_score:
            best_score = s_opt
            best_c = c_opt.copy()
    
    # Final polish on the best result
    if best_c is not None and time.time() < polish_limit - 5:
        best_c, best_score = _polish(best_c, polish_limit, start_time)
    
    # Fallback
    if best_c is None:
        best_c = _initial_grid(1)
    
    # Ensure validity
    best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c

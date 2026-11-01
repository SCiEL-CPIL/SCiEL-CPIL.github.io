"""
Refined program for packing 50 equal circles in a unit circle.
Strategy:
1. Robust Soft-Min Surrogate with vectorized gradient computation.
2. Multi-stage optimization: coarse annealing -> fine annealing -> high-precision polish.
3. Diverse initializations: ring-based, hexagonal lattice, and random.
4. Efficient score computation and gradient evaluation.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 118.0  # Safety margin for 120s limit

# EVOLVE-BLOCK-START

def _compute_score(centers):
    """
    Compute the maximum radius r for which circles centered at 'centers' 
    are disjoint and inside the unit circle.
    r = min( min_{i<j} |c_i - c_j| / 2, min_i (1 - |c_i|) )
    """
    if centers.size == 0:
        return 0.0
    n = centers.shape[0]
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Mask diagonal to avoid zero distance
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Boundary distances
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    min_bound_dist = np.min(1.0 - norms)
    
    return min(min_pair_dist / 2.0, min_bound_dist)

def _soft_min_obj_and_grad(params, n, beta):
    """
    Objective: -soft_min(constraints)
    Constraints: d_ij / 2 for pairs, 1 - |c_i| for boundary.
    We want to maximize the minimum of these, so we minimize the negative soft-min.
    
    Soft min definition:
    soft_min(x) = -1/beta * log(sum(exp(-beta * x_i)))
    This is a smooth approximation of min(x_i).
    
    To improve numerical stability, we shift x by the current minimum:
    Let m = min(x).
    soft_min(x) = m - 1/beta * log(sum(exp(-beta * (x_i - m))))
    """
    centers = params.reshape(n, 2)
    
    # 1. Compute pair distances (vectorized)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    # 2. Compute boundary distances
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    bound_dists = 1.0 - norms
    
    # 3. Combine all constraints into a single vector
    vals = np.concatenate([pair_dists / 2.0, bound_dists])
    
    # 4. Compute Soft Min
    m = np.min(vals)
    shifted = vals - m
    exp_terms = np.exp(-beta * shifted)
    S = np.sum(exp_terms)
    
    obj_val = -(m - (1.0 / beta) * np.log(S))
    
    # 5. Compute Gradient (vectorized)
    weights = exp_terms / S
    
    grad_centers = np.zeros_like(centers)
    
    # Pair contributions (vectorized)
    pair_weights = weights[:n*(n-1)//2]
    # Filter out negligible weights
    mask = pair_weights > 1e-12
    if np.any(mask):
        pw = pair_weights[mask]
        ii = i_idx[mask]
        jj = j_idx[mask]
        d = dists[ii, jj]
        valid = d > 1e-8
        if np.any(valid):
            pw_v = pw[valid]
            ii_v = ii[valid]
            jj_v = jj[valid]
            d_v = d[valid]
            vecs = (centers[ii_v] - centers[jj_v]) / d_v[:, np.newaxis]
            # grad_centers[i] -= w * 0.5 * vec
            # grad_centers[j] += w * 0.5 * vec
            np.add.at(grad_centers, ii_v, -pw_v[:, np.newaxis] * 0.5 * vecs)
            np.add.at(grad_centers, jj_v, pw_v[:, np.newaxis] * 0.5 * vecs)
    
    # Boundary contributions (vectorized)
    bound_weights = weights[n*(n-1)//2:]
    mask_b = bound_weights > 1e-12
    if np.any(mask_b):
        bw = bound_weights[mask_b]
        bi = np.arange(n)[mask_b]
        bn = norms[bi]
        valid_b = bn > 1e-8
        if np.any(valid_b):
            bw_v = bw[valid_b]
            bi_v = bi[valid_b]
            bn_v = bn[valid_b]
            vecs_b = centers[bi_v] / bn_v[:, np.newaxis]
            # grad_centers[i] += w * vec
            np.add.at(grad_centers, bi_v, bw_v[:, np.newaxis] * vecs_b)
            
    return obj_val, grad_centers

def _get_ring_initialization(n):
    """
    Generate a ring-based initialization.
    Approximate optimal distribution for N=50:
    Center: 1
    Ring 1: 6
    Ring 2: 12
    Ring 3: 12
    Ring 4: 9
    Ring 5: 10
    Total: 50
    """
    centers = []
    centers.append([0.0, 0.0])
    
    rings = [
        (6, 0.22),
        (12, 0.44),
        (12, 0.66),
        (9, 0.82),
        (10, 0.92)
    ]
    
    count = 1
    for count_in_ring, radius in rings:
        if count >= n:
            break
        angles = np.linspace(0, 2 * np.pi, count_in_ring, endpoint=False)
        for angle in angles:
            if count >= n:
                break
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
            count += 1
            
    return np.array(centers[:n])

def _get_hex_lattice_initialization(n):
    """
    Generate a hexagonal lattice initialization.
    """
    r_est = 0.12
    s = 2 * r_est
    points = []
    max_range = int(2.0 / s) + 3
    
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 0.95:
                points.append([x, y])
                
    if len(points) < n:
        return _get_ring_initialization(n)
        
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    sorted_idx = np.argsort(dists)
    return points[sorted_idx[:n]]

def _get_perturbed_hex_lattice(n, seed=0):
    """
    Generate a hexagonal lattice with random perturbations.
    """
    rng = np.random.default_rng(seed)
    r_est = 0.12
    s = 2 * r_est
    points = []
    max_range = int(2.0 / s) + 3
    
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 0.95:
                px = x + rng.uniform(-0.01, 0.01)
                py = y + rng.uniform(-0.01, 0.01)
                if np.hypot(px, py) <= 0.95:
                    points.append([px, py])
                    
    if len(points) < n:
        return _get_hex_lattice_initialization(n)
        
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    sorted_idx = np.argsort(dists)
    return points[sorted_idx[:n]]

def _get_random_initialization(n, seed=42):
    """Generate random initialization."""
    rng = np.random.default_rng(seed)
    centers = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.95)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        centers.append([x, y])
    return np.array(centers)

def _optimize_phase(centers, start_time, time_budget, beta_start, beta_end, cooling_rate=0.95):
    """
    Run a phase of simulated annealing with L-BFGS-B.
    """
    current = centers.copy()
    best_score = _compute_score(current)
    best_centers = current.copy()
    
    end_time = start_time + time_budget
    T = 0.05
    T_end = 1e-6
    beta = beta_start
    
    while time.time() < end_time:
        try:
            res = minimize(
                _soft_min_obj_and_grad, 
                current.flatten(), 
                args=(N, beta),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 50, 'gtol': 1e-5}
            )
            candidate = res.x.reshape(N, 2)
            if not np.all(np.isfinite(candidate)):
                continue
                
            # Check validity
            norms = np.linalg.norm(candidate, axis=1)
            if np.any(norms > 1.0):
                mask = norms > 1.0
                candidate = candidate.copy()
                candidate[mask] *= 1.0 / norms[mask]
                
            score = _compute_score(candidate)
            
            if score > best_score:
                best_score = score
                best_centers = candidate.copy()
                current = candidate.copy()
            else:
                delta = score - best_score
                if delta > 0:
                    current = candidate.copy()
                else:
                    prob = np.exp(delta / T)
                    if np.random.rand() < prob:
                        current = candidate.copy()
                        
        except Exception:
            pass
            
        T *= cooling_rate
        # Interpolate beta
        progress = min(1.0, (beta_start - beta) / (beta_end - beta_start)) if beta_end != beta_start else 0.0
        beta = beta_start + progress * (beta_end - beta_start)
        beta *= 1.05
        
        if T < T_end:
            break
            
    return best_centers, best_score

def _final_polish(centers, start_time):
    """
    High-precision polish using very high beta soft-min.
    """
    current = centers.copy()
    best_score = _compute_score(current)
    best_centers = current.copy()
    
    betas = [500, 2000, 10000]
    
    for beta in betas:
        if time.time() > DEADLINE - 3:
            break
        try:
            res = minimize(
                _soft_min_obj_and_grad, 
                current.flatten(), 
                args=(N, beta),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12}
            )
            candidate = res.x.reshape(N, 2)
            if np.all(np.isfinite(candidate)):
                score = _compute_score(candidate)
                if score > best_score:
                    best_score = score
                    best_centers = candidate.copy()
                    current = candidate
        except Exception:
            pass
            
    # Sub-gradient polish for final nudge
    if time.time() < DEADLINE - 2:
        current = best_centers.copy()
        step = 1e-5
        for _ in range(2000):
            if time.time() > DEADLINE - 2:
                break
            diff = current[:, np.newaxis, :] - current[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair_dist = np.min(dists)
            norms = np.sqrt(np.sum(current * current, axis=1))
            min_bound_dist = np.min(1.0 - norms)
            
            if min_pair_dist / 2.0 < min_bound_dist:
                i, j = np.unravel_index(np.argmin(dists), dists.shape)
                if i > j: i, j = j, i
                d = dists[i, j]
                if d > 1e-8:
                    vec = (current[i] - current[j]) / d
                    current[i] += vec * step * 0.5
                    current[j] -= vec * step * 0.5
            else:
                i = np.argmin(1.0 - norms)
                norm = norms[i]
                if norm > 1e-8:
                    vec = current[i] / norm
                    current[i] -= vec * step
                    
            new_norms = np.sqrt(np.sum(current * current, axis=1))
            mask = new_norms > 0.9999
            if np.any(mask):
                current[mask] *= 0.9999 / new_norms[mask]
                
            score = _compute_score(current)
            if score > best_score:
                best_score = score
                best_centers = current.copy()
                
    return best_centers, best_score

def construct():
    start_time = time.time()
    best_score = 0.0
    best_centers = None
    
    # Diverse initializations
    initializations = [
        _get_ring_initialization(N),
        _get_hex_lattice_initialization(N),
        _get_perturbed_hex_lattice(N, seed=1),
        _get_perturbed_hex_lattice(N, seed=2),
        _get_random_initialization(N, seed=1),
        _get_random_initialization(N, seed=42),
        _get_random_initialization(N, seed=99),
    ]
    
    # Phase 1: Coarse optimization with lower beta
    for init in initializations:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 40:
            break
            
        norms = np.linalg.norm(init, axis=1)
        mask = norms > 0.95
        if np.any(mask):
            init = init.copy()
            init[mask] *= 0.95 / norms[mask]
            
        time_budget = min(8.0, DEADLINE - 40 - elapsed)
        if time_budget < 1.0:
            break
            
        optimized, score = _optimize_phase(init, time.time(), time_budget, beta_start=50.0, beta_end=200.0)
        if score > best_score:
            best_score = score
            best_centers = optimized.copy()
            
    if best_centers is None:
        return initializations[0]
        
    # Phase 2: Fine optimization with higher beta
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 20:
        time_budget = min(15.0, DEADLINE - 20 - elapsed)
        if time_budget > 1.0:
            optimized, score = _optimize_phase(best_centers, time.time(), time_budget, beta_start=200.0, beta_end=1000.0)
            if score > best_score:
                best_score = score
                best_centers = optimized.copy()
                
    # Final Polish
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - 5:
        final_centers, final_score = _final_polish(best_centers, start_time)
        if final_score > best_score:
            best_score = final_score
            best_centers = final_centers
            
    return best_centers

# EVOLVE-BLOCK-END

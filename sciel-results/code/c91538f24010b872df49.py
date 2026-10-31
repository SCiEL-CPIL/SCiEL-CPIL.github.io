"""
Combined approach for Heilbronn triangles:
1. Barycentric coordinates for simple box constraints (inspiration 1).
2. L-BFGS-B optimization with a smooth soft-min objective (inspiration 1 & 2).
3. Inverse squared distance penalty to enforce separation and smooth landscape (directive).
4. Multi-stage optimization: Repulsion -> Balancing -> Max-Min Area.
5. Simulated Annealing for diverse initializations.
"""

import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2.0
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])
V = np.array([V0, V1, V2])

# Precompute triangle indices for vectorized area computation
_TRI_IDX = np.array(list(combinations(range(N), 3)))
_N_TRI = _TRI_IDX.shape[0]

# Precompute pair indices for distance penalty
_PAIR_IDX = np.array(list(combinations(range(N), 2)))
_N_PAIR = _PAIR_IDX.shape[0]

def _bary_to_cart(bary_flat):
    """
    Convert flat barycentric array (2*N,) to Cartesian points (N, 2).
    Input bary_flat[i] = u_i, bary_flat[i+1] = v_i.
    w_i = 1 - u_i - v_i.
    """
    u = bary_flat[0::2]
    v = bary_flat[1::2]
    w = 1.0 - u - v
    
    # Points = u*V0 + v*V1 + w*V2
    # Since V0 is origin, this simplifies to v*V1 + w*V2
    # But for clarity and generalization:
    pts = np.zeros((N, 2))
    pts += u[:, None] * V0
    pts += v[:, None] * V1
    pts += w[:, None] * V2
    return pts

def _all_areas_cart(pts):
    """Vectorized area computation for all triangles."""
    pi = pts[_TRI_IDX[:, 0]]
    pj = pts[_TRI_IDX[:, 1]]
    pk = pts[_TRI_IDX[:, 2]]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    areas = 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)
    return areas

def _all_dists_cart(pts):
    """Vectorized squared distance computation for all pairs."""
    pi = pts[_PAIR_IDX[:, 0]]
    pj = pts[_PAIR_IDX[:, 1]]
    dx = pj[:, 0] - pi[:, 0]
    dy = pj[:, 1] - pi[:, 1]
    dists_sq = dx**2 + dy**2
    return dists_sq

def _objective(bary_flat, beta, penalty_weight, time_start, deadline):
    """
    Objective function for L-BFGS-B.
    Minimizes: -soft_min_area + penalty_weight * sum(1/d^2)
    
    soft_min_area = -1/beta * log(sum(exp(-beta * areas)))
    """
    # Time guard
    if time.time() > deadline:
        return 1e10

    pts = _bary_to_cart(bary_flat)
    
    # Calculate areas
    areas = _all_areas_cart(pts)
    
    # Soft min of areas
    # We want to maximize min(areas), so we minimize -min(areas).
    # Smooth approximation: -min(areas) approx -soft_min(areas)
    # soft_min(areas) = -1/beta * log(sum(exp(-beta * areas)))
    # So we minimize: 1/beta * log(sum(exp(-beta * areas)))
    
    # Numerical stability: shift by max area
    max_a = np.max(areas)
    if max_a < 1e-12:
        # Degenerate case
        return 1e10
        
    exp_terms = np.exp(-beta * (areas - max_a))
    log_sum = np.log(np.sum(exp_terms))
    soft_min_val = -(max_a + log_sum / beta)
    
    # We want to maximize this, so objective is -soft_min_val
    obj_area = -soft_min_val
    
    # Calculate distance penalty
    # Penalty = sum(1/d^2)
    # If points are very close, this blows up, pushing them apart.
    dists_sq = _all_dists_cart(pts)
    
    # Avoid division by zero
    dists_sq = np.maximum(dists_sq, 1e-12)
    penalty = np.sum(1.0 / dists_sq)
    
    return obj_area + penalty_weight * penalty

def _generate_initial_bary(seed):
    """Generate a random valid barycentric configuration."""
    rng = np.random.default_rng(seed)
    u = rng.random(N)
    v = rng.random(N) * (1.0 - u)
    return np.concatenate([u, v])

def _generate_symmetric_bary():
    """
    Generate a symmetric initial configuration in barycentric coordinates.
    Vertices: (1,0,0), (0,1,0), (0,0,1) -> (0,0), (1,0), (0,1) in (u,v)
    Note: Our mapping is u*V0 + v*V1 + w*V2.
    V0=(0,0), V1=(1,0), V2=(0.5, H).
    So (u,v) = (1,0) is V0. (0,1) is V1. (0,0) is V2.
    
    Let's place points near vertices and midpoints.
    """
    # Vertices (slightly inside)
    eps = 0.05
    # V0: u=1-eps, v=0, w=0 -> u=1-eps, v=0
    # V1: u=0, v=1-eps, w=0 -> u=0, v=1-eps
    # V2: u=0, v=0, w=1-eps -> u=0, v=0
    
    pts_bary = [
        [1-eps, 0],
        [0, 1-eps],
        [0, 0],
        # Midpoints
        [0.5*(1-eps), 0.5*(1-eps)], # Mid V0-V1
        [0.5*(1-eps), 0],           # Mid V0-V2
        [0, 0.5*(1-eps)],           # Mid V1-V2
        # Interior
        [1/3, 1/3],
        [1/3 + 0.1, 1/3],
        [1/3, 1/3 + 0.1],
        [0.2, 0.2],
        [0.4, 0.4],
    ]
    
    # Ensure validity u+v <= 1
    valid_pts = []
    for p in pts_bary:
        if p[0] + p[1] <= 1.0:
            valid_pts.append(p)
        else:
            # Scale down
            s = 1.0 / (p[0] + p[1])
            valid_pts.append([p[0]*s, p[1]*s])
            
    return np.array(valid_pts).flatten()

def _simulated_annealing_bary(start_bary, time_limit, seed):
    """
    Simple SA in barycentric space to find good initializations.
    """
    rng = np.random.default_rng(seed)
    x = start_bary.copy()
    
    def score(x):
        pts = _bary_to_cart(x)
        areas = _all_areas_cart(pts)
        return np.min(areas)
        
    current_score = score(x)
    best_score = current_score
    best_x = x.copy()
    
    T = 0.01
    alpha = 0.999
    step_size = 0.05
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        if T < 1e-6:
            T = 1e-6
            step_size *= 0.8
            if step_size < 0.001:
                break
                
        # Pick random point
        idx = rng.integers(N)
        
        # Perturb u and v
        du = rng.normal(0, step_size)
        dv = rng.normal(0, step_size)
        
        new_u = x[2*idx] + du
        new_v = x[2*idx+1] + dv
        
        # Clamp to valid region [0,1] and u+v<=1
        new_u = np.clip(new_u, 0, 1)
        new_v = np.clip(new_v, 0, 1)
        if new_u + new_v > 1:
            # Project to boundary u+v=1
            s = 1.0 / (new_u + new_v)
            new_u *= s
            new_v *= s
            
        old_u = x[2*idx]
        old_v = x[2*idx+1]
        x[2*idx] = new_u
        x[2*idx+1] = new_v
        
        new_score = score(x)
        delta = new_score - current_score
        
        if delta > 0 or (T > 0 and rng.random() < np.exp(delta / (T + 1e-15))):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_x = x.copy()
        else:
            x[2*idx] = old_u
            x[2*idx+1] = old_v
            
        T *= alpha
        step_size *= 0.9999
        
    return best_x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    best_bary = None
    best_score = -1.0
    
    # Initializations
    inits = [
        _generate_symmetric_bary(),
        _generate_initial_bary(42),
        _generate_initial_bary(123),
        _generate_initial_bary(789),
        _generate_initial_bary(101),
        _generate_initial_bary(555),
    ]
    
    # Run SA on a few inits to improve them
    for i in range(len(inits)):
        if time.time() > deadline - 30:
            break
        if i > 0: # Skip SA for first one to save time, or run briefly
            try:
                improved_bary, _ = _simulated_annealing_bary(inits[i], time_limit=2.0, seed=1000+i)
                inits[i] = improved_bary
            except:
                pass
    
    # Optimization Stages
    # Stage 1: High repulsion, low beta (smooth landscape, spread out)
    # Stage 2: Medium repulsion, medium beta
    # Stage 3: Low repulsion, high beta (precise max-min)
    # Stage 4: No repulsion, very high beta (final polish)
    
    stages = [
        (10.0, 100.0, 100),   # beta, penalty, maxiter
        (50.0, 10.0, 150),
        (200.0, 1.0, 200),
        (1000.0, 0.1, 300),
        (5000.0, 0.0, 500),   # Final high precision
    ]
    
    bounds = [(0.0, 1.0)] * (2 * N)
    
    for init_idx, init_bary in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        x_current = init_bary.copy()
        
        for stage_idx, (beta, pw, maxiter) in enumerate(stages):
            if time.time() > deadline - 5:
                break
                
            try:
                res = minimize(
                    _objective,
                    x_current,
                    args=(beta, pw, start_time, deadline),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                # Check if result is better
                if res.fun < _objective(x_current, beta, pw, start_time, deadline):
                    x_current = res.x
                elif res.success:
                    x_current = res.x
            except Exception:
                pass
        
        # Evaluate final min area for this run
        pts = _bary_to_cart(x_current)
        # Ensure points are strictly inside by clamping barycentric coords
        u = x_current[0::2]
        v = x_current[1::2]
        # Clamp to [1e-6, 1-2e-6] roughly to avoid boundary issues if needed,
        # but L-BFGS-B should keep them in [0,1].
        # The triangle boundary is u=0, v=0, u+v=1.
        # Points on boundary are allowed ("on or inside").
        
        areas = _all_areas_cart(pts)
        min_a = np.min(areas)
        
        if min_a > best_score:
            best_score = min_a
            best_bary = x_current.copy()
            
    # Final Polish on the best found so far
    if best_bary is not None and time.time() < deadline - 2.0:
        try:
            # One more high-precision run
            res = minimize(
                _objective,
                best_bary,
                args=(10000.0, 0.0, start_time, deadline),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            if res.fun < _objective(best_bary, 10000.0, 0.0, start_time, deadline):
                best_bary = res.x
        except Exception:
            pass

    # Convert best barycentric to Cartesian
    if best_bary is None:
        # Fallback
        best_bary = _generate_symmetric_bary()
        
    final_pts = _bary_to_cart(best_bary)
    
    # Safety check: ensure points are within triangle
    # Barycentric coords u, v, w=1-u-v must be >= 0
    # L-BFGS-B with bounds [0,1] on u,v ensures u>=0, v>=0, u<=1, v<=1.
    # We also need u+v <= 1. L-BFGS-B doesn't enforce u+v<=1 directly with simple bounds.
    # However, the penalty function and the nature of the problem usually keep them inside.
    # But to be safe, we should project any point outside back to the triangle.
    
    # Check u+v <= 1
    u = final_pts[:, 0] # Wait, final_pts is Cartesian.
    # Let's re-verify barycentric validity
    u_bary = best_bary[0::2]
    v_bary = best_bary[1::2]
    w_bary = 1.0 - u_bary - v_bary
    
    # If any w_bary < 0, the point is outside the triangle (beyond edge V1-V2)
    # We need to clamp.
    # Simple clamp: if u+v > 1, scale u,v such that u+v=1.
    for i in range(N):
        if u_bary[i] + v_bary[i] > 1.0:
            s = 1.0 / (u_bary[i] + v_bary[i])
            u_bary[i] *= s
            v_bary[i] *= s
        # Also clamp to 0
        u_bary[i] = max(0.0, u_bary[i])
        v_bary[i] = max(0.0, v_bary[i])
        
    # Reconstruct points from clamped barycentric coords
    final_pts = _bary_to_cart(np.concatenate([u_bary, v_bary]))
    
    return final_pts

# EVOLVE-BLOCK-END

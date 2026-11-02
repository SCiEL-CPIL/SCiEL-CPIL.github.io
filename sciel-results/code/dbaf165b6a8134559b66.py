"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

# Vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

# Precompute triplets
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

# Matrix for bary to cart: p = b*V1 + c*V2
# x = b + 0.5c
# y = S3H*c
_V_MATRIX = np.array([[1.0, 0.0],
                      [0.5, S3H]])

DEADLINE = 145  # 5s margin

def _bary_to_cart(ab_flat):
    """Convert flat array of (a, b) pairs to Cartesian coords.
    a, b are first two barycentric coords. c = 1 - a - b.
    We pass (a, b) for each point.
    Input: (2*N,)
    Output: (N, 2)
    """
    ab = ab_flat.reshape(N, 2)
    a = ab[:, 0]
    b = ab[:, 1]
    c = 1.0 - a - b
    # p = b*V1 + c*V2
    # V1 = [1, 0], V2 = [0.5, S3H]
    # x = b*1 + c*0.5
    # y = b*0 + c*S3H
    x = b + 0.5 * c
    y = S3H * c
    return np.column_stack((x, y))

def _min_area_cart(pts):
    """Compute min triangle area for N points."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)

def _objective(ab_flat):
    """
    Objective for minimization.
    Returns negative min area + penalty for invalid points.
    """
    # Check validity: c >= 0 => 1 - a - b >= 0 => a + b <= 1
    # Also a >= 0, b >= 0
    ab = ab_flat.reshape(N, 2)
    a = ab[:, 0]
    b = ab[:, 1]
    
    # Penalty for out of bounds
    # If a < 0 or b < 0 or a+b > 1, penalty is high
    penalty = 0.0
    # We clip for calculation to avoid NaNs, but add penalty
    a_safe = np.clip(a, 0, 1)
    b_safe = np.clip(b, 0, 1)
    # If original was out of bounds, add penalty
    out_of_bounds = (a < 0) | (b < 0) | (a + b > 1)
    if np.any(out_of_bounds):
        # Penalty proportional to violation
        viol = np.maximum(0, -a) + np.maximum(0, -b) + np.maximum(0, a + b - 1)
        penalty = np.sum(viol) * 10.0
    
    pts = _bary_to_cart(ab_flat)
    area = _min_area_cart(pts)
    return -area + penalty

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_ab = None
    best_score = -np.inf  # We want to maximize area, so minimize -area. 
                          # But here we track best_area directly for clarity.
    best_area = 0.0
    
    # Simulated Annealing
    # Initial state: random points
    # We will do a few restarts
    
    n_restarts = 5
    
    for r in range(n_restarts):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Random init
        ab_init = rng.random((2 * N,))
        # Ensure valid start? 
        # Generate valid barycentric coords
        bary = rng.random((N, 3))
        bary /= bary.sum(axis=1, keepdims=True)
        ab_init = bary[:, :2].flatten()
        
        current_ab = ab_init.copy()
        current_obj = _objective(current_ab)
        
        # SA parameters
        T = 0.01  # Initial temperature
        T_min = 1e-6
        cooling = 0.995
        steps_per_iter = 50
        
        iter_count = 0
        while T > T_min:
            if time.time() - start_time > DEADLINE - 5:
                break
                
            for _ in range(steps_per_iter):
                if time.time() - start_time > DEADLINE - 5:
                    break
                    
                # Perturb one coordinate
                idx = rng.integers(0, 2 * N)
                new_ab = current_ab.copy()
                # Random step size
                step = rng.normal(0, 0.05)
                new_ab[idx] += step
                
                # Clip to [0, 1] to keep a, b somewhat bounded, 
                # though a+b<=1 is checked in objective
                new_ab[idx] = np.clip(new_ab[idx], 0, 1)
                
                new_obj = _objective(new_ab)
                delta = new_obj - current_obj
                
                # Accept if better, or with probability exp(-delta/T)
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_ab = new_ab
                    current_obj = new_obj
                    
                    # Check if this is a valid solution and better than global best
                    # Valid means no penalty (or low penalty)
                    # We can check if c >= 0 for all
                    ab = current_ab.reshape(N, 2)
                    a = ab[:, 0]
                    b = ab[:, 1]
                    if np.all(a >= 0) and np.all(b >= 0) and np.all(a + b <= 1 + 1e-9):
                        pts = _bary_to_cart(current_ab)
                        area = _min_area_cart(pts)
                        if area > best_area:
                            best_area = area
                            best_ab = current_ab.copy()
                            
            T *= cooling
            iter_count += 1
            
    if best_ab is None:
        # Fallback
        bary = rng.random((N, 3))
        bary /= bary.sum(axis=1, keepdims=True)
        ab = bary[:, :2].flatten()
        return _bary_to_cart(ab)
        
    return _bary_to_cart(best_ab)
# EVOLVE-BLOCK-END

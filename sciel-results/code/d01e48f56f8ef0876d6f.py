"""
Heilbronn Triangles: Maximize minimum triangle area for 11 points in a unit equilateral triangle.
Strategy: Multi-start Annealed L-BFGS-B optimization using a soft-min surrogate.
"""
import numpy as np
import time
from itertools import combinations
from scipy.optimize import minimize

N = 11
H = np.sqrt(3) / 2.0

# Precompute all C(11,3) = 165 index triples
_IDX = np.array(list(combinations(range(N), 3)))

# Vertices of the equilateral triangle in Cartesian coordinates
# V0 = (0,0), V1 = (1,0), V2 = (0.5, H)
# Barycentric to Cartesian: P = u*V0 + v*V1 + w*V2 where w = 1-u-v
# P = u*(0,0) + v*(1,0) + (1-u-v)*(0.5, H)
# P_x = v + 0.5*(1-u-v) = 0.5 + 0.5v - 0.5u
# P_y = H*(1-u-v)
def _bary_to_cart(uv):
    """
    uv: (N, 2) array of (u, v) barycentric coordinates.
    Returns (N, 2) Cartesian coordinates.
    """
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    # P = v * (1,0) + w * (0.5, H)  (since u*(0,0) is 0)
    px = v * 1.0 + w * 0.5
    py = w * H
    return np.column_stack([px, py])

def _calc_min_area_cart(cart):
    """Calculate the minimum area of triangles formed by points in Cartesian coords."""
    pi = cart[_IDX[:, 0]]
    pj = cart[_IDX[:, 1]]
    pk = cart[_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _soft_min_objective(x, beta, penalty_weight):
    """
    Objective function for L-BFGS-B.
    x: flat array of length 2*N (u, v pairs).
    beta: sharpness of the soft-min.
    penalty_weight: weight for constraint violation penalty.
    Returns: -soft_min_area + penalty. (Minimizing this maximizes area and satisfies constraints).
    """
    uv = x.reshape(N, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    
    # Penalty for violating u >= 0, v >= 0, u+v <= 1
    # We use squared penalties to keep it smooth
    viol_u = np.maximum(0.0, -u)
    viol_v = np.maximum(0.0, -v)
    viol_sum = np.maximum(0.0, u + v - 1.0)
    
    penalty = penalty_weight * (np.sum(viol_u**2) + np.sum(viol_v**2) + np.sum(viol_sum**2))
    
    # Project to valid region for area calculation to avoid weird geometry if slightly out
    # Actually, L-BFGS-B handles bounds, but we might drift if penalty is low.
    # For robustness, clamp for calculation but keep gradient from penalty.
    u_clamped = np.clip(u, 0, 1)
    v_clamped = np.clip(v, 0, 1-u_clamped)
    
    uv_clamped = np.column_stack([u_clamped, v_clamped])
    cart = _bary_to_cart(uv_clamped)
    
    pi = cart[_IDX[:, 0]]
    pj = cart[_IDX[:, 1]]
    pk = cart[_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # Shift for numerical stability
    max_a = np.max(areas)
    if max_a < 1e-12:
        return 1e10 + penalty # Degenerate points
    
    # Avoid overflow in exp
    # exp(-beta * (areas - max_a)) is <= 1
    exp_terms = np.exp(-beta * (areas - max_a))
    log_sum = np.log(np.sum(exp_terms))
    
    soft_min = max_a - log_sum / beta
    
    return -soft_min + penalty

def _generate_initial_configs():
    """Generate diverse initial configurations in barycentric (u,v) space."""
    configs = []
    rng = np.random.default_rng(42)
    
    # 1. Random valid points
    for _ in range(5):
        u = rng.random(N)
        v = rng.random(N) * (1.0 - u)
        configs.append(np.column_stack([u, v]))
        
    # 2. Structured Grid-like
    # Try to spread points roughly evenly
    # 3x3 grid in barycentric: u in {0.1, 0.5, 0.9}, v in {0.1, 0.5, 0.9}
    # Filter valid ones
    grid_u = [0.1, 0.3, 0.5, 0.7, 0.9]
    grid_v = [0.1, 0.3, 0.5, 0.7, 0.9]
    valid_grid = []
    for u in grid_u:
        for v in grid_v:
            if u + v <= 1.0:
                valid_grid.append((u, v))
    if len(valid_grid) >= N:
        # Shuffle to break symmetry slightly
        rng.shuffle(valid_grid)
        configs.append(np.array(valid_grid[:N]))
        
    # 3. Symmetric/Geometric points
    # Vertices: (0,0), (1,0), (0,1) in barycentric (u,v)
    # Note: Barycentric (u,v) maps to:
    # (0,0) -> V0 (0,0)
    # (1,0) -> V1 (1,0)
    # (0,1) -> V2 (0.5, H)
    # Midpoints: (0.5, 0), (0.5, 0.5), (0, 0.5)
    # Centroid: (1/3, 1/3)
    special = [
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 0.0),
        (0.0, 0.5),
        (0.5, 0.5), # This is (0.25, H/2) in cart? 
                    # P = 0.5*(1,0) + 0.5*(0.5,H) = (0.75, H/2). Wait.
                    # Let's re-verify barycentric mapping.
                    # P = u*V0 + v*V1 + w*V2
                    # If u=0.5, v=0.5, w=0: P = 0.5*V1 + 0.5*V2 = 0.5*(1,0) + 0.5*(0.5,H) = (0.75, H/2).
                    # This is midpoint of V1-V2.
                    # Midpoint of V0-V1 is u=0, v=0.5, w=0.5 -> P = 0.5*V1 + 0.5*V0 = (0.5, 0).
                    # Midpoint of V0-V2 is u=0.5, v=0, w=0.5 -> P = 0.5*V0 + 0.5*V2 = (0.25, H/2).
        (1/3, 1/3),
        (0.2, 0.2),
        (0.6, 0.2),
        (0.2, 0.6),
        (0.5, 0.25)
    ]
    configs.append(np.array(special[:N]))
    
    return configs

def construct():
    """
    Place 11 points on or inside a unit-area equilateral triangle to maximize
    the smallest area of a triangle formed by three points.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin for 120s limit
    
    best_score = -1.0
    best_uv = None
    
    configs = _generate_initial_configs()
    
    # Bounds for L-BFGS-B: u in [0, 1], v in [0, 1]
    # Constraint u+v<=1 is handled by penalty, but we can tighten bounds slightly if needed.
    # Standard box bounds:
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Annealing schedule for beta (sharpness of soft-min)
    # Start low to find basin, increase to sharpen
    betas = [10.0, 50.0, 200.0, 1000.0, 5000.0]
    penalty_weights = [100.0, 1000.0, 10000.0, 100000.0, 1000000.0]
    
    # Process each initial config
    for init_idx, init_uv in enumerate(configs):
        if time.time() > deadline - 5.0:
            break
            
        current_uv = init_uv.copy()
        
        # Annealing loop
        for stage, (beta, pw) in enumerate(zip(betas, penalty_weights)):
            if time.time() > deadline - 5.0:
                break
                
            # Determine max iterations based on remaining time and stage
            # Later stages (higher beta) need more precision but take longer per step? 
            # L-BFGS-B is generally fast.
            remaining_time = deadline - time.time() - 5.0
            if remaining_time < 1.0:
                break
            
            # Heuristic: give more iterations to later stages if time permits
            # But don't let one config eat all time.
            # Approx 1-2 seconds per L-BFGS run for this size.
            max_iter = 100
            if stage >= 2:
                max_iter = 200
            
            try:
                res = minimize(
                    _soft_min_objective,
                    current_uv.flatten(),
                    args=(beta, pw),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                # Update if improved
                if res.fun < _soft_min_objective(current_uv.flatten(), beta, pw):
                    current_uv = res.x.reshape(N, 2)
                elif res.success:
                    current_uv = res.x.reshape(N, 2)
                    
            except Exception:
                # If optimization fails, keep previous state
                pass
        
        # Evaluate the true hard-min area for this candidate
        # Clamp to valid region
        u = current_uv[:, 0]
        v = current_uv[:, 1]
        # Ensure valid
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1-u)
        current_uv = np.column_stack([u, v])
        
        cart = _bary_to_cart(current_uv)
        score = _calc_min_area_cart(cart)
        
        if score > best_score:
            best_score = score
            best_uv = current_uv.copy()
            
    # Final Polish on the best candidate found
    if best_uv is not None and time.time() < deadline - 2.0:
        # Use very high beta for final refinement
        beta_final = 10000.0
        pw_final = 1e6
        
        try:
            res = minimize(
                _soft_min_objective,
                best_uv.flatten(),
                args=(beta_final, pw_final),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if res.fun < _soft_min_objective(best_uv.flatten(), beta_final, pw_final):
                best_uv = res.x.reshape(N, 2)
        except Exception:
            pass
            
        # Re-evaluate final score
        u = best_uv[:, 0]
        v = best_uv[:, 1]
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1-u)
        best_uv = np.column_stack([u, v])
        
        cart = _bary_to_cart(best_uv)
        final_score = _calc_min_area_cart(cart)
        
        if final_score > best_score:
            best_score = final_score
            # best_uv is already updated
            
    # If no valid config was found (shouldn't happen), return fallback
    if best_uv is None:
        # Fallback: Random
        rng = np.random.default_rng(123)
        u = rng.random(N)
        v = rng.random(N) * (1.0 - u)
        best_uv = np.column_stack([u, v])
        
    # Convert best barycentric to Cartesian
    u = best_uv[:, 0]
    v = best_uv[:, 1]
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1-u)
    final_uv = np.column_stack([u, v])
    
    points = _bary_to_cart(final_uv)
    
    return points

if __name__ == "__main__":
    # Quick self-test
    start = time.time()
    pts = construct()
    end = time.time()
    print(f"Time taken: {end - start:.2f}s")
    print(f"Points:\n{pts}")
    cart = pts
    pi = cart[_IDX[:, 0]]
    pj = cart[_IDX[:, 1]]
    pk = cart[_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    print(f"Min Area: {np.min(areas):.6f}")

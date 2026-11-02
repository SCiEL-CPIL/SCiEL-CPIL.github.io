"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Bisection on the target minimum area 'A'.
   - We want to find the largest A such that there exists a configuration 
     of 11 points where all 165 triangle areas are >= A.
2. For a given A, solve the feasibility problem using Simulated Annealing.
   - Objective: Minimize the "violation" = sum of (A - area_i)^2 for all areas < A.
     Or simpler: Minimize sum(max(0, A - area_i)).
   - Using barycentric coordinates (u, v) for each point (w=1-u-v) to 
     naturally stay inside the triangle.
   - Simulated Annealing allows escaping local minima which are prevalent 
     in this non-convex landscape.
3. Vectorized area calculation for speed.
"""

import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 148.0  # Safety margin before 150s limit

# Pre-compute triplet indices for vectorized area calculation
_triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triplets.append((i, j, k))
_triplets = np.array(_triplets)
_i_idx = _triplets[:, 0]
_j_idx = _triplets[:, 1]
_k_idx = _triplets[:, 2]

def _bary_to_cart(bary_pts):
    """
    Convert barycentric coordinates (u, v) to Cartesian (x, y).
    w = 1 - u - v
    P = u*V0 + v*V1 + w*V2
    V0=(0,0), V1=(1,0), V2=(0.5, sqrt(3)/2)
    x = v + 0.5*(1-u-v) = 0.5 + 0.5v - 0.5u
    y = S3H*(1-u-v)
    """
    u = bary_pts[:, 0]
    v = bary_pts[:, 1]
    x = 0.5 + 0.5 * v - 0.5 * u
    y = S3H * (1 - u - v)
    return np.column_stack((x, y))

def _min_triangle_area_cart(pts):
    """Compute the minimum area of any triangle formed by three points in Cartesian coords."""
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_j_idx] # Wait, standard cross product needs k-i and j-i or similar.
    # Let's stick to the robust cross product from the original code
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j[:, 0] - p_i[:, 0]
    dy1 = p_j[:, 1] - p_i[:, 1]
    dx2 = p_k[:, 0] - p_i[:, 0]
    dy2 = p_k[:, 1] - p_i[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas), areas

def _penalty_objective(bary_pts, target_area):
    """
    Calculate the sum of violations: sum(max(0, target_area - area_i)).
    We want this to be 0.
    """
    cart_pts = _bary_to_cart(bary_pts)
    _, areas = _min_triangle_area_cart(cart_pts)
    violations = np.maximum(0, target_area - areas)
    return np.sum(violations)

def _simulated_annealing_feasibility(bary_init, target_area, max_iter=2000, start_temp=0.01, end_temp=0.0001, rng=None):
    """
    Attempt to find a configuration where all areas >= target_area.
    Returns (best_bary_pts, best_penalty).
    """
    if rng is None:
        rng = random.Random()
        
    current_bary = np.array(bary_init, dtype=float)
    current_penalty = _penalty_objective(current_bary, target_area)
    best_bary = current_bary.copy()
    best_penalty = current_penalty
    
    # Clamp initial to valid barycentric space (u>=0, v>=0, u+v<=1)
    def clamp_bary(b):
        u = b[:, 0]
        v = b[:, 1]
        # Simple projection to simplex
        # u, v >= 0
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
        # u + v <= 1
        s = u + v
        mask = s > 1
        if np.any(mask):
            # Scale down
            factor = 1.0 / s[mask]
            u[mask] *= factor
            v[mask] *= factor
        return np.column_stack((u, v))
        
    current_bary = clamp_bary(current_bary)
    current_penalty = _penalty_objective(current_bary, target_area)
    best_bary = current_bary.copy()
    best_penalty = current_penalty
    
    # If already feasible, return immediately
    if current_penalty < 1e-9:
        return best_bary, 0.0

    for i in range(max_iter):
        # Cooling schedule (geometric)
        temp = start_temp * (end_temp / start_temp) ** (i / max_iter)
        
        # Generate neighbor
        # Random walk in barycentric coordinates
        noise = np.random.normal(0, temp, bary_init.shape)
        neighbor_bary = current_bary + noise
        neighbor_bary = clamp_bary(neighbor_bary)
        
        neighbor_penalty = _penalty_objective(neighbor_bary, target_area)
        
        # Acceptance probability
        delta = neighbor_penalty - current_penalty
        if delta < 0:
            # Better, always accept
            current_bary = neighbor_bary
            current_penalty = neighbor_penalty
            if current_penalty < best_penalty:
                best_bary = current_bary.copy()
                best_penalty = current_penalty
        else:
            # Worse, accept with probability exp(-delta / temp)
            if np.random.random() < np.exp(-delta / temp):
                current_bary = neighbor_bary
                current_penalty = neighbor_penalty
                
        # Early termination if feasible
        if best_penalty < 1e-9:
            break
            
    return best_bary, best_penalty

def _structured_bary_init():
    """
    Create a structured initialization in barycentric coordinates.
    Vertices: (1,0), (0,1), (0,0) in (u,v) space? 
    V0=(0,0) cart -> (1,0) bary? 
    Let's check:
    P = u*V0 + v*V1 + w*V2.
    If P=V0, u=1, v=0, w=0.
    If P=V1, u=0, v=1, w=0.
    If P=V2, u=0, v=0, w=1.
    """
    bary_pts = []
    # Vertices
    bary_pts.append([1.0, 0.0])
    bary_pts.append([0.0, 1.0])
    bary_pts.append([0.0, 0.0])
    
    # Midpoints of edges
    # Edge V0-V1: u=0.5, v=0.5, w=0
    bary_pts.append([0.5, 0.5])
    # Edge V1-V2: u=0, v=0.5, w=0.5
    bary_pts.append([0.0, 0.5])
    # Edge V2-V0: u=0.5, v=0, w=0.5
    bary_pts.append([0.5, 0.0])
    
    # Centroid
    bary_pts.append([1/3, 1/3])
    
    # Quarter points
    bary_pts.append([0.75, 0.25])
    bary_pts.append([0.25, 0.75])
    bary_pts.append([0.25, 0.25])
    
    # Two more spread out
    bary_pts.append([0.6, 0.1])
    bary_pts.append([0.1, 0.6])
    
    return np.array(bary_pts)

def _random_bary_init(rng):
    pts = []
    for _ in range(N):
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        pts.append([u, v])
    return np.array(pts)

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses Bisection on target area with Simulated Annealing for feasibility checks.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    # Initial guess for bounds
    # Max area is 1 (the whole triangle). Min area 0.
    # A good starting high bound is the score of the structured init.
    init_bary = _structured_bary_init()
    init_cart = _bary_to_cart(init_bary)
    initial_score, _ = _min_triangle_area_cart(init_cart)
    
    low = 0.0
    high = initial_score * 1.5 # Try to go higher
    
    # First, do a quick optimization to find a better high bound
    # Run a few SA iterations on the structured init to see if we can push it up
    # Actually, let's just use the structured init score as a conservative high bound 
    # and let bisection find the max. But if we start too low, we waste time.
    # Let's try to improve the initial configuration with a quick SA run 
    # maximizing the min area directly? No, SA is for feasibility.
    
    # Let's run a few SA runs to try to beat the initial score to set a better 'high'
    # Or just set high = initial_score and expand if feasible?
    # Bisection usually requires a known feasible low and infeasible high.
    # We know low=0 is feasible.
    # We need an infeasible high.
    # Let's try high = initial_score * 2.0. If that's feasible, we expand.
    
    # Pre-allocate best result
    best_bary_global = init_bary.copy()
    best_score_global = initial_score
    
    # Bisection loop
    # We have limited time. Each SA run takes ~0.1-0.5s depending on iterations.
    # 150s / 0.3s ~ 500 iterations.
    # Bisection takes log2(range/precision) steps.
    # Range ~ 1.0, precision 1e-4 -> ~14 steps.
    # This is very fast. We can afford multiple restarts for the SA in each bisection step.
    
    max_sa_iters = 500
    sa_starts = 5 # Number of SA restarts per bisection step
    
    while time.time() - start_time < DEADLINE - 5:
        mid = (low + high) / 2.0
        
        # Try to find a feasible configuration for area >= mid
        found_feasible = False
        best_bary_for_mid = None
        best_pen_for_mid = float('inf')
        
        for _ in range(sa_starts):
            if time.time() - start_time > DEADLINE - 10:
                break
                
            # Create diverse initializations for SA
            # Mix of structured and random
            if rng.random() < 0.5:
                bary_init = _structured_bary_init()
                # Add noise
                noise = np.random.normal(0, 0.05, bary_init.shape)
                bary_init += noise
            else:
                bary_init = _random_bary_init(rng)
            
            bary_res, pen_res = _simulated_annealing_feasibility(
                bary_init, mid, max_iter=max_sa_iters, 
                start_temp=0.05, end_temp=0.001, rng=rng
            )
            
            if pen_res < best_pen_for_mid:
                best_pen_for_mid = pen_res
                best_bary_for_mid = bary_res
                
            if pen_res < 1e-7:
                found_feasible = True
                break
        
        if found_feasible:
            # We found a config for 'mid'. Update low bound.
            low = mid
            # Update global best if this 'mid' is better than current global best
            if mid > best_score_global:
                best_score_global = mid
                best_bary_global = best_bary_for_mid.copy()
        else:
            # Not feasible for 'mid'. Update high bound.
            high = mid
            
        # Check convergence
        if high - low < 1e-4:
            break

    # Final polish: Use the best_bary_global found.
    # It might be slightly "feasible" for best_score_global.
    # Let's do one final SA run to try to maximize the min area starting from best_bary_global
    # using a direct objective? No, stick to the plan.
    
    # Convert best bary to cartesian
    final_cart = _bary_to_cart(best_bary_global)
    
    # Verify it's inside (barycentric ensures this, but double check)
    x = final_cart[:, 0]
    y = final_cart[:, 1]
    if np.any(y < -1e-9) or np.any(y > S3H + 1e-9) or np.any(x < -1e-9) or np.any(x > 1 + 1e-9):
        # Fallback to structured if something went wrong
        final_cart = _bary_to_cart(_structured_bary_init())
        
    return final_cart

if __name__ == "__main__":
    pts = construct()
    score, _ = _min_triangle_area_cart(pts)
    print(f"Score: {score}")

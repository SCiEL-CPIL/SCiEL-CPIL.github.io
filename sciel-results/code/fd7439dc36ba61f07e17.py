"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Use Barycentric coordinates to naturally enforce the triangle constraint.
   Points are represented as (u, v) where u,v >= 0 and u+v <= 1.
   This eliminates the need for complex geometric projection and constraint handling.
2. Use a Soft-Min (Log-Sum-Exp) approximation of the minimum area objective.
   This creates a smooth, differentiable landscape, avoiding the non-smoothness
   of the hard 'min' function which traps gradient-based methods.
3. Employ Simulated Annealing with gradient-based local refinement.
   The annealing allows escaping local minima, while the smooth objective
   allows for efficient local improvements.
"""
import numpy as np
import time
import random

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

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

def _bary_to_cart(bary):
    """
    Convert barycentric coordinates (u, v) to Cartesian coordinates.
    The third barycentric coordinate is w = 1 - u - v.
    Vertices: A(0,0), B(1,0), C(0.5, sqrt(3)/2)
    P = u*B + v*C + w*A = u*(1,0) + v*(0.5, sqrt(3)/2) + (1-u-v)*(0,0)
    x = u + 0.5v
    y = sqrt(3)/2 * v
    """
    u = bary[:, 0]
    v = bary[:, 1]
    x = u + 0.5 * v
    y = S3H * v
    return np.column_stack((x, y))

def _cart_to_bary(pts):
    """
    Convert Cartesian points to barycentric coordinates (u, v).
    y = S3H * v  => v = y / S3H
    x = u + 0.5 * v => u = x - 0.5 * v
    """
    x = pts[:, 0]
    y = pts[:, 1]
    v = y / S3H
    u = x - 0.5 * v
    return np.column_stack((u, v))

def _vectorized_areas_cart(pts):
    """
    Compute all triangle areas from Cartesian points.
    Returns array of shape (165,)
    """
    p_i = pts[_i_idx]
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return areas

def _soft_min_area(pts, beta):
    """
    Compute the soft minimum of the triangle areas using Log-Sum-Exp.
    soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
    As beta -> infinity, this approaches min(x).
    """
    areas = _vectorized_areas_cart(pts)
    # Avoid underflow/overflow issues
    # We want to minimize -soft_min, which is equivalent to maximizing soft_min.
    # But we are using this as a cost function for minimization? 
    # Let's define the objective as -soft_min_area to maximize it.
    # Or simply minimize the negative of the soft min.
    
    # For numerical stability:
    # log_sum_exp(-beta * areas) = max(-beta * areas) + log(sum(exp(-beta * areas - max(-beta * areas))))
    neg_beta_areas = -beta * areas
    max_neg = np.max(neg_beta_areas)
    if max_neg == -np.inf:
        return 0.0
        
    log_sum = max_neg + np.log(np.sum(np.exp(neg_beta_areas - max_neg)))
    soft_min_val = -log_sum / beta
    
    return soft_min_val

def _min_triangle_area(pts):
    """Compute the hard minimum area."""
    areas = _vectorized_areas_cart(pts)
    return np.min(areas)

def _project_bary(bary):
    """
    Project barycentric coordinates to the valid simplex:
    u >= 0, v >= 0, u + v <= 1.
    """
    bary = np.array(bary, dtype=float)
    # Simple projection: clip negative values, then normalize if sum > 1?
    # Actually, for a triangle, the valid region is a simplex.
    # If u < 0, set u=0. If v < 0, set v=0.
    # If u+v > 1, we need to project onto the line u+v=1.
    
    u = bary[:, 0]
    v = bary[:, 1]
    
    # Clip negatives
    u[u < 0] = 0
    v[v < 0] = 0
    
    # Check sum
    sums = u + v
    mask = sums > 1.0
    if np.any(mask):
        # For points outside, scale down to sum=1
        scale = 1.0 / sums[mask]
        u[mask] *= scale
        v[mask] *= scale
        
    bary[:, 0] = u
    bary[:, 1] = v
    return bary

def _random_bary(rng):
    """Generate random points in barycentric coordinates."""
    pts = []
    for _ in range(N):
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            # Rejection sampling or scaling
            # Scaling preserves distribution shape somewhat but biases towards center
            # Rejection is cleaner
            while u + v > 1:
                u = rng.random()
                v = rng.random()
        pts.append([u, v])
    return np.array(pts)

def _structured_init_bary():
    """
    Create a structured initialization in barycentric coordinates.
    Vertices: A(0,0)->(0,0), B(1,0)->(1,0), C(0.5, S3H)->(0,1)
    Midpoints: AB(0.5,0)->(0.5,0), BC(0.75, S3H/2)->(0.5,0.5), CA(0.25, S3H/2)->(0,0.5)
    Center: (0.5, S3H/3) -> u=0.5-0.5*(1/3)=1/3, v=1/3. So (1/3, 1/3).
    """
    pts = [
        [0.0, 0.0], # A
        [1.0, 0.0], # B
        [0.0, 1.0], # C
        [0.5, 0.0], # Mid AB
        [0.5, 0.5], # Mid BC
        [0.0, 0.5], # Mid CA
        [1/3, 1/3], # Center
        [0.25, 0.0], # 1/4 AB
        [0.75, 0.0], # 3/4 AB
        [0.0, 0.25], # 1/4 CA
        [0.0, 0.75]  # 3/4 CA
    ]
    return np.array(pts)

def _local_refinement(bary, beta, max_iter=50, step_size=0.05):
    """
    Perform local refinement using gradient descent on the soft-min objective.
    We approximate the gradient numerically or use simple coordinate descent.
    Given time constraints, simple stochastic coordinate descent is robust.
    """
    bary = np.array(bary, dtype=float)
    current_score = -_soft_min_area(_bary_to_cart(bary), beta) # Minimize negative score
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            for dim in range(2):
                # Try small perturbations
                for delta in [step_size, -step_size, step_size*0.5, -step_size*0.5]:
                    new_bary = bary.copy()
                    new_bary[i, dim] += delta
                    new_bary = _project_bary(new_bary)
                    
                    new_pts = _bary_to_cart(new_bary)
                    new_score = -_soft_min_area(new_pts, beta)
                    
                    if new_score < current_score:
                        bary = new_bary
                        current_score = new_score
                        improved = True
        
        if not improved:
            break
            
    return bary

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses Simulated Annealing with Barycentric coordinates and Soft-Min objective.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_bary = None
    best_score = -1.0
    
    # Initializations
    inits = []
    
    # 1. Structured
    p_struct = _structured_init_bary()
    inits.append(p_struct)
    
    # 2. Random
    for _ in range(5):
        p_rand = _random_bary(rng)
        inits.append(p_rand)
        
    # 3. Greedy-like (maximize min distance in barycentric space is not ideal, 
    #    but let's just add a few more randoms)
    for _ in range(5):
        p_rand = _random_bary(rng)
        inits.append(p_rand)

    # Simulated Annealing Parameters
    T_start = 0.1
    T_end = 1e-5
    cooling_rate = 0.99
    beta_start = 10.0
    beta_end = 1000.0
    
    # We will run annealing on each initial guess, or just one global annealing?
    # Global annealing is better for escaping local minima.
    # Let's pick the best initial guess to start the main annealing.
    
    best_init_bary = None
    best_init_score = -1.0
    for init in inits:
        pts = _bary_to_cart(init)
        score = _min_triangle_area(pts)
        if score > best_init_score:
            best_init_score = score
            best_init_bary = init.copy()
            
    current_bary = best_init_bary
    current_score = -_soft_min_area(_bary_to_cart(current_bary), beta_start)
    
    # Annealing Loop
    T = T_start
    beta = beta_start
    
    while T > T_end:
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Generate a neighbor
        # Pick a random point and a random dimension
        i = rng.randint(0, N - 1)
        dim = rng.randint(0, 1)
        # Step size proportional to temperature?
        step = rng.gauss(0, 0.1 * (T / T_start + 0.1))
        
        new_bary = current_bary.copy()
        new_bary[i, dim] += step
        new_bary = _project_bary(new_bary)
        
        new_pts = _bary_to_cart(new_bary)
        new_score = -_soft_min_area(new_pts, beta)
        
        delta_E = new_score - current_score
        
        # Accept if better, or with probability exp(-delta_E / T)
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            current_bary = new_bary
            current_score = new_score
            
            # Update global best based on HARD score
            hard_score = _min_triangle_area(new_pts)
            if hard_score > best_score:
                best_score = hard_score
                best_bary = current_bary.copy()
                
        # Cool down
        T *= cooling_rate
        # Increase beta (sharpen the soft-min)
        beta *= 1.01
        
        # Periodic local refinement to escape shallow minima
        if int(T_start / T) % 50 == 0:
             # Do a quick local refinement
             refined_bary = _local_refinement(current_bary, beta, max_iter=20, step_size=0.05)
             refined_pts = _bary_to_cart(refined_bary)
             refined_hard_score = _min_triangle_area(refined_pts)
             if refined_hard_score > best_score:
                 best_score = refined_hard_score
                 best_bary = refined_bary.copy()
             # Accept the refined position if it's better in soft score
             refined_soft_score = -_soft_min_area(refined_pts, beta)
             if refined_soft_score < current_score:
                 current_bary = refined_bary
                 current_score = refined_soft_score

    # Final Local Refinement on the best found configuration
    if best_bary is not None:
        # Increase beta significantly for final refinement
        final_beta = 1000.0
        refined_bary = _local_refinement(best_bary, final_beta, max_iter=100, step_size=0.01)
        refined_pts = _bary_to_cart(refined_bary)
        refined_hard_score = _min_triangle_area(refined_pts)
        if refined_hard_score > best_score:
            best_score = refined_hard_score
            best_bary = refined_bary.copy()

    # Fallback if annealing failed completely
    if best_bary is None:
        best_bary = best_init_bary
        
    # Convert to Cartesian
    best_pts = _bary_to_cart(best_bary)
    
    # Ensure validity (should be guaranteed by barycentric projection, but just in case)
    # Check if points are inside
    for p in best_pts:
        x, y = p
        if y < -1e-9 or x < -1e-9 or x > 1 + 1e-9:
             # This shouldn't happen with _project_bary, but if it does, project
             bary = _cart_to_bary(np.array([p]))
             bary = _project_bary(bary)
             best_pts[best_pts.tolist().index(list(p))] = _bary_to_cart(bary)[0]
             
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area(pts)}")

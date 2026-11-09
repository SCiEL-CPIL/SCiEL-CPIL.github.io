import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 11
H = np.sqrt(3) / 2

def soft_min_area(points, beta=100.0):
    """
    Compute a smooth approximation of the minimum triangle area using log-sum-exp.
    This makes the objective differentiable and suitable for gradient-based methods.
    """
    p1 = points[idx_i]
    p2 = points[idx_j]
    p3 = points[idx_k]
    
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # This is a smooth approximation of min(areas)
    max_area = np.max(areas)
    exp_terms = np.exp(-beta * (areas - max_area))
    soft_min = max_area - np.log(np.sum(exp_terms)) / beta
    
    return soft_min

def simulate_annealing(points, start_time, deadline, initial_temp=0.1, cooling_rate=0.995, max_iter=5000):
    """
    Simulated annealing on the smooth soft-min objective.
    """
    current_points = points.copy()
    current_score = -soft_min_area(current_points, beta=50.0)
    best_points = current_points.copy()
    best_score = current_score
    
    rng = np.random.default_rng(42)
    temperature = initial_temp
    
    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        # Generate a random perturbation
        perturbation = rng.normal(0, temperature, size=current_points.shape)
        new_points = current_points + perturbation
        
        # Check if new points are inside the triangle
        if not _all_inside(new_points):
            continue
        
        new_score = -soft_min_area(new_points, beta=50.0)
        
        # Accept if better or with probability exp(-delta/T)
        if new_score > current_score or rng.random() < np.exp((new_score - current_score) / temperature):
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        
        # Cool down
        temperature *= cooling_rate
        
        # Early stopping if temperature too low
        if temperature < 1e-8:
            break
    
    return best_points

# Precompute indices for vectorized area calculation
# i < j < k
idx_i, idx_j, idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            idx_i.append(i)
            idx_j.append(j)
            idx_k.append(k)

idx_i = np.array(idx_i)
idx_j = np.array(idx_j)
idx_k = np.array(idx_k)

def get_full_points(reduced_params):
    """
    Reconstruct 11x2 points from 11 reduced parameters.
    reduced_params: [cy, x1, y1, ..., x5, y5]
    Structure:
    - Point 0: (0.5, cy)
    - Points 1-5: (x_i, y_i) in right half (x >= 0.5)
    - Points 6-10: (1-x_i, y_i) in left half (x <= 0.5)
    """
    cy = reduced_params[0]
    pts = np.empty((11, 2))
    pts[0] = [0.5, cy]
    
    right_pts = reduced_params[1:].reshape(5, 2)
    # Ensure x is in [0.5, 1]
    # The bounds in DE will handle this, but we can clip for safety in evaluation
    # However, for the objective function, we want to penalize invalid points.
    # So we just construct them as is.
    pts[1:6] = right_pts
    pts[6:11] = np.column_stack([1.0 - right_pts[:, 0], right_pts[:, 1]])
    
    return pts

def min_triangle_area_vec(points):
    """
    points: (N, 2) array
    Returns the minimum area of any triangle formed by 3 points.
    """
    p1 = points[idx_i]
    p2 = points[idx_j]
    p3 = points[idx_k]
    
    # Area = 0.5 * | (p2-p1) x (p3-p1) |
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _all_inside(points):
    """
    Check if all points are inside or on the boundary of the equilateral triangle.
    Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    """
    # Check y >= 0
    if np.any(points[:, 1] < -1e-10):
        return False
    
    # Check y <= sqrt(3) * x for left side (x <= 0.5)
    left_mask = points[:, 0] <= 0.5
    if np.any(left_mask):
        if np.any(points[left_mask, 1] > np.sqrt(3) * points[left_mask, 0] + 1e-10):
            return False
    
    # Check y <= sqrt(3) * (1 - x) for right side (x >= 0.5)
    right_mask = points[:, 0] >= 0.5
    if np.any(right_mask):
        if np.any(points[right_mask, 1] > np.sqrt(3) * (1 - points[right_mask, 0]) + 1e-10):
            return False
    
    return True

def is_inside_reduced(reduced_params):
    """
    Check if the reduced points correspond to a valid configuration inside the triangle.
    """
    cy = reduced_params[0]
    if cy < -1e-9 or cy > H + 1e-9:
        return False
        
    right_pts = reduced_params[1:].reshape(5, 2)
    x = right_pts[:, 0]
    y = right_pts[:, 1]
    
    # Right half constraints:
    # 1. x >= 0.5
    if np.any(x < 0.5 - 1e-9):
        return False
    # 2. y >= 0
    if np.any(y < -1e-9):
        return False
    # 3. y <= sqrt(3) * (1 - x)
    if np.any(y > np.sqrt(3) * (1 - x) + 1e-9):
        return False
        
    return True

def objective_reduced(params):
    """
    Objective function for minimization on reduced parameters.
    Returns -min_area if valid, penalty if invalid.
    """
    if not is_inside_reduced(params):
        return 1.0  # Penalty
        
    points = get_full_points(params)
    area = min_triangle_area_vec(points)
    return -area

def refine_local(reduced_params, start_time, deadline):
    """
    Perform local refinement using simulated annealing on smooth surrogate,
    followed by Nelder-Mead polish.
    """
    if time.time() > deadline:
        return reduced_params
    
    best_params = reduced_params.copy()
    best_score = -objective_reduced(best_params)
    
    # Phase 1: Simulated annealing on full 22D space
    points = get_full_points(best_params)
    sa_deadline = deadline - 10  # Leave time for final polish
    
    if time.time() < sa_deadline:
        sa_points = simulate_annealing(points, start_time, sa_deadline, 
                                       initial_temp=0.05, cooling_rate=0.997, max_iter=8000)
        sa_score = min_triangle_area_vec(sa_points)
        
        if sa_score > best_score:
            best_score = sa_score
            # Convert back to reduced params if symmetric, or use directly
            # For now, we'll try to extract symmetric params
            best_params = _points_to_reduced_params(sa_points)
    
    # Phase 2: Nelder-Mead polish on reduced space
    if time.time() < deadline - 2:
        rng = np.random.default_rng(42)
        for attempt in range(5):
            if time.time() > deadline - 2:
                break
            
            if attempt == 0:
                x0 = best_params.copy()
            else:
                perturbation = rng.normal(0, 0.005, size=best_params.shape)
                x0 = best_params + perturbation
                if not is_inside_reduced(x0):
                    continue
            
            try:
                res = minimize(
                    objective_reduced,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                
                if time.time() > deadline - 2:
                    break
                    
                new_params = res.x
                if is_inside_reduced(new_params):
                    new_score = -res.fun
                    if new_score > best_score:
                        best_score = new_score
                        best_params = new_params.copy()
                        
            except Exception:
                continue
    
    return best_params

def _points_to_reduced_params(points):
    """
    Convert 11 points to reduced symmetric parameterization.
    Assumes points are approximately symmetric about x=0.5.
    """
    # Point 0 should be at x=0.5
    cy = points[0, 1]
    
    # Average left and right halves to enforce symmetry
    right_pts = points[1:6].copy()
    left_pts = points[6:11].copy()
    
    # For each pair, average to get symmetric representative
    avg_right = np.zeros((5, 2))
    for i in range(5):
        # right_pts[i] and left_pts[i] should be mirrors
        # Mirror of (x, y) is (1-x, y)
        mirror_right = np.array([1 - left_pts[i, 0], left_pts[i, 1]])
        avg_right[i] = (right_pts[i] + mirror_right) / 2
    
    # Ensure x >= 0.5
    avg_right[:, 0] = np.maximum(avg_right[:, 0], 0.5)
    
    return np.concatenate([[cy], avg_right.flatten()])

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Bounds for reduced parameters:
    # cy: [0, H]
    # x_i: [0.5, 1]
    # y_i: [0, H] (will be constrained by is_inside)
    
    bounds = [(0.0, H)]  # cy
    for i in range(5):
        bounds.append((0.5, 1.0))  # x_i
        bounds.append((0.0, H))    # y_i
        
    best_params = None
    best_score = -float('inf')
    
    # Use Differential Evolution for global search on reduced space
    try:
        result = differential_evolution(
            objective_reduced,
            bounds,
            maxiter=300,  # More iterations possible due to lower dimension
            popsize=20,
            tol=0.01,
            seed=42,
            polish=True,
            init='sobol'
        )
        
        candidate_params = result.x
        if is_inside_reduced(candidate_params):
            candidate_score = -result.fun
            if candidate_score > best_score:
                best_score = candidate_score
                best_params = candidate_params.copy()
                
    except Exception:
        pass
        
    if best_params is None:
        # Fallback: random valid reduced params
        rng = np.random.default_rng(42)
        for _ in range(100):
            cy = rng.uniform(0, H)
            xs = rng.uniform(0.5, 1.0, 5)
            # Max y for given x is sqrt(3)*(1-x)
            max_ys = np.sqrt(3) * (1 - xs)
            ys = rng.uniform(0, max_ys)
            params = np.concatenate([[cy], np.column_stack([xs, ys]).flatten()])
            if is_inside_reduced(params):
                best_params = params
                best_score = -objective_reduced(best_params)
                break
        if best_params is None:
            # Last resort
            best_params = np.array([H/3] + [0.5, 0.0]*5) # Degenerate, but valid structure
            best_params[1] = 0.5
            best_params[3] = 0.0
            # Ensure it's inside
            if not is_inside_reduced(best_params):
                 best_params = np.zeros(11)
                 best_params[0] = H/3
                 for i in range(5):
                     best_params[1+2*i] = 0.5
                     best_params[2+2*i] = 0.0

    # Local refinement with simulated annealing + polish
    if time.time() < deadline - 5:
        best_params = refine_local(best_params, start_time, deadline)
    
    # Final validation and return
    final_points = get_full_points(best_params)
    
    # Ensure points are inside triangle (clamp if needed)
    if not _all_inside(final_points):
        # Try to fix by projecting to nearest valid point
        for i in range(N):
            x, y = final_points[i]
            if y < 0:
                y = 0
            if x <= 0.5 and y > np.sqrt(3) * x:
                y = np.sqrt(3) * x
            if x >= 0.5 and y > np.sqrt(3) * (1 - x):
                y = np.sqrt(3) * (1 - x)
            final_points[i] = [x, y]
    
    return final_points

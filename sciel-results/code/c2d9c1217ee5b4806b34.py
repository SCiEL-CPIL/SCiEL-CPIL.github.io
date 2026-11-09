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

def sinkhorn_objective(points, t, beta=50.0, gamma=1.0):
    """
    Sinkhorn-inspired objective: maximize t subject to soft-constraints that
    all triangle areas are >= t. Uses log-sum-exp for smoothness.
    
    The idea: minimize the "violation" of the constraint areas >= t,
    weighted by a softmax to focus on the most violated constraints.
    """
    p1 = points[idx_i]
    p2 = points[idx_j]
    p3 = points[idx_k]
    
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    
    # For each triangle, compute violation: max(0, t - area)
    # Smooth version using softplus: log(1 + exp(gamma * (t - area))) / gamma
    violations = np.logaddexp(0, gamma * (t - areas)) / gamma
    
    # Sinkhorn-style: weight violations by their relative importance
    # Use softmax weighting to focus on the most violated constraints
    weights = np.exp(-gamma * violations)
    weights /= np.sum(weights)
    
    # Objective: minimize weighted sum of violations, minus t (to push t up)
    obj = np.sum(weights * violations) - t
    
    return obj

def gradient_descent_sinkhorn(points, start_time, deadline, 
                                max_iter=2000, lr_init=0.01, lr_decay=0.995,
                                beta_init=20.0, beta_final=200.0,
                                gamma=1.0, t_init=None):
    """
    Gradient descent with Sinkhorn-inspired smooth objective.
    Uses adaptive beta (soft-min sharpness) and gamma (violation sharpness).
    """
    current_points = points.copy()
    n_params = current_points.shape[0] * 2
    
    # Initialize t to current min area
    if t_init is None:
        t = min_triangle_area_vec(current_points) * 0.9
    else:
        t = t_init
    
    best_points = current_points.copy()
    best_t = t
    best_min_area = min_triangle_area_vec(best_points)
    
    rng = np.random.default_rng(42)
    lr = lr_init
    beta = beta_init
    
    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        # Animate beta from beta_init to beta_final
        progress = iteration / max_iter
        beta = beta_init + (beta_final - beta_init) * progress
        
        # Compute gradient of sinkhorn_objective w.r.t. points and t
        # Use numerical gradient for simplicity and robustness
        eps = 1e-6
        grad_points = np.zeros_like(current_points)
        grad_t = 0.0
        
        # Gradient w.r.t. t is straightforward: d/dt [sum(w*v) - t] = -sum(w) + sum(w * dv/dt)
        # But let's use numerical for both for consistency
        
        # Numerical gradient w.r.t. t
        obj_plus = sinkhorn_objective(current_points, t + eps, beta=beta, gamma=gamma)
        obj_minus = sinkhorn_objective(current_points, t - eps, beta=beta, gamma=gamma)
        grad_t = (obj_plus - obj_minus) / (2 * eps)
        
        # Numerical gradient w.r.t. points (vectorized)
        for i in range(N):
            for j in range(2):
                old_val = current_points[i, j]
                current_points[i, j] = old_val + eps
                obj_p = sinkhorn_objective(current_points, t, beta=beta, gamma=gamma)
                current_points[i, j] = old_val - eps
                obj_m = sinkhorn_objective(current_points, t, beta=beta, gamma=gamma)
                current_points[i, j] = old_val
                grad_points[i, j] = (obj_p - obj_m) / (2 * eps)
        
        # Gradient descent step
        new_points = current_points - lr * grad_points
        new_t = t - lr * grad_t
        
        # Keep t non-negative
        new_t = max(0.0, new_t)
        
        # Project points back into triangle
        if not _all_inside(new_points):
            # Try smaller step
            new_points = current_points - 0.5 * lr * grad_points
            if not _all_inside(new_points):
                new_points = current_points - 0.1 * lr * grad_points
                if not _all_inside(new_points):
                    # Skip this iteration
                    lr *= 0.9
                    continue
        
        # Evaluate actual min area (not the smooth surrogate)
        new_min_area = min_triangle_area_vec(new_points)
        
        # Accept if better
        if new_min_area > best_min_area:
            best_min_area = new_min_area
            best_points = new_points.copy()
            best_t = new_t
        
        # Always move to new position (even if not better, for exploration)
        current_points = new_points
        t = new_t
        
        # Decay learning rate
        lr *= lr_decay
        
        # Early stopping if lr too small
        if lr < 1e-10:
            break
    
    return best_points

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

def multi_start_sinkhorn(points_list, start_time, deadline, max_iter_per_start=500):
    """
    Run Sinkhorn-inspired gradient descent from multiple starting points.
    """
    best_points = points_list[0].copy()
    best_score = min_triangle_area_vec(best_points)
    
    n_starts = len(points_list)
    # Allocate time evenly
    time_per_start = (deadline - 10 - start_time) / max(1, n_starts)
    
    for i, init_points in enumerate(points_list):
        if time.time() > deadline - 10:
            break
        
        start_i = time.time()
        deadline_i = min(deadline - 10, start_i + time_per_start)
        
        result = gradient_descent_sinkhorn(
            init_points, start_i, deadline_i,
            max_iter=max_iter_per_start,
            lr_init=0.005,
            lr_decay=0.998,
            beta_init=20.0,
            beta_final=150.0,
            gamma=1.0
        )
        
        score = min_triangle_area_vec(result)
        if score > best_score:
            best_score = score
            best_points = result.copy()
    
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
    Perform local refinement using Sinkhorn-inspired gradient descent,
    followed by Nelder-Mead polish.
    """
    if time.time() > deadline:
        return reduced_params
    
    best_params = reduced_params.copy()
    best_score = -objective_reduced(best_params)
    
    # Phase 1: Sinkhorn-inspired gradient descent with multiple starts
    points = get_full_points(best_params)
    sinkhorn_deadline = deadline - 15  # Leave time for final polish
    
    if time.time() < sinkhorn_deadline:
        # Generate multiple starting points
        rng = np.random.default_rng(42)
        start_points = [points.copy()]
        
        # Add perturbed versions
        for _ in range(4):
            perturbation = rng.normal(0, 0.02, size=points.shape)
            new_pts = points + perturbation
            if _all_inside(new_pts):
                start_points.append(new_pts)
        
        # Run multi-start Sinkhorn
        sh_points = multi_start_sinkhorn(
            start_points, start_time, sinkhorn_deadline,
            max_iter_per_start=400
        )
        sh_score = min_triangle_area_vec(sh_points)
        
        if sh_score > best_score:
            best_score = sh_score
            best_params = _points_to_reduced_params(sh_points)
    
    # Phase 2: Simulated annealing polish
    if time.time() < deadline - 8:
        points2 = get_full_points(best_params)
        sa_deadline = deadline - 8
        
        sa_points = simulate_annealing(points2, start_time, sa_deadline, 
                                       initial_temp=0.02, cooling_rate=0.998, max_iter=3000)
        sa_score = min_triangle_area_vec(sa_points)
        
        if sa_score > best_score:
            best_score = sa_score
            best_params = _points_to_reduced_params(sa_points)
    
    # Phase 3: Nelder-Mead polish on reduced space
    if time.time() < deadline - 2:
        rng = np.random.default_rng(42)
        for attempt in range(3):
            if time.time() > deadline - 2:
                break
            
            if attempt == 0:
                x0 = best_params.copy()
            else:
                perturbation = rng.normal(0, 0.003, size=best_params.shape)
                x0 = best_params + perturbation
                if not is_inside_reduced(x0):
                    continue
            
            try:
                res = minimize(
                    objective_reduced,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
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

import numpy as np
from scipy.spatial.distance import pdist
import time
import scipy.optimize as so

def _compute_score(points):
    """Compute R = (d_min / d_max)^2 for a set of points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    if d_min == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _sqp_maximize_min_dist(points, max_dist_sq, max_iter=100, tol=1e-8):
    """
    Maximize the minimum distance between points, subject to the constraint 
    that all pairwise distances are <= max_dist_sq.
    Uses Sequential Quadratic Programming (SLSQP).
    """
    n = len(points)
    x0 = points.flatten()
    
    def objective(v):
        # v[0:2n] are points, v[2n] is r
        return -v[2*n]
        
    def jac(v):
        grad = np.zeros(2*n + 1)
        grad[2*n] = -1.0
        return grad

    constraints = []
    
    def max_dist_constraint(v):
        pts = v[:2*n].reshape(n, 2)
        d = pdist(pts)
        return max_dist_sq - d**2

    cons_max_dist = {'type': 'ineq', 'fun': max_dist_constraint}
    constraints.append(cons_max_dist)
    
    def min_dist_constraint(v):
        pts = v[:2*n].reshape(n, 2)
        r = v[2*n]
        d = pdist(pts)
        return d**2 - r**2

    cons_min_dist = {'type': 'ineq', 'fun': min_dist_constraint}
    constraints.append(cons_min_dist)
    
    bounds = [(-10, 10)] * (2*n) + [(1e-6, 10)]
    v0 = np.append(x0, 1e-4)
    
    try:
        res = so.minimize(objective, v0, method='SLSQP', jac=jac, bounds=bounds, 
                          constraints=constraints, options={'maxiter': max_iter, 'ftol': tol})
        if res.success:
            return res.x
        else:
            return x0
    except Exception:
        return x0

def _sqp_smooth(points, max_dist, max_iter=150, tol=1e-10):
    """
    Smooth SQP using log-sum-exp for max distance and log-sum-exp for min distance.
    Maximizes r subject to soft constraints.
    """
    n = len(points)
    m = n * (n - 1) // 2
    
    def compute_dists(v):
        pts = v[:2*n].reshape(n, 2)
        return pdist(pts)
    
    def logsumexp_max(x, beta=50.0):
        """Approximate max(x) via log-sum-exp for smoothness."""
        x_max = np.max(x)
        return x_max + np.log(np.mean(np.exp(beta * (x - x_max)))) / beta
    
    def logsumexp_min(x, beta=50.0):
        """Approximate min(x) via -log-sum-exp(-x)."""
        x_min = np.min(x)
        return x_min - np.log(np.mean(np.exp(-beta * (x - x_min)))) / beta
    
    def objective(v):
        # v[0:2n] are points, v[2n] is r
        d = compute_dists(v)
        r = v[2*n]
        # Penalty if max dist exceeds max_dist
        d_max_soft = logsumexp_max(d, beta=50.0)
        penalty = 0.0
        if d_max_soft > max_dist:
            penalty = 100.0 * (d_max_soft - max_dist)**2
        return -r + penalty
    
    def jac(v):
        grad = np.zeros(2*n + 1)
        grad[2*n] = -1.0
        return grad
    
    def max_dist_smooth(v):
        pts = v[:2*n].reshape(n, 2)
        d = pdist(pts)
        d_max_soft = logsumexp_max(d, beta=50.0)
        return max_dist - d_max_soft
    
    cons_max = {'type': 'ineq', 'fun': max_dist_smooth}
    
    def min_dist_smooth(v):
        pts = v[:2*n].reshape(n, 2)
        d = pdist(pts)
        r = v[2*n]
        d_min_soft = logsumexp_min(d, beta=50.0)
        return d_min_soft - r
    
    cons_min = {'type': 'ineq', 'fun': min_dist_smooth}
    
    constraints = [cons_max, cons_min]
    
    bounds = [(-10, 10)] * (2*n) + [(1e-6, 10)]
    v0 = np.append(points.flatten(), 0.1)
    
    try:
        res = so.minimize(objective, v0, method='SLSQP', jac=jac, bounds=bounds,
                          constraints=constraints, options={'maxiter': max_iter, 'ftol': tol})
        if res.success:
            return res.x
        else:
            return None
    except Exception:
        return None

def _bisection_solve(initial_points, time_limit=500.0):
    """
    Use bisection on the ratio R = r_max / d_max.
    """
    n = len(initial_points)
    best_points = initial_points.copy()
    best_ratio = 0.0
    
    start_time = time.time()
    
    for _ in range(10):
        if time.time() - start_time > time_limit:
            break
        
        d = pdist(best_points)
        d_max = d.max()
        if d_max > 1e-9:
            normalized_points = best_points / d_max
        else:
            normalized_points = best_points
        
        result_x = _sqp_maximize_min_dist(normalized_points, 1.0, max_iter=50, tol=1e-6)
        
        if result_x is not None and len(result_x) == 2*n + 1:
            new_points = result_x[:2*n].reshape(n, 2)
            
            d_new = pdist(new_points)
            d_min_new = d_new.min()
            d_max_new = d_new.max()
            
            if d_max_new <= 1.0 + 1e-5 and d_min_new > 1e-9:
                ratio_new = d_min_new / d_max_new
                if ratio_new > best_ratio:
                    best_ratio = ratio_new
                    best_points = new_points.copy()
            else:
                pass

    return best_points, best_ratio

def _generate_initial_points(seed=0):
    """Generate multiple structured initial configurations."""
    n = 16
    rng = np.random.RandomState(seed)
    candidates = []

    # 1. Regular polygon + center (8-gon + inner 8-gon scaled)
    angles_outer = 2 * np.pi * np.arange(8) / 8
    outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    angles_inner = 2 * np.pi * np.arange(8) / 8 + np.pi / 8
    inner = 0.5 * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    candidates.append(np.vstack([outer, inner]))

    # 2. 4x4 grid
    xs = np.linspace(-1, 1, 4)
    ys = np.linspace(-1, 1, 4)
    grid = np.column_stack(np.meshgrid(xs, ys, indexing='ij')).reshape(-1, 2)
    candidates.append(grid)

    # 3. Two concentric 8-gons
    angles1 = 2 * np.pi * np.arange(8) / 8
    angles2 = 2 * np.pi * np.arange(8) / 8 + np.pi / 8
    r1 = 1.0
    r2 = 0.6
    p1 = r1 * np.column_stack([np.cos(angles1), np.sin(angles1)])
    p2 = r2 * np.column_stack([np.cos(angles2), np.sin(angles2)])
    candidates.append(np.vstack([p1, p2]))

    # 4. Hexagonal-ish arrangement: 1 center + 6 ring + 9 outer
    center = np.array([[0.0, 0.0]])
    ring1_angles = 2 * np.pi * np.arange(6) / 6
    ring1 = 0.8 * np.column_stack([np.cos(ring1_angles), np.sin(ring1_angles)])
    ring2_angles = 2 * np.pi * np.arange(9) / 9
    ring2 = 1.5 * np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles)])
    candidates.append(np.vstack([center, ring1, ring2]))

    # 5. Random
    candidates.append(rng.randn(n, 2) * 0.5)

    # 6. Another structured: 2x8 rectangle with jitter
    xs = np.tile(np.array([-1, 1]), 8)
    ys = np.linspace(-1, 1, 16)
    rect = np.column_stack([xs, ys])
    candidates.append(rect + rng.randn(16, 2) * 0.05)
    
    # 7. Random with larger spread
    candidates.append(rng.randn(n, 2) * 2.0)

    return candidates

def _simulated_annealing(points, time_budget=100.0, seed=42):
    """
    Simulated annealing to maximize R = (d_min/d_max)^2.
    """
    n = len(points)
    rng = np.random.RandomState(seed)
    
    # Normalize so max distance is ~1
    d = pdist(points)
    d_max = d.max()
    if d_max > 1e-9:
        current_points = points / d_max
    else:
        current_points = points.copy()
    
    current_score = _compute_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    start_time = time.time()
    T = 0.05  # Initial temperature
    alpha = 0.999  # Cooling rate
    step_size = 0.05
    
    while time.time() - start_time < time_budget:
        if T < 1e-8:
            break
        
        # Generate a perturbation
        idx = rng.randint(0, n)
        delta = rng.randn(2) * step_size
        new_points = current_points.copy()
        new_points[idx] += delta
        
        new_score = _compute_score(new_points)
        delta_score = new_score - current_score
        
        # Accept or reject
        if delta_score > 0 or rng.random() < np.exp(delta_score / T):
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        
        T *= alpha
    
    return best_points, best_score

def _polish(points, time_budget=30.0):
    """High-precision polish pass with tighter tolerances and more iterations."""
    n = len(points)
    start_time = time.time()
    best_points = points.copy()
    best_score = _compute_score(best_points)
    
    # Multiple rounds of SLSQP with decreasing tolerance
    for tol in [1e-7, 1e-9, 1e-11]:
        if time.time() - start_time > time_budget:
            break
        
        d = pdist(best_points)
        d_max = d.max()
        if d_max > 1e-9:
            normalized = best_points / d_max
        else:
            normalized = best_points.copy()
        
        # Standard SQP
        result_x = _sqp_maximize_min_dist(normalized, 1.0, max_iter=200, tol=tol)
        if result_x is not None and len(result_x) == 2*n + 1:
            new_points = result_x[:2*n].reshape(n, 2)
            new_score = _compute_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
        
        # Smooth SQP
        remaining = time_budget - (time.time() - start_time)
        if remaining > 2.0:
            result_smooth = _sqp_smooth(normalized, 1.0, max_iter=100, tol=tol)
            if result_smooth is not None and len(result_smooth) == 2*n + 1:
                new_points = result_smooth[:2*n].reshape(n, 2)
                new_score = _compute_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
    
    return best_points, best_score

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    n = 16
    deadline = time.time() + 610.0  # Safety margin before 630s kill
    
    # Generate initial candidates
    initial_points_list = _generate_initial_points(seed=42)
    
    best_points = initial_points_list[0].copy()
    best_score = _compute_score(best_points)
    
    # Phase 1: SLSQP optimization from each initial configuration
    for i, init_pts in enumerate(initial_points_list):
        if time.time() > deadline - 60.0:
            break
        
        time_budget = min(30.0, deadline - time.time() - 30.0)
        if time_budget < 5.0:
            break
        
        try:
            result_pts, result_ratio = _bisection_solve(init_pts, time_limit=time_budget)
            result_score = result_ratio ** 2
            if result_score > best_score:
                best_points = result_pts
                best_score = result_score
        except Exception as e:
            pass

    # Phase 2: Simulated annealing from the best point found
    remaining = deadline - time.time() - 10.0
    if remaining > 20.0:
        try:
            sa_points, sa_score = _simulated_annealing(best_points, time_budget=remaining - 10.0, seed=123)
            if sa_score > best_score:
                best_points = sa_points
                best_score = sa_score
        except Exception:
            pass

    # Phase 3: Final high-precision polish
    remaining = deadline - time.time() - 5.0
    if remaining > 10.0:
        try:
            polished_points, polished_score = _polish(best_points, time_budget=min(remaining - 5.0, 50.0))
            if polished_score > best_score:
                best_points = polished_points
                best_score = polished_score
        except Exception:
            pass

    return best_points

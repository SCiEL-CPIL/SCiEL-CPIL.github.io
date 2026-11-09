import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _radius_score(centers):
    """Compute the radius score for a given set of centers."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Check bounds
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
        
    # Boundary distances
    d_left = centers[:, 0] + 0.5
    d_right = 0.5 - centers[:, 0]
    d_bottom = centers[:, 1] + 0.5
    d_top = 0.5 - centers[:, 1]
    
    min_boundary = np.min(np.min(np.stack([d_left, d_right, d_bottom, d_top]), axis=0))
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal (self-distance)
    np.fill_diagonal(dist, np.inf)
    
    min_pairwise = np.min(dist)
    min_pairwise_radius = min_pairwise / 2.0
    
    return min(min_boundary, min_pairwise_radius)

def _solve_epigraph(x0, t0):
    """
    Solve the epigraph problem:
    Maximize t
    Subject to:
    0.5 - |x_i - x_j| >= t  for all i < j
    0.5 - |x_i - wall| >= t for all i, walls
    
    This is equivalent to minimizing -t.
    Using SLSQP with explicit constraints.
    """
    n = N
    # Flatten centers to 1D array of length 2N
    x = x0.flatten()
    t = t0
    
    def objective(z):
        # z = [x0, y0, ..., xN-1, yN-1, t]
        return -z[-1]
    
    def constraint_pairwise(z):
        centers = z[:-1].reshape(n, 2)
        t_val = z[-1]
        cons = []
        for i in range(n):
            for j in range(i+1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                cons.append(0.5 - dist - t_val)
        return cons

    def constraint_boundary(z):
        centers = z[:-1].reshape(n, 2)
        t_val = z[-1]
        cons = []
        for i in range(n):
            # Left: x_i + 0.5 >= t  => x_i - t + 0.5 >= 0
            cons.append(centers[i, 0] - t_val + 0.5)
            # Right: 0.5 - x_i >= t => -x_i - t + 0.5 >= 0
            cons.append(-centers[i, 0] - t_val + 0.5)
            # Bottom: y_i + 0.5 >= t => y_i - t + 0.5 >= 0
            cons.append(centers[i, 1] - t_val + 0.5)
            # Top: 0.5 - y_i >= t => -y_i - t + 0.5 >= 0
            cons.append(-centers[i, 1] - t_val + 0.5)
        return cons

    constraints = []
    for c in constraint_pairwise(z):
        constraints.append({'type': 'ineq', 'fun': lambda z, c=c: c}) 
        # Wait, the lambda capture in a loop is tricky for scipy if not careful.
        # Better to define the constraint functions to return arrays.
    
    # Redefine constraints to return arrays for efficiency
    def con_func(z):
        return constraint_pairwise(z)
    
    def con_bound_func(z):
        return constraint_boundary(z)
        
    constraints = [
        {'type': 'ineq', 'fun': con_func},
        {'type': 'ineq', 'fun': con_bound_func}
    ]
    
    bounds = [(-0.5, 0.5)] * (2*n) + [(0, 1.0)]
    
    try:
        res = minimize(objective, z, method='SLSQP', bounds=bounds, 
                       constraints=constraints, options={'maxiter': 300, 'ftol': 1e-12})
        if res.success:
            return res.x
        else:
            return z
    except Exception:
        return z

def _construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    best_centers = None
    best_radius = -np.inf
    
    rng = np.random.default_rng(42)
    
    # 1. Generate diverse initializations
    inits = []
    
    # Grid 7x8
    pts = []
    for i in range(8):
        for j in range(7):
            if len(pts) >= N:
                break
            x = (j + 0.5) / 7.0 - 0.5
            y = (i + 0.5) / 8.0 - 0.5
            pts.append([x, y])
    inits.append(np.array(pts[:N]))
    
    # Grid 8x7
    pts = []
    for i in range(7):
        for j in range(8):
            if len(pts) >= N:
                break
            x = (j + 0.5) / 8.0 - 0.5
            y = (i + 0.5) / 7.0 - 0.5
            pts.append([x, y])
    inits.append(np.array(pts[:N]))
    
    # Hexagonal-ish
    pts = []
    y = -0.45
    row = 0
    while len(pts) < N and y < 0.45:
        offset = 0.0 if row % 2 == 0 else 0.06
        x = -0.45 + offset
        while x < 0.45 and len(pts) < N:
            pts.append([x, y])
            x += 0.12
        y += 0.10
        row += 1
    if len(pts) >= N:
        inits.append(np.array(pts[:N]))
        
    # Random
    for _ in range(5):
        inits.append(rng.uniform(-0.45, 0.45, size=(N, 2)))
        
    # 2. Run Epigraph Optimization on each init
    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Jitter to break symmetry
        jittered = init + rng.normal(0, 0.005, size=(N, 2))
        jittered = np.clip(jittered, -0.49, 0.49)
        
        # Initial t estimate: 0.5 * min_distance
        diff = jittered[:, np.newaxis, :] - jittered[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_dist = np.min(dist)
        
        # Boundary min
        d_left = jittered[:, 0] + 0.5
        d_right = 0.5 - jittered[:, 0]
        d_bottom = jittered[:, 1] + 0.5
        d_top = 0.5 - jittered[:, 1]
        min_bound = np.min(np.min(np.stack([d_left, d_right, d_bottom, d_top]), axis=0))
        
        t_init = min(min_dist / 2.0, min_bound)
        
        # Solve
        z0 = np.concatenate([jittered.flatten(), [t_init]])
        z_res = _solve_epigraph(z0, t_init)
        
        centers = z_res[:-1].reshape(N, 2)
        t_val = z_res[-1]
        
        # Verify and update best
        # The solver might return a slightly invalid config due to numerical errors
        # Clip to ensure validity
        centers = np.clip(centers, -0.5, 0.5)
        
        r = _radius_score(centers)
        
        if r > best_radius:
            best_radius = r
            best_centers = centers
            
        # 3. Polish the best result found so far with more iterations / tighter tolerance
        # We can run a second pass on the best result
        if i % 2 == 1 and time.time() < deadline - 20.0:
             # Try to improve the current best
             z_best = np.concatenate([best_centers.flatten(), [best_radius]])
             z_polished = _solve_epigraph(z_best, best_radius)
             centers_polished = z_polished[:-1].reshape(N, 2)
             centers_polished = np.clip(centers_polished, -0.5, 0.5)
             r_polished = _radius_score(centers_polished)
             if r_polished > best_radius:
                 best_radius = r_polished
                 best_centers = centers_polished

    # Fallback
    if best_centers is None:
        best_centers = inits[0]
        
    return best_centers

def construct():
    return _construct()

import numpy as np
import time
from scipy.optimize import minimize, NonlinearConstraint
from scipy.spatial.distance import cdist

N = 50
START_TIME = time.time()
TIME_LIMIT = 110.0 # Leave 10s margin for cleanup/return

def check_time():
    return time.time() - START_TIME < TIME_LIMIT

def compute_score(centers):
    """
    Compute the score (radius) for a given set of centers.
    Returns 0 if invalid.
    """
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    
    # Distance from center to origin
    dist_to_origin = np.linalg.norm(centers, axis=1)
    # Max radius allowed by container
    r_container = 1.0 - np.max(dist_to_origin)
    
    if r_container < 0:
        return 0.0

    # Pairwise distances
    dists = cdist(centers, centers)
    # Min distance between any two centers
    # Exclude diagonal (distance 0)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    r_separation = min_dist / 2.0
    
    r = min(r_container, r_separation)
    
    if r < 0:
        return 0.0
        
    return r

def hex_lattice_init(n_points):
    """Generate a hexagonal lattice initialization."""
    points = []
    # Try to generate enough points
    # Spacing roughly 2/r. We don't know r yet.
    # Approx r ~ 0.125. Spacing ~ 0.25.
    # Area of unit disk ~ 3.14. Area per circle ~ 3.14/50 ~ 0.06.
    # Hex spacing s. Area per point ~ s^2 * sqrt(3)/2.
    # s^2 * 0.866 ~ 0.06 => s^2 ~ 0.07 => s ~ 0.26.
    
    s = 0.25
    k = 10
    candidates = []
    for j in range(-k, k+1):
        for i in range(-k, k+1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3) / 2
            if x*x + y*y <= 1.0:
                candidates.append([x, y])
    
    candidates = np.array(candidates)
    if len(candidates) < n_points:
        # Fallback to random
        return generate_random_init(n_points)
        
    # Select n_points closest to origin to keep them inside
    dists = np.linalg.norm(candidates, axis=1)
    idx = np.argsort(dists)[:n_points]
    return candidates[idx]

def generate_random_init(n_points):
    """Generate random points in a disk."""
    angles = np.random.uniform(0, 2*np.pi, n_points)
    radii = np.sqrt(np.random.uniform(0, 0.8**2, n_points))
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))

def ring_init(n_points):
    """Generate a ring-based initialization (1 center, then rings)."""
    points = [[0.0, 0.0]]
    remaining = n_points - 1
    ring_radius = 0.2
    ring_width = 0.15
    
    while remaining > 0:
        # How many points fit in this ring?
        circumference = 2 * np.pi * ring_radius
        # Approximate spacing
        spacing = 0.25 
        count = int(circumference / spacing)
        if count > remaining:
            count = remaining
            
        if count == 0:
             count = 1 # Force at least 1 to avoid infinite loop, will be optimized out
            
        angles = np.linspace(0, 2*np.pi, count, endpoint=False)
        # Add some jitter to break symmetry
        angles += np.random.uniform(-0.1, 0.1, count)
        
        for a in angles:
            points.append([ring_radius * np.cos(a), ring_radius * np.sin(a)])
            
        remaining -= count
        ring_radius += ring_width
        if ring_radius > 0.9:
             # Fill rest randomly
             pts = generate_random_init(remaining)
             points.extend(pts.tolist())
             remaining = 0
             break
             
    return np.array(points[:n_points])

def optimize_packing(init_points):
    """
    Optimize the packing starting from init_points.
    Uses SLSQP to maximize r.
    """
    if not check_time():
        return init_points, compute_score(init_points)

    x0 = np.concatenate([init_points.flatten(), [0.1]]) # Initial r guess
    
    # Objective: minimize -r
    def objective(x):
        return -x[-1]
        
    # Constraints
    # 1. Pairwise distances
    # 2. Container boundary
    
    # SLSQP constraints are functions returning array of values >= 0
    
    def constraint_pairwise(x):
        centers = x[:2*N].reshape(N, 2)
        r = x[-1]
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        min_dists = np.min(dists)
        # We want all pairwise distances >= 2r
        # This is O(N^2) constraints. SLSQP might struggle with 1225 constraints.
        # Instead, we can use a penalty method or reduce constraints.
        # For SLSQP, we can define the constraint as the minimum distance minus 2r.
        # But SLSQP expects a vector of constraints.
        # Using all pairs is too many.
        # Let's switch to a penalty-based approach with L-BFGS-B or BFGS for speed and robustness.
        pass
        
    # Let's use a penalty function with BFGS/L-BFGS-B instead.
    # Minimize: -r + lambda * sum(max(0, 2r - d_ij)^2) + mu * sum(max(0, |c_i| - (1-r))^2)
    
    def penalty_objective(x, lam=100.0, mu=100.0):
        centers = x[:2*N].reshape(N, 2)
        r = x[-1]
        
        # Separation penalty
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        # We only care about violations where dist < 2r
        violations_sep = np.maximum(0, 2*r - dists)
        pen_sep = np.sum(violations_sep**2)
        
        # Container penalty
        dists_origin = np.linalg.norm(centers, axis=1)
        violations_cont = np.maximum(0, dists_origin - (1 - r))
        pen_cont = np.sum(violations_cont**2)
        
        return -r + lam * pen_sep + mu * pen_cont

    # Gradient is expensive to derive analytically for the penalty, let scipy approximate it.
    # Or use SLSQP with a subset of constraints?
    # Actually, for N=50, SLSQP with all constraints might be slow but accurate.
    # Let's try SLSQP with a "soft" constraint formulation or just use the penalty with BFGS.
    # BFGS is fast.
    
    def objective_bfgs(x):
        return penalty_objective(x)
        
    # Bounds:
    # x, y in [-1, 1]
    # r in [0, 1]
    bounds = [(-1.0, 1.0)] * (2*N) + [(0.0, 1.0)]
    
    try:
        res = minimize(objective_bfgs, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        if check_time():
             # Polish with SLSQP if time permits?
             # SLSQP is slower but handles constraints exactly.
             # Let's try to extract r from res and see if it's valid.
             best_x = res.x
             best_r = best_x[-1]
             
             # Verify validity
             score = compute_score(best_x[:2*N].reshape(N, 2))
             
             return best_x[:2*N].reshape(N, 2), score
    except Exception:
        pass
        
    return init_points, compute_score(init_points)

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_centers = None
    best_score = 0.0
    
    # Initialization strategies
    inits = []
    
    # 1. Hex lattice
    inits.append(hex_lattice_init(N))
    
    # 2. Ring structure
    inits.append(ring_init(N))
    
    # 3. Random
    for _ in range(5):
        inits.append(generate_random_init(N))
        
    # 4. "Known" good structure approximation?
    # We can try to place points in concentric rings with optimal spacing.
    
    for init in inits:
        if not check_time():
            break
            
        centers, score = optimize_packing(init)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
            
            # If we found a very good score, we might want to polish it further
            # But let's first finish all initializations.
            
    # If we have time, run a few more random starts or refine the best
    if check_time() and best_centers is not None:
        # Refine the best found
        # Run optimize_packing on the best result found
        # Note: optimize_packing modifies the input array? No, it creates new x0.
        # We can just call it again on the best_centers
        centers, score = optimize_packing(best_centers)
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
            
    if best_centers is None:
        # Fallback
        best_centers = hex_lattice_init(N)
        
    return best_centers

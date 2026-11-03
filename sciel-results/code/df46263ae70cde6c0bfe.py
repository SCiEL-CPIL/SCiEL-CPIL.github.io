import numpy as np
from scipy.optimize import minimize
import time

N = 50
RING_COUNTS = [1, 6, 12, 18, 13]  # Standard ring structure for 50 circles

def _pairwise_distances(pts):
    """Compute all pairwise distances between N points."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    return dist

def _compute_score(pts):
    """Compute the packing score: min of half-min-pairwise-dist and min-boundary-dist."""
    n = len(pts)
    dist = _pairwise_distances(pts)
    # Get min pairwise distance (excluding diagonal)
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    # Boundary distances
    boundary = 1.0 - np.sqrt(np.sum(pts**2, axis=1))
    min_boundary = np.min(boundary)
    return min(min_pair / 2.0, min_boundary)

def _softmin_logsumexp(values, beta=100.0):
    """Approximate min using log-sum-exp for smoothness."""
    m = np.min(values)
    return m - np.log(np.sum(np.exp(-beta * (values - m)))) / beta

def _smooth_objective(vars, r_target, beta=100.0):
    """
    Smooth objective: minimize -r_target + penalty for violations.
    Uses soft-min to make the objective differentiable.
    vars: [ring_radii, ring_angles_offsets, ...] or flat point coords
    """
    # We'll use a simpler approach: optimize points directly with smooth penalty
    pts = vars[:2*N].reshape(N, 2)
    r = vars[2*N]
    
    dist = _pairwise_distances(pts)
    np.fill_diagonal(dist, np.inf)
    
    # Soft min of pairwise distances / 2
    soft_min_pair = _softmin_logsumexp(dist / 2.0, beta)
    
    # Boundary
    boundary = 1.0 - np.sqrt(np.sum(pts**2, axis=1))
    soft_min_boundary = _softmin_logsumexp(boundary, beta)
    
    # We want r <= min(soft_min_pair, soft_min_boundary)
    # Penalty if r exceeds the soft min
    violation = np.maximum(0, r - soft_min_pair) + np.maximum(0, r - soft_min_boundary)
    
    return -r + 10.0 * violation

def _ring_to_points(ring_radii, ring_angles):
    """Convert ring parameters to point coordinates."""
    points = []
    for i, (rr, ra) in enumerate(zip(ring_radii, ring_angles)):
        count = RING_COUNTS[i]
        if count == 1:
            points.append([0.0, 0.0])
        else:
            base_angle = ra
            angles = base_angle + np.linspace(0, 2*np.pi, count, endpoint=False)
            for a in angles:
                points.append([rr * np.cos(a), rr * np.sin(a)])
    return np.array(points)

def _flatten_ring(ring_radii, ring_angles):
    """Flatten ring parameters into a single vector."""
    return np.concatenate([ring_radii, ring_angles])

def _unflatten_ring(vars):
    """Unflatten ring parameters."""
    n_rings = len(RING_COUNTS)
    ring_radii = vars[:n_rings]
    ring_angles = vars[n_rings:2*n_rings]
    return ring_radii, ring_angles

def _ring_objective(vars, beta=100.0):
    """Objective function for ring-based optimization."""
    ring_radii, ring_angles = _unflatten_ring(vars)
    # Ensure radii are positive and ordered
    ring_radii = np.maximum(0.01, ring_radii)
    pts = _ring_to_points(ring_radii, ring_angles)
    
    dist = _pairwise_distances(pts)
    np.fill_diagonal(dist, np.inf)
    
    soft_min_pair = _softmin_logsumexp(dist / 2.0, beta)
    boundary = 1.0 - np.sqrt(np.sum(pts**2, axis=1))
    soft_min_boundary = _softmin_logsumexp(boundary, beta)
    
    # The achievable radius is the min of these
    # We want to maximize the minimum, so minimize -min
    # Use soft min of the two soft mins
    achievable = _softmin_logsumexp(np.array([soft_min_pair, soft_min_boundary]), beta)
    
    # Also penalize if inner rings have larger radius than outer
    order_penalty = 0.0
    for i in range(len(ring_radii)-1):
        if ring_radii[i] > ring_radii[i+1]:
            order_penalty += (ring_radii[i] - ring_radii[i+1])**2
    
    return -achievable + 0.1 * order_penalty

def _anneal_points(pts, r_target, n_iter=500, start_temp=0.05, end_temp=0.001):
    """Simple simulated annealing to improve a packing."""
    current_pts = pts.copy()
    current_score = _compute_score(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    for i in range(n_iter):
        temp = start_temp * (end_temp / start_temp) ** (i / n_iter)
        
        # Random perturbation
        new_pts = current_pts + np.random.randn(N, 2) * temp
        
        # Keep inside container
        norms = np.sqrt(np.sum(new_pts**2, axis=1))
        scale = np.minimum(1.0, (1.0 - r_target*0.5) / np.maximum(norms, 1e-10))
        new_pts *= scale[:, np.newaxis]
        
        new_score = _compute_score(new_pts)
        
        if new_score >= current_score or np.random.rand() < np.exp((new_score - current_score) / (temp * 0.1)):
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_pts = current_pts.copy()
                best_score = current_score
    
    return best_pts

def _constraint_push(pts, r_target, n_iters=100):
    """Iteratively push overlapping circles apart."""
    pts = pts.copy()
    for _ in range(n_iters):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt(np.sum((pts[i] - pts[j])**2))
                if d < 2*r_target and d > 1e-10:
                    # Push apart
                    overlap = 2*r_target - d
                    direction = (pts[i] - pts[j]) / d
                    pts[i] += direction * overlap * 0.5
                    pts[j] -= direction * overlap * 0.5
                    changed = True
        # Push inside boundary
        for i in range(N):
            norm = np.sqrt(np.sum(pts[i]**2))
            if norm > 1.0 - r_target:
                scale = (1.0 - r_target) / norm
                pts[i] *= scale
                changed = True
        if not changed:
            break
    return pts

def _hexagonal_init():
    """Generate hexagonal lattice initialization."""
    spacing = 0.25
    points = []
    for i in range(-10, 11):
        for j in range(-10, 11):
            x = i * spacing + (j % 2) * spacing * 0.5
            y = j * spacing * np.sqrt(3) / 2
            if x*x + y*y < 0.9**2:
                points.append([x, y])
    if len(points) > N:
        d = np.hypot(np.array(points)[:,0], np.array(points)[:,1])
        idx = np.argsort(d)[:N]
        return np.array(points)[idx]
    else:
        while len(points) < N:
            points.append([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)])
        return np.array(points)

def _ring_initialization():
    """Generate ring-based initialization."""
    ring_radii = np.linspace(0.1, 0.85, len(RING_COUNTS))
    ring_angles = np.zeros(len(RING_COUNTS))
    return _ring_to_points(ring_radii, ring_angles)

def _binary_search_polish(pts, r_low, r_high, max_iter=20, beta=200.0):
    """Binary search for optimal radius with local optimization."""
    for _ in range(max_iter):
        r_mid = (r_low + r_high) / 2
        # Try to optimize at this radius
        x0 = np.concatenate([pts.flatten(), [r_mid]])
        bounds = [(-1.0, 1.0)] * (2*N) + [(r_low*0.9, r_high*1.05)]
        
        try:
            res = minimize(_smooth_objective, x0, args=(r_mid, beta), 
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            r_found = res.x[2*N]
            pts_found = res.x[:2*N].reshape(N, 2)
            
            # Check validity
            score = _compute_score(pts_found)
            if score >= r_mid - 1e-9:
                r_low = r_mid
                pts = pts_found
            else:
                r_high = r_mid
        except:
            r_high = r_mid
    
    return pts, r_low

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_r = 0.0
    best_points = None
    
    def update_best(pts):
        nonlocal best_r, best_points
        r = _compute_score(pts)
        if r > best_r:
            best_r = r
            best_points = pts.copy()
    
    # Generate multiple initializations
    inits = []
    
    # 1. Ring initialization
    inits.append(_ring_initialization())
    
    # 2. Hexagonal
    inits.append(_hexagonal_init())
    
    # 3. Spiral
    t = np.linspace(0, 1, N)
    r = 0.1 + 0.8 * t
    theta = 2 * np.pi * 10 * t
    inits.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
    
    # 4. Random
    inits.append(np.random.uniform(-0.8, 0.8, (N, 2)))
    
    # 5. Ring with different radii
    ring_radii = np.array([0.05, 0.25, 0.45, 0.65, 0.82])
    ring_angles = np.array([0, np.pi/6, 0, np.pi/12, 0])
    inits.append(_ring_to_points(ring_radii, ring_angles))
    
    for init_pts in inits:
        if time.time() > deadline:
            break
        
        # Constraint push to get a valid starting point
        pushed = _constraint_push(init_pts, 0.1, n_iters=50)
        r0 = _compute_score(pushed)
        
        # Stage 1: Ring-based optimization if we started from rings
        if len(RING_COUNTS) == 5:
            # Try ring optimization
            n_rings = len(RING_COUNTS)
            ring_radii_init = np.linspace(0.1, 0.85, n_rings)
            ring_angles_init = np.zeros(n_rings)
            x0_ring = _flatten_ring(ring_radii_init, ring_angles_init)
            
            try:
                res_ring = minimize(_ring_objective, x0_ring, method='L-BFGS-B',
                                   bounds=[(0.01, 0.95)] * n_rings + [(-np.pi, np.pi)] * n_rings,
                                   options={'maxiter': 300, 'ftol': 1e-10})
                rr, ra = _unflatten_ring(res_ring.x)
                ring_pts = _ring_to_points(rr, ra)
                ring_pts = _constraint_push(ring_pts, 0.1, n_iters=30)
                update_best(ring_pts)
            except:
                pass
        
        # Stage 2: Annealing
        if time.time() > deadline:
            break
        annealed = _anneal_points(pushed, r0, n_iter=300)
        update_best(annealed)
        
        # Stage 3: L-BFGS polish with smooth objective
        if time.time() > deadline:
            break
        x0 = np.concatenate([annealed.flatten(), [r0]])
        bounds = [(-1.0, 1.0)] * (2*N) + [(0.0, 0.5)]
        try:
            res = minimize(_smooth_objective, x0, args=(r0, 100.0),
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            r_found = res.x[2*N]
            pts_found = res.x[:2*N].reshape(N, 2)
            update_best(pts_found)
        except:
            pass
        
        # Stage 4: Binary search polish
        if time.time() > deadline:
            break
        if best_points is not None:
            polished, r_final = _binary_search_polish(best_points, max(0.05, r0*0.9), min(0.5, r0*1.1), max_iter=15, beta=200.0)
            update_best(polished)
    
    if best_points is None:
        # Fallback
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                return points[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98
    
    return best_points

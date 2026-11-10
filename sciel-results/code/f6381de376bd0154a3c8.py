import numpy as np
from scipy.optimize import minimize

N = 50

def compute_score(points):
    """
    Compute the score: min(min_pair_dist/2, min_wall_dist)
    """
    norms = np.linalg.norm(points, axis=1)
    wall_dists = 1.0 - norms
    min_wall = np.min(wall_dists)
    
    min_pair = 1e9
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_pair:
                min_pair = d
                
    return min(min_pair / 2.0, min_wall)

def objective(params, beta=50.0):
    """
    Smooth objective function to maximize the minimum distance.
    Minimizes -soft_min(distances).
    """
    points = params.reshape(-1, 2)
    
    # Calculate distances to walls
    norms = np.linalg.norm(points, axis=1)
    wall_dists = 1.0 - norms
    
    # Calculate pair distances
    # Vectorized calculation for speed
    # N=50, 1225 pairs
    dists = []
    for i in range(N):
        diffs = points[i+1:] - points[i]
        d = np.linalg.norm(diffs, axis=1)
        dists.append(d)
    pair_dists = np.concatenate(dists) if dists else np.array([])
    
    # Combine all distances
    all_dists = np.concatenate([pair_dists, wall_dists])
    
    # Handle potential numerical issues
    # Clip distances to avoid log(0) or overflow
    all_dists = np.clip(all_dists, 1e-12, 10.0)
    
    # Soft min approximation: -1/beta * log(sum(exp(-beta * dists)))
    # This is a convex function of dists, but we are maximizing it (minimizing its negative).
    # Wait, soft min is a lower bound on the min? No, -1/beta log sum exp(-beta x) approximates min(x) from below?
    # Let's check. If x1=1, x2=2. beta=10.
    # exp(-10) + exp(-20) ~ 4.5e-5 + 2e-9 ~ 4.5e-5.
    # log(4.5e-5) ~ -10.
    # -1/10 * (-10) = 1.
    # Min is 1. Correct.
    # If x1=1, x2=1.
    # 2 * exp(-10) ~ 9e-5.
    # log(9e-5) ~ -9.3.
    # -1/10 * (-9.3) = 0.93.
    # Min is 1. It's slightly lower.
    # So it approximates the min from below. Maximizing it pushes all distances up.
    
    # To avoid overflow in exp, we use the log-sum-exp trick
    # log(sum(exp(-beta * d))) = -beta * max(d) + log(sum(exp(-beta * (d - max(d)))))
    # But we want to MINIMIZE the negative of this.
    # Actually, let's just use a large beta and be careful.
    # With N=50, max dist ~ 2, min dist ~ 0.1.
    # beta=50. exp(-50*0.1) = exp(-5) ~ 0.006.
    # exp(-50*2) = exp(-100) ~ 0.
    # This is safe.
    
    exp_terms = np.exp(-beta * all_dists)
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0:
        return 0.0
        
    soft_min = -1.0 / beta * np.log(sum_exp)
    
    return -soft_min

def construct():
    best_points = None
    best_score = -1e9
    
    # Initial guesses
    inits = []
    
    # 1. Hexagonal lattice
    # Generate a grid and pick the N closest to origin
    # Spacing needs to be small enough to have >50 points
    spacing = 0.15
    k = 10
    hex_points = []
    for i in range(-k, k+1):
        for j in range(-k, k+1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) < 1.0:
                hex_points.append([x, y])
    hex_points = np.array(hex_points)
    dists = np.linalg.norm(hex_points, axis=1)
    hex_points = hex_points[np.argsort(dists)[:N]]
    inits.append(hex_points)
    
    # 2. Square lattice
    sq_points = []
    spacing_sq = 0.15
    for i in range(-7, 8):
        for j in range(-7, 8):
            x = i * spacing_sq
            y = j * spacing_sq
            if np.hypot(x, y) < 1.0:
                sq_points.append([x, y])
    sq_points = np.array(sq_points)
    dists = np.linalg.norm(sq_points, axis=1)
    sq_points = sq_points[np.argsort(dists)[:N]]
    inits.append(sq_points)
    
    # 3. Random initialization (few)
    for seed in [1, 2, 3, 4, 5]:
        rng = np.random.default_rng(seed)
        while True:
            pts = rng.uniform(-1, 1, size=(N, 2))
            norms = np.linalg.norm(pts, axis=1)
            if np.all(norms <= 1.0):
                break
        inits.append(pts)
        
    # 4. Ring-based
    radii = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
    counts = [1, 6, 12, 12, 9, 10]
    points = []
    angle_offset = 0
    for r, count in zip(radii, counts):
        if count == 0: continue
        angles = np.linspace(0, 2*np.pi, count, endpoint=False) + angle_offset
        for a in angles:
            points.append([r * np.cos(a), r * np.sin(a)])
        angle_offset += np.pi / count
    points = np.array(points[:N])
    inits.append(points)

    # Optimization settings
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    for init_pts in inits:
        params = init_pts.flatten()
        
        try:
            # Use L-BFGS-B which is faster and handles bounds well
            # Beta controls the sharpness of the soft-min. 
            # Start with moderate beta, maybe increase in subsequent runs?
            # For simplicity, we use a fixed beta=50.
            res = minimize(objective, params, method='L-BFGS-B', 
                           bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            
            opt_pts = res.x.reshape(-1, 2)
            
            # Check validity
            if np.all(np.isfinite(opt_pts)):
                # Ensure points are inside (with small tolerance)
                if np.max(np.linalg.norm(opt_pts, axis=1)) < 1.0 + 1e-9:
                    score = compute_score(opt_pts)
                    if score > best_score:
                        best_score = score
                        best_points = opt_pts
        except Exception:
            continue
            
    # Fallback
    if best_points is None:
        best_points = inits[0]
        best_score = compute_score(best_points)
        
    return best_points

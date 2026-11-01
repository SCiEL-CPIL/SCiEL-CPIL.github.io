import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = time.time() + 145.0

def _score(pts):
    """
    Compute the valid packing radius (or 0 if invalid).
    """
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    dists_origin = np.linalg.norm(pts, axis=1)
    if np.any(dists_origin > 1.0):
        return 0.0
    r_cont = 1.0 - np.max(dists_origin)
    if r_cont < 0:
        return 0.0
    
    if N < 2:
        return r_cont
        
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dist_mat = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)
    
    return min(r_cont, r_pair)

def _hex_init(spacing):
    """
    Generate a hexagonal lattice of points.
    """
    points = []
    k = int(2.0 / spacing) + 2
    for j in range(-k, k + 1):
        y = j * spacing * np.sqrt(3.0) / 2.0
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            r = np.hypot(x, y)
            if r + spacing / 2.0 <= 1.0:
                points.append((x, y))
    
    pts = np.array(points)
    if len(pts) < N:
        return None
    
    radii = np.linalg.norm(pts, axis=1)
    idx = np.argsort(radii, kind='stable')[:N]
    return pts[idx]

def _random_init():
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 1.0, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _objective_flat(x_flat):
    """
    Objective function for scipy.optimize.minimize.
    We want to maximize the minimum distance, so we minimize the negative.
    The radius r is min(r_cont, r_pair).
    To maximize r, we minimize -r.
    """
    pts = x_flat.reshape(N, 2)
    
    # Check validity quickly to return large penalty if invalid
    dists_origin = np.linalg.norm(pts, axis=1)
    if np.any(dists_origin > 1.0):
        # Penalty for being outside
        return np.sum(dists_origin[dists_origin > 1.0] - 1.0) * 100.0
    
    r_cont = 1.0 - np.max(dists_origin)
    
    # Pairwise distances
    # Use broadcasting for speed
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dist_mat = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)
    
    r = min(r_cont, r_pair)
    
    return -r

def _optimize_nm(pts_init, max_time=10.0):
    """
    Optimize using Nelder-Mead.
    """
    x0 = pts_init.flatten()
    
    start_time = time.time()
    
    # Nelder-Mead options
    options = {
        'maxiter': 10000,
        'xatol': 1e-8,
        'fatol': 1e-10,
        'adaptive': True
    }
    
    try:
        res = minimize(_objective_flat, x0, method='Nelder-Mead', options=options,
                       callback=lambda x: None) # No callback to save time, or check time
        # Actually, scipy's minimize doesn't support time-based stopping natively in callback easily 
        # without raising exceptions. We will rely on maxiter and the fact that NM converges.
        # However, NM can be slow. Let's add a manual check if possible or just trust maxiter.
        # To be safe with the 120s limit, we should wrap this.
        
        # Since we can't easily interrupt minimize, we run it and check time after.
        # If it takes too long, we return the best found so far? No, minimize returns the best.
        # We just need to ensure we don't call it too many times.
        
        best_pts = res.x.reshape(N, 2)
        best_score = _score(best_pts)
        
        return best_pts, best_score
        
    except Exception:
        return pts_init, _score(pts_init)

def _optimize_sa(pts_init, max_time=10.0):
    """
    Optimise using Simulated Annealing as a fallback or secondary method.
    """
    pts = pts_init.copy().astype(np.float64)
    best_pts = pts.copy()
    best_score = _score(pts)
    
    T = 0.05
    cooling = 0.995
    step_scale = 0.02
    
    start_time = time.time()
    
    while time.time() < min(start_time + max_time, DEADLINE):
        # Generate a random perturbation
        noise = np.random.normal(0, step_scale, (N, 2))
        new_pts = pts + noise
        
        # Clamp to boundary
        radii = np.linalg.norm(new_pts, axis=1)
        mask_out = radii > 1.0
        if np.any(mask_out):
            new_pts[mask_out] = new_pts[mask_out] / radii[mask_out, np.newaxis]
            
        new_score = _score(new_pts)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts.copy()
            T *= cooling
        else:
            # Accept with probability
            delta = new_score - _score(pts)
            if delta < 0:
                prob = np.exp(delta / (T + 1e-9))
                if np.random.rand() < prob:
                    pts = new_pts.copy()
            else:
                pts = new_pts.copy()
            T *= cooling
            
        if T < 1e-6:
            break
            
    return best_pts, best_score

def construct():
    start_time = time.time()
    best_pts = None
    best_score = 0.0
    
    # 1. Try hexagonal lattice initialization with various spacings
    # This provides a good structured starting point
    base_spacing = 0.2516
    for spacing_factor in [0.95, 1.0, 1.05]:
        spacing = base_spacing * spacing_factor
        pts_init = _hex_init(spacing)
        if pts_init is not None:
            score_init = _score(pts_init)
            if score_init > best_score:
                best_score = score_init
                best_pts = pts_init.copy()
                
            # Optimize with Nelder-Mead
            if time.time() < DEADLINE - 20.0:
                opt_pts, opt_score = _optimize_nm(pts_init, max_time=15.0)
                if opt_score > best_score:
                    best_score = opt_score
                    best_pts = opt_pts.copy()
                    
    # 2. Try random initializations with Nelder-Mead
    # Nelder-Mead can escape some local minima if started from different places
    num_random_trials = 3
    for i in range(num_random_trials):
        if time.time() > DEADLINE - 20.0:
            break
            
        pts_init = _random_init()
        # Ensure inside unit circle
        radii = np.linalg.norm(pts_init, axis=1)
        mask = radii > 1.0
        if np.any(mask):
            pts_init[mask] = pts_init[mask] / radii[mask, np.newaxis]
            
        score_init = _score(pts_init)
        if score_init > best_score:
            best_score = score_init
            best_pts = pts_init.copy()
            
        # Optimize
        if time.time() < DEADLINE - 10.0:
            opt_pts, opt_score = _optimize_nm(pts_init, max_time=10.0)
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts.copy()
                
    # 3. Final polish with Simulated Annealing on the best result found
    if best_pts is not None and time.time() < DEADLINE - 5.0:
        opt_pts, opt_score = _optimize_sa(best_pts, max_time=5.0)
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts.copy()

    # Fallback
    if best_pts is None or _score(best_pts) <= 0:
        # Simple grid fallback
        pts_fallback = []
        for i in range(10):
            for j in range(5):
                x = (i - 4.5) * 0.2
                y = (j - 2.0) * 0.2
                if np.hypot(x, y) < 0.9:
                    pts_fallback.append([x, y])
        if len(pts_fallback) < N:
             while len(pts_fallback) < N:
                 x, y = np.random.uniform(-0.8, 0.8, 2)
                 if np.hypot(x,y) < 0.8:
                     pts_fallback.append([x,y])
        best_pts = np.array(pts_fallback[:N])
        
    return best_pts

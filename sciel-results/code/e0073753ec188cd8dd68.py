import numpy as np
from scipy.optimize import minimize
import math

def construct():
    n = 8
    
    def compute_score(pts):
        """Compute the score for a set of points, scaling to diameter 2 first."""
        diff = pts[:, np.newaxis] - pts[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        max_d = np.max(dists)
        
        if max_d < 1e-10:
            return 0.0
        
        pts_scaled = pts * (2.0 / max_d)
        
        diff = pts_scaled[:, np.newaxis] - pts_scaled[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        mask = ~np.eye(n, dtype=bool)
        
        # Avoid log(0)
        dists = np.maximum(dists, 1e-10)
        
        log_delta = np.sum(np.log(dists[mask]))
        log_score = log_delta - n * math.log(n)
        score = math.exp(log_score)
        
        return score

    def objective(params):
        """Minimize negative log of product of distances, with diameter constraint."""
        pts = params.reshape(n, 2)
        diff = pts[:, np.newaxis] - pts[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        max_dist = np.max(dists)
        
        # If diameter is 0, return large penalty
        if max_dist < 1e-10:
            return 1e10
        
        # Scale to unit diameter for evaluation
        pts_unit = pts / max_dist
        
        diff = pts_unit[:, np.newaxis] - pts_unit[np.newaxis, :]
        dists_unit = np.linalg.norm(diff, axis=2)
        mask = ~np.eye(n, dtype=bool)
        
        # Avoid log(0)
        dists_unit = np.maximum(dists_unit, 1e-10)
        
        log_prod = np.sum(np.log(dists_unit[mask]))
        
        return -log_prod

    # Generate multiple initial guesses
    starts = []
    
    # 1. Regular Octagon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    starts.append(np.array([np.cos(angles), np.sin(angles)]).T.flatten())
    
    # 2. Two concentric squares with different radius ratios
    for r_inner in [0.3, 0.5, 0.6, 0.7, 0.8]:
        outer = np.array([[1,0], [-1,0], [0,1], [0,-1]])
        inner = np.array([[r_inner,r_inner], [-r_inner,r_inner], [-r_inner,-r_inner], [r_inner,-r_inner]])
        pts_sq = np.vstack([outer, inner])
        starts.append(pts_sq.flatten())
    
    # 3. Two concentric squares rotated 45 degrees relative to each other
    for r_outer, r_inner in [(1.0, 0.5), (1.0, 0.7), (0.9, 0.6), (1.0, 0.4)]:
        # Outer square at 45 degrees
        outer_angles = np.array([np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4])
        outer = np.array([[r_outer*np.cos(a), r_outer*np.sin(a)] for a in outer_angles])
        
        # Inner square at 0 degrees
        inner_angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        inner = np.array([[r_inner*np.cos(a), r_inner*np.sin(a)] for a in inner_angles])
        
        pts_sq = np.vstack([outer, inner])
        starts.append(pts_sq.flatten())
    
    # 4. Regular octagon with various noise levels
    np.random.seed(42)
    for noise_level in [0.05, 0.1, 0.2]:
        noise = np.random.normal(0, noise_level, (n, 2))
        pts_noise = np.array([np.cos(angles), np.sin(angles)]).T + noise
        starts.append(pts_noise.flatten())
    
    # 5. Antipodal pairs with varying radii
    for r1, r2, r3, r4 in [(1.0, 0.8, 0.6, 0.4), (1.0, 0.9, 0.7, 0.5), (1.0, 0.85, 0.75, 0.65)]:
        radii = [r1, r1, r2, r2, r3, r3, r4, r4]
        angle_offset = 2*np.pi/8
        pts = np.array([[r*np.cos(k*angle_offset), r*np.sin(k*angle_offset)] for k, r in enumerate(radii)])
        starts.append(pts.flatten())
    
    best_score = -np.inf
    best_pts = None
    
    # Try multiple optimization methods and initial guesses
    for x0 in starts:
        # Method 1: Nelder-Mead
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.success or res.fun < 1e10:
                pts = res.x.reshape(n, 2)
                score = compute_score(pts)
                if score > best_score:
                    best_score = score
                    best_pts = pts
        except Exception:
            pass
        
        # Method 2: Powell
        try:
            res = minimize(objective, x0, method='Powell', options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.success or res.fun < 1e10:
                pts = res.x.reshape(n, 2)
                score = compute_score(pts)
                if score > best_score:
                    best_score = score
                    best_pts = pts
        except Exception:
            pass
        
        # Method 3: COBYLA
        try:
            res = minimize(objective, x0, method='COBYLA', options={'maxiter': 5000, 'rhobeg': 1e-3})
            if res.success or res.fun < 1e10:
                pts = res.x.reshape(n, 2)
                score = compute_score(pts)
                if score > best_score:
                    best_score = score
                    best_pts = pts
        except Exception:
            pass
    
    # If optimization failed or returned bad results, fallback to regular octagon
    if best_pts is None or best_score <= 1.0:
        return [[math.cos(2 * math.pi * k / 8), math.sin(2 * math.pi * k / 8)] for k in range(8)]
        
    return best_pts.tolist()

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def compute_exact_score(pts):
        """Compute the exact packing radius for given centers."""
        if len(pts) != N:
            return -1.0
        if not np.all(np.isfinite(pts)):
            return -1.0
        
        # Radial distances
        dists = np.linalg.norm(pts, axis=1)
        if np.any(dists > 1.0):
            return -1.0
        
        # Pairwise distances using vectorized upper triangle
        i_idx, j_idx = np.triu_indices(N, k=1)
        diffs = pts[i_idx] - pts[j_idx]
        pair_dists = np.linalg.norm(diffs, axis=1)
        
        min_pair_dist = np.min(pair_dists)
        min_radial_dist = 1.0 - np.max(dists)
        
        r = min(min_pair_dist / 2.0, min_radial_dist)
        return r

    def generate_hexagonal_initialization():
        """Generate a hexagonal lattice clipped to the unit circle."""
        best_pts = None
        best_score = -1.0
        
        for scale in [0.9, 0.95, 1.0, 1.05, 1.1]:
            spacing = 2.0 / (np.sqrt(N) * scale)
            
            nx = int(2.0 / spacing) + 3
            ny = int(2.0 / (spacing * np.sqrt(3.0) / 2.0)) + 3
            
            x_coords = (np.arange(-nx//2, nx//2 + 1) + 0.5) * spacing
            y_coords = (np.arange(-ny//2, ny//2 + 1)) * spacing * np.sqrt(3.0) / 2.0
            
            pts_list = []
            for i, y in enumerate(y_coords):
                offset = 0.0 if i % 2 == 0 else spacing / 2.0
                for x in x_coords:
                    px = x + offset
                    py = y
                    if np.hypot(px, py) <= 1.0 - spacing / 2.0:
                        pts_list.append([px, py])
            
            if len(pts_list) < N:
                continue
                
            pts = np.array(pts_list)
            norms = np.linalg.norm(pts, axis=1)
            idx = np.argsort(norms)
            pts = pts[idx[:N]]
            
            score = compute_exact_score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts
        
        if best_pts is None:
            best_pts = np.random.rand(N, 2) * 2 - 1
            norms = np.linalg.norm(best_pts, axis=1)
            best_pts = best_pts * (0.9 / norms)[:, None]
            best_score = compute_exact_score(best_pts)
            
        return best_pts, best_score

    def generate_ring_initialization():
        """Generate ring-based packing initialization."""
        best_pts = None
        best_score = -1.0
        
        # Try multiple ring configurations
        ring_configs = [
            # (counts, radii, offsets)
            (np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 29]), 
             np.linspace(0.02, 0.98, 11), np.zeros(11)),
            (np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 29]),
             np.linspace(0.02, 0.98, 12), np.zeros(12)),
            (np.array([1, 8, 16, 24, 32, 40, 48, 56, 75]),
             np.linspace(0.05, 0.98, 9), np.zeros(9)),
            (np.array([1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 69]),
             np.linspace(0.02, 0.98, 12), np.zeros(12)),
        ]
        
        for counts, radii, offsets in ring_configs:
            counts = counts.copy()
            diff = N - np.sum(counts)
            for i in range(abs(diff)):
                if diff > 0:
                    counts[i % len(counts)] += 1
                else:
                    counts[i % len(counts)] -= 1
            counts = np.maximum(1, counts)
            
            points = []
            for n_circ, r_ring, offset in zip(counts, radii, offsets):
                if n_circ == 0:
                    continue
                if n_circ == 1:
                    if r_ring < 1e-9:
                        points.append([0.0, 0.0])
                    else:
                        points.append([r_ring * np.cos(offset), r_ring * np.sin(offset)])
                else:
                    angles = offset + np.linspace(0, 2*np.pi, n_circ, endpoint=False)
                    for a in angles:
                        points.append([r_ring * np.cos(a), r_ring * np.sin(a)])
            
            if len(points) != N:
                continue
                
            pts = np.array(points)
            score = compute_exact_score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts
        
        if best_pts is None:
            return None, -1.0
            
        return best_pts, best_score

    def soft_min(x, y, beta):
        """Smooth approximation of min(x, y) using log-sum-exp trick."""
        if np.isscalar(x) and np.isscalar(y):
            m = min(x, y)
            return -1.0 / beta * (np.log(np.exp(-beta * (x - m)) + np.exp(-beta * (y - m))) - beta * m)
        else:
            return np.minimum(x, y)

    def optimize_with_soft_min(pts_init, beta, max_iter=500):
        """Optimize using soft-min radius objective with L-BFGS-B."""
        if time.time() > deadline - 10:
            return pts_init, compute_exact_score(pts_init)
        
        def objective(params):
            if time.time() > deadline:
                return 1e10
            
            pts = params.reshape(-1, 2)
            
            if not np.all(np.isfinite(pts)):
                return 1e10
            
            dists = np.linalg.norm(pts, axis=1)
            if np.any(dists > 1.0):
                return 1e10 + np.sum(np.maximum(dists - 1.0, 0.0))
            
            i_idx, j_idx = np.triu_indices(N, k=1)
            diffs = pts[i_idx] - pts[j_idx]
            pair_dists = np.linalg.norm(diffs, axis=1)
            
            min_pair_dist = np.min(pair_dists)
            min_radial_dist = 1.0 - np.max(dists)
            
            x = min_pair_dist / 2.0
            y = min_radial_dist
            
            val = soft_min(x, y, beta)
            return -val
        
        x0 = pts_init.flatten()
        bounds = [(-0.999, 0.999)] * (2 * N)
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': max_iter, 'maxfun': max_iter * 5,
                                      'ftol': 1e-12, 'gtol': 1e-10})
            if result.fun < 1e10:
                new_pts = result.x.reshape(-1, 2)
                new_score = compute_exact_score(new_pts)
                return new_pts, new_score
        except Exception:
            pass
        
        return pts_init, compute_exact_score(pts_init)

    def optimize_with_auxiliary_t(pts_init, max_iter=300):
        """Optimize using auxiliary variable t with smooth penalty."""
        if time.time() > deadline - 10:
            return pts_init, compute_exact_score(pts_init)
        
        def objective(params):
            if time.time() > deadline:
                return 1e10
            
            t = params[0]
            pts = params[1:].reshape(-1, 2)
            
            if not np.all(np.isfinite(pts)) or not np.isfinite(t):
                return 1e10
            
            if t < 0:
                t = 0
            
            dists = np.linalg.norm(pts, axis=1)
            radial_violation = np.maximum(0, dists + t - 1.0)
            radial_penalty = np.sum(radial_violation ** 2)
            
            i_idx, j_idx = np.triu_indices(N, k=1)
            diffs = pts[i_idx] - pts[j_idx]
            pair_dists = np.linalg.norm(diffs, axis=1)
            viol = np.maximum(0, 2*t - pair_dists)
            pair_penalty = np.sum(viol ** 2)
            
            lambda_val = 100.0
            obj = -t + lambda_val * (radial_penalty + pair_penalty)
            
            return obj
        
        pts = pts_init.copy()
        dists = np.linalg.norm(pts, axis=1)
        i_idx, j_idx = np.triu_indices(N, k=1)
        diffs = pts[i_idx] - pts[j_idx]
        pair_dists = np.linalg.norm(diffs, axis=1)
        min_pair_dist = np.min(pair_dists)
        
        t_init = min(min_pair_dist / 2.0, 1.0 - np.max(dists)) * 0.95
        t_init = max(0.01, min(t_init, 0.08))
        
        x0 = np.concatenate([[t_init], pts.flatten()])
        bounds = [(0.001, 0.08)] + [(-0.999, 0.999)] * (2 * N)
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': max_iter, 'maxfun': max_iter * 5,
                                      'ftol': 1e-12, 'gtol': 1e-10})
            if result.fun < 1e10:
                t_opt = result.x[0]
                pts_opt = result.x[1:].reshape(-1, 2)
                score = compute_exact_score(pts_opt)
                return pts_opt, score
        except Exception:
            pass
        
        return pts_init, compute_exact_score(pts_init)

    def polish(pts, max_iter=200):
        """Final polish with tighter tolerance."""
        if time.time() > deadline - 5:
            return pts, compute_exact_score(pts)
        
        def objective(params):
            if time.time() > deadline:
                return 1e10
            
            pts = params.reshape(-1, 2)
            
            if not np.all(np.isfinite(pts)):
                return 1e10
            
            dists = np.linalg.norm(pts, axis=1)
            if np.any(dists > 1.0):
                return 1e10
            
            i_idx, j_idx = np.triu_indices(N, k=1)
            diffs = pts[i_idx] - pts[j_idx]
            pair_dists = np.linalg.norm(diffs, axis=1)
            
            min_dist = np.min(pair_dists)
            r = min(min_dist / 2.0, 1.0 - np.max(dists))
            return -r
        
        x0 = pts.flatten()
        bounds = [(-0.999, 0.999)] * (2 * N)
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': max_iter, 'maxfun': max_iter * 3,
                                      'ftol': 1e-14, 'gtol': 1e-12})
            if result.fun < 1e10:
                new_pts = result.x.reshape(-1, 2)
                new_score = compute_exact_score(new_pts)
                if new_score > compute_exact_score(pts):
                    return new_pts, new_score
        except Exception:
            pass
        
        return pts, compute_exact_score(pts)

    # Strategy 1: Generate multiple initializations and pick the best
    hex_pts, hex_score = generate_hexagonal_initialization()
    ring_pts, ring_score = generate_ring_initialization()
    
    best_pts = hex_pts
    best_score = hex_score
    
    if ring_pts is not None and ring_score > best_score:
        best_pts = ring_pts
        best_score = ring_score
    
    # Strategy 2: Multi-stage optimization with increasing beta
    betas = [10.0, 50.0, 100.0, 200.0, 500.0]
    
    current_pts = best_pts
    current_score = best_score
    
    for beta in betas:
        if time.time() > deadline - 30:
            break
        
        opt_pts, opt_score = optimize_with_soft_min(current_pts, beta, max_iter=200)
        if opt_score > current_score:
            current_pts = opt_pts
            current_score = opt_score
    
    if current_score > best_score:
        best_pts = current_pts
        best_score = current_score
    
    # Strategy 3: Try auxiliary t optimization from current best
    if time.time() < deadline - 25:
        opt_pts, opt_score = optimize_with_auxiliary_t(best_pts, max_iter=200)
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts
    
    # Strategy 4: Random restarts if time permits
    if time.time() < deadline - 20:
        np.random.seed(42)
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            
            pts_perturbed = best_pts + np.random.randn(N, 2) * 0.01
            dists = np.linalg.norm(pts_perturbed, axis=1)
            mask = dists > 0.99
            pts_perturbed[mask] = pts_perturbed[mask] / dists[mask, None] * 0.99
            
            score = compute_exact_score(pts_perturbed)
            if score > best_score:
                best_score = score
                best_pts = pts_perturbed
            
            if time.time() < deadline - 8:
                opt_pts, opt_score = optimize_with_soft_min(pts_perturbed, 100.0, max_iter=100)
                if opt_score > best_score:
                    best_score = opt_score
                    best_pts = opt_pts
    
    # Final polish
    if time.time() < deadline - 5:
        final_pts, final_score = polish(best_pts, max_iter=150)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts
    
    # Final validation
    final_score = compute_exact_score(best_pts)
    if final_score <= 0 or not np.all(np.isfinite(best_pts)):
        # Return a valid fallback
        best_pts = np.zeros((N, 2))
        n_per_row = int(np.sqrt(N))
        spacing = 0.03
        for i in range(N):
            row = i // n_per_row
            col = i % n_per_row
            best_pts[i, 0] = (col - n_per_row/2) * spacing
            best_pts[i, 1] = (row - n_per_row/2) * spacing
    
    return best_pts
# EVOLVE-BLOCK-END

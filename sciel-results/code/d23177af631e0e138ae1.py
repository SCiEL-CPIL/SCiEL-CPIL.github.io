"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    t_start = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Best known radius for reference
    R_KNOWN = 0.1258254895304
    
    def _pairwise_distances(points):
        """Compute all pairwise distances for N points."""
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        return np.sqrt(np.sum(diff**2, axis=2))
    
    def _score(points):
        """Compute the actual packing score: min(min_pair_dist/2, min_boundary_dist)."""
        if not np.all(np.isfinite(points)):
            return 0.0
        radii = np.hypot(points[:, 0], points[:, 1])
        if np.any(radii > 1.0):
            return 0.0
        min_boundary = np.min(1.0 - radii)
        dists = _pairwise_distances(points)
        iu = np.triu_indices(N, k=1)
        min_pair = np.min(dists[iu]) / 2.0
        return min(min_pair, min_boundary)
    
    def _softmin_logsumexp(values, beta):
        """Compute soft min using log-sum-exp for numerical stability."""
        if len(values) == 0:
            return 0.0
        neg_beta_x = -beta * values
        max_val = np.max(neg_beta_x)
        if not np.isfinite(max_val):
            return -10.0
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_x - max_val)))
        return -log_sum_exp / beta
    
    def _smooth_objective(points_flat, beta=50.0):
        """Smooth objective for optimization."""
        points = points_flat.reshape(N, 2)
        radii = np.hypot(points[:, 0], points[:, 1])
        
        # Penalty for going outside
        outside = np.maximum(0, radii - 1.0)
        if np.any(outside > 0):
            penalty = np.sum(outside**2) * 100.0
        
        dists = _pairwise_distances(points)
        iu = np.triu_indices(N, k=1)
        min_pair_dists = dists[iu]
        boundary_dists = 1.0 - radii
        
        all_dists = np.concatenate([min_pair_dists, boundary_dists])
        soft_min = _softmin_logsumexp(all_dists, beta)
        
        return -soft_min + penalty
    
    def _ring_initialization(radii, angles):
        """Create points from ring structure."""
        points = np.zeros((N, 2))
        # Ring 0: center
        points[0] = [0.0, 0.0]
        idx = 1
        ring_counts = [6, 12, 18, 14]
        for k, count in enumerate(ring_counts):
            R = radii[k]
            phi = angles[k]
            theta = phi + 2 * np.pi * np.arange(count) / count
            points[idx:idx+count, 0] = R * np.cos(theta)
            points[idx:idx+count, 1] = R * np.sin(theta)
            idx += count
        return points
    
    def _hexagonal_init():
        """Generate a hexagonal lattice initial configuration."""
        # Try to fit 50 circles in a hexagonal pattern
        # Optimal hexagonal packing has density ~0.9069
        # For 50 circles, we need spacing such that 50 circles fit in unit circle
        # Area per circle ~ pi*r^2, total area ~ 50*pi*r^2 <= pi * 0.9069
        # r^2 <= 0.9069/50, r <= 0.1347 (theoretical max for hexagonal)
        
        # Try different spacings
        best_points = None
        best_score = 0
        
        for spacing_mult in np.linspace(0.85, 1.15, 31):
            # Estimate spacing from area
            base_spacing = np.sqrt(1.0 / (50 * 0.9069))
            spacing = base_spacing * spacing_mult
            
            # Generate hexagonal lattice points
            k = int(1.0 / (spacing * np.sqrt(3)/2)) + 3
            pts = []
            for j in range(-k, k+1):
                for i in range(-k, k+1):
                    x = i * spacing + (j % 2) * spacing / 2
                    y = j * spacing * np.sqrt(3) / 2
                    r = np.sqrt(x*x + y*y)
                    if r <= 1.0 - spacing/2:
                        pts.append((x, y, r))
            
            if len(pts) >= N:
                # Select N points closest to center
                pts.sort(key=lambda p: p[2])
                selected = np.array([(p[0], p[1]) for p in pts[:N]])
                s = _score(selected)
                if s > best_score:
                    best_score = s
                    best_points = selected
        
        if best_points is not None:
            return best_points
        
        # Fallback: uniform random
        return np.random.randn(N, 2) * 0.3
    
    def _optimize_from_init(points, beta=50.0, maxiter=300):
        """Optimize points from initial configuration."""
        x0 = points.flatten()
        
        def obj(x):
            return _smooth_objective(x, beta)
        
        res = minimize(obj, x0, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12})
        return res.x.reshape(N, 2)
    
    def _binary_search_polish(points, r_low, r_high, max_iter=10):
        """Binary search on radius to polish the packing."""
        for _ in range(max_iter):
            r_mid = (r_low + r_high) / 2
            # Check if packing is valid at radius r_mid
            valid = True
            radii = np.hypot(points[:, 0], points[:, 1])
            if np.any(radii > 1.0 - r_mid):
                valid = False
            else:
                dists = _pairwise_distances(points)
                iu = np.triu_indices(N, k=1)
                if np.any(dists[iu] < 2 * r_mid):
                    valid = False
            
            if valid:
                r_low = r_mid
            else:
                r_high = r_mid
                # Try to fix violations
                # Push points that violate constraints
                for i in range(N):
                    # Boundary violation
                    ri = np.hypot(points[i, 0], points[i, 1])
                    if ri > 1.0 - r_mid:
                        scale = (1.0 - r_mid) / ri if ri > 0 else 1.0
                        points[i] *= scale
                    # Pairwise violations
                    for j in range(i+1, N):
                        d = points[i] - points[j]
                        dist = np.sqrt(np.sum(d**2))
                        if dist < 2 * r_mid and dist > 1e-10:
                            push = (2 * r_mid - dist) / 2
                            d_unit = d / dist
                            points[i] += push * d_unit
                            points[j] -= push * d_unit
                        elif dist < 2 * r_mid:
                            # Coincident points, separate randomly
                            angle = np.random.uniform(0, 2*np.pi)
                            points[i] += np.array([np.cos(angle), np.sin(angle)]) * r_mid * 0.1
                            points[j] -= np.array([np.cos(angle), np.sin(angle)]) * r_mid * 0.1
                
                # Re-optimize after fixing
                points = _optimize_from_init(points, beta=80.0, maxiter=50)
        
        return points, r_low
    
    def _de_ring_optimize():
        """Differential Evolution on ring parameters."""
        ring_counts = [6, 12, 18, 14]
        
        def ring_obj(params):
            radii = params[:4]
            angles = params[4:8]
            points = _ring_initialization(radii, angles)
            return _smooth_objective(points.flatten(), beta=50.0)
        
        # Bounds: radii in [0.1, 0.95], angles in [0, 2*pi]
        bounds = [(0.1, 0.95)] * 4 + [(0, 2*np.pi)] * 4
        x0 = np.array([0.25, 0.45, 0.65, 0.85, 0, np.pi/6, 0, np.pi/14])
        
        res = minimize(ring_obj, x0, method='Nelder-Mead',
                      options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
        
        best_params = res.x
        best_score = -res.fun
        
        # Try a few random restarts
        for _ in range(5):
            x_rand = np.array([np.random.uniform(0.15, 0.9) for _ in range(4)] +
                             [np.random.uniform(0, 2*np.pi) for _ in range(4)])
            res_r = minimize(ring_obj, x_rand, method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10})
            if -res_r.fun > best_score:
                best_score = -res_r.fun
                best_params = res_r.x
        
        radii = best_params[:4]
        angles = best_params[4:8]
        points = _ring_initialization(radii, angles)
        return points
    
    # Stage 1: Multiple initialization strategies
    candidates = []
    
    # Hexagonal init
    hex_points = _hexagonal_init()
    candidates.append(hex_points)
    
    # Ring init with DE
    ring_points = _de_ring_optimize()
    candidates.append(ring_points)
    
    # Random init
    for _ in range(3):
        random_points = np.random.randn(N, 2) * 0.4
        candidates.append(random_points)
    
    # Stage 2: Optimize each candidate
    optimized = []
    for i, pts in enumerate(candidates):
        if time.time() > deadline - 30:
            break
        opt_pts = _optimize_from_init(pts, beta=50.0, maxiter=200)
        optimized.append(opt_pts)
    
    # Stage 3: Select best and polish
    best_points = None
    best_score = 0
    for pts in optimized:
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts
    
    if best_points is None:
        best_points = hex_points
        best_score = _score(hex_points)
    
    # Stage 4: Binary search polish
    if time.time() < deadline - 15:
        polished, r_low = _binary_search_polish(best_points.copy(), best_score * 0.99, R_KNOWN * 1.01, max_iter=8)
        polished_score = _score(polished)
        if polished_score > best_score:
            best_points = polished
            best_score = polished_score
    
    # Stage 5: Final fine optimization
    if time.time() < deadline - 5:
        final_points = _optimize_from_init(best_points, beta=100.0, maxiter=100)
        final_score = _score(final_points)
        if final_score > best_score:
            best_points = final_points
            best_score = final_score
    
    # Ensure validity
    final_score = _score(best_points)
    if final_score <= 0:
        # Fallback to a simple valid configuration
        best_points = np.zeros((N, 2))
        # Place circles in a spiral
        for i in range(N):
            angle = i * 2.4  # golden angle
            r = 0.8 * np.sqrt(i / N)
            best_points[i] = [r * np.cos(angle), r * np.sin(angle)]
    
    return best_points

# EVOLVE-BLOCK-END

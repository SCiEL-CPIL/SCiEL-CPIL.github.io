<<<<<<< SEARCH
import numpy as np
import time
from scipy.optimize import minimize

N = 300

def construct():
    t0 = time.time()
    # 150s limit. Keep 10s margin.
    DEADLINE = t0 + 135.0

    def elapsed():
        return time.time() - t0

    def score(points):
        """
        Calculate the maximum radius for the given centers.
        points: (N, 2) array.
        """
        if points.shape[0] < 2:
            return 0.0
        
        # Calculate pairwise squared distances
        # Using broadcasting
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Set diagonal to infinity to ignore self-distance
        np.fill_diagonal(dist_sq, np.inf)
        
        min_dist_sq = np.min(dist_sq)
        min_dist = np.sqrt(min_dist_sq)
        
        # Calculate maximum distance from origin
        norms = np.linalg.norm(points, axis=1)
        max_norm = np.max(norms)
        
        # Radius is limited by both pair distances and boundary
        r_pair = 0.5 * min_dist
        r_bound = 1.0 - max_norm
        
        return min(r_pair, r_bound)

    def objective(params):
        """
        Objective function for minimization.
        We want to maximize score, so we minimize -score.
        """
        pts = params.reshape(N, 2)
        # Add a small penalty for non-finite values just in case
        if not np.all(np.isfinite(pts)):
            return 1e6
        s = score(pts)
        return -s

    def gen_hex_seed():
        """
        Generate a hexagonal lattice seed.
        """
        # Estimate initial spacing
        # r ~ sqrt(0.9 / N)
        r_est = np.sqrt(0.9 / N)
        spacing = 2.0 * r_est
        
        k = int(1.0 / spacing) + 3
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    pts.append([x, y])
        
        pts = np.array(pts)
        
        # If we have more than N points, select the ones closest to the center
        # This creates a "filled" disk shape
        if len(pts) > N:
            radii = np.linalg.norm(pts, axis=1)
            idx = np.argsort(radii)
            pts = pts[idx[:N]]
        elif len(pts) < N:
            # Shrink spacing and retry
            while len(pts) < N:
                spacing *= 0.95
                k = int(1.0 / spacing) + 3
                pts = []
                for j in range(-k, k + 1):
                    for i in range(-k, k + 1):
                        x = (i + 0.5 * (j % 2)) * spacing
                        y = j * spacing * np.sqrt(3.0) / 2.0
                        if x*x + y*y <= 1.0:
                            pts.append([x, y])
                pts = np.array(pts)
                if len(pts) > N:
                    radii = np.linalg.norm(pts, axis=1)
                    idx = np.argsort(radii)
                    pts = pts[idx[:N]]
        
        return pts

    # Initial Seed
    best_pts = gen_hex_seed()
    best_score = score(best_pts)
    
    # Optimization Loop
    # We use L-BFGS-B which is fast.
    # We perform multiple restarts to escape local minima.
    
    current_pts = best_pts.copy()
    
    while elapsed() < DEADLINE - 5:
        # Perturb the current solution for restart
        # Use a small noise to avoid getting stuck in the exact same local minimum
        noise_scale = 0.005
        current_pts = current_pts + np.random.normal(0, noise_scale, current_pts.shape)
        
        # Ensure points are roughly inside (clip to [-1, 1] to help optimizer)
        # The score function handles the actual constraint, but keeping them
        # in bounds helps the numerical gradient.
        current_pts = np.clip(current_pts, -1.0, 1.0)
        
        x0 = current_pts.flatten()
        
        try:
            # Run optimizer
            # maxiter=100 is usually enough for a local polish
            res = minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
            )
            
            if res.x is not None:
                cand_pts = res.x.reshape(N, 2)
                
                # Validate
                if np.all(np.isfinite(cand_pts)):
                    cand_score = score(cand_pts)
                    
                    if cand_score > best_score:
                        best_score = cand_score
                        best_pts = cand_pts
                        # Update current to be the best found so far for next restart
                        # Or keep it as the result of this run?
                        # Using the best found is safer.
                        current_pts = best_pts.copy()
                        
        except Exception:
            pass

    return best_pts
=======
import numpy as np
import time
from scipy.optimize import minimize

N = 300

def construct():
    t0 = time.time()
    # 150s limit. Keep 10s margin.
    DEADLINE = t0 + 135.0

    def elapsed():
        return time.time() - t0

    def score(points):
        """
        Calculate the maximum radius for the given centers.
        points: (N, 2) array.
        """
        if points.shape[0] < 2:
            return 0.0
        
        # Calculate pairwise squared distances
        # Using broadcasting
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Set diagonal to infinity to ignore self-distance
        np.fill_diagonal(dist_sq, np.inf)
        
        min_dist_sq = np.min(dist_sq)
        min_dist = np.sqrt(min_dist_sq)
        
        # Calculate maximum distance from origin
        norms = np.linalg.norm(points, axis=1)
        max_norm = np.max(norms)
        
        # Radius is limited by both pair distances and boundary
        r_pair = 0.5 * min_dist
        r_bound = 1.0 - max_norm
        
        return min(r_pair, r_bound)

    def gen_ring_seed():
        """
        Generate a ring-based seed.
        We divide N points into concentric rings.
        """
        # Estimate number of rings
        # Area ~ N * (2r)^2 / (2/sqrt(3)) approx.
        # r ~ 0.053
        # Let's try to distribute points roughly evenly by area.
        # Ring k has radius r_k. Number of points n_k ~ 2 * pi * r_k * dr / (2r_circle)
        # Let's just guess a distribution: more points in outer rings.
        
        # Simple heuristic:
        # Try different numbers of rings
        best_seed = None
        best_seed_score = -np.inf
        
        for num_rings in range(5, 20):
            # Radii: from 0 to 1
            # Use sqrt distribution to equalize area
            radii = np.sqrt(np.linspace(0.05, 1.0, num_rings + 1))
            
            # Determine points per ring
            # n_k proportional to circumference * thickness
            # thickness = r_{k+1} - r_k
            # n_k ~ 2 * pi * r_k * (r_{k+1} - r_k) / (2 * r_circle_est)
            # r_circle_est ~ 0.053
            # Let's just normalize the weights
            weights = []
            for k in range(num_rings):
                r_mid = (radii[k] + radii[k+1]) / 2.0
                thickness = radii[k+1] - radii[k]
                w = 2.0 * np.pi * r_mid * thickness
                weights.append(w)
            
            weights = np.array(weights)
            weights /= np.sum(weights)
            
            # Allocate N points
            counts = np.floor(weights * N).astype(int)
            # Distribute remainder
            remainder = N - np.sum(counts)
            # Add to rings with largest fractional parts
            fracs = weights * N - counts
            idx = np.argsort(fracs)[::-1]
            for i in range(remainder):
                counts[idx[i % len(counts)]] += 1
            
            # Ensure center ring has at least 1 point if num_rings > 1? 
            # Actually, let's put 1 point at center if possible, or just start rings from small r.
            # If counts[0] is 0, we might have a hole. Let's force counts[0] = 1 if num_rings > 1 and N > 10.
            if num_rings > 1 and N > 10:
                if counts[0] == 0:
                    # Take from the ring with most points
                    max_idx = np.argmax(counts[1:]) + 1
                    if counts[max_idx] > 1:
                        counts[max_idx] -= 1
                        counts[0] = 1
                    else:
                        # Just leave it, or put 1 at center anyway?
                        # For now, let's just use the counts as is.
                        pass
            
            # Generate points
            pts = []
            for k in range(num_rings):
                n_k = counts[k]
                r_k = radii[k]
                if n_k == 1 and k == 0:
                    # Center point
                    pts.append([0.0, 0.0])
                elif n_k > 0:
                    angles = np.linspace(0, 2*np.pi, n_k, endpoint=False)
                    # Add a small offset to avoid perfect alignment with other rings?
                    # Maybe randomize phase
                    phase = np.random.uniform(0, 2*np.pi)
                    angles += phase
                    for a in angles:
                        pts.append([r_k * np.cos(a), r_k * np.sin(a)])
            
            pts = np.array(pts)
            
            # Check if we have exactly N points
            if len(pts) != N:
                continue
                
            s = score(pts)
            if s > best_seed_score:
                best_seed_score = s
                best_seed = pts
        
        if best_seed is None:
            # Fallback to hex
            return gen_hex_seed()
            
        return best_seed

    def gen_hex_seed():
        """
        Generate a hexagonal lattice seed.
        """
        # Estimate initial spacing
        # r ~ sqrt(0.9 / N)
        r_est = np.sqrt(0.9 / N)
        spacing = 2.0 * r_est
        
        k = int(1.0 / spacing) + 3
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x*x + y*y <= 1.0:
                    pts.append([x, y])
        
        pts = np.array(pts)
        
        # If we have more than N points, select the ones closest to the center
        # This creates a "filled" disk shape
        if len(pts) > N:
            radii = np.linalg.norm(pts, axis=1)
            idx = np.argsort(radii)
            pts = pts[idx[:N]]
        elif len(pts) < N:
            # Shrink spacing and retry
            while len(pts) < N:
                spacing *= 0.95
                k = int(1.0 / spacing) + 3
                pts = []
                for j in range(-k, k + 1):
                    for i in range(-k, k + 1):
                        x = (i + 0.5 * (j % 2)) * spacing
                        y = j * spacing * np.sqrt(3.0) / 2.0
                        if x*x + y*y <= 1.0:
                            pts.append([x, y])
                pts = np.array(pts)
                if len(pts) > N:
                    radii = np.linalg.norm(pts, axis=1)
                    idx = np.argsort(radii)
                    pts = pts[idx[:N]]
        
        return pts

    # Initial Seed
    # Try both ring and hex seeds, pick the better one
    ring_seed = gen_ring_seed()
    hex_seed = gen_hex_seed()
    
    ring_score = score(ring_seed)
    hex_score = score(hex_seed)
    
    if ring_score > hex_score:
        best_pts = ring_seed
        best_score = ring_score
    else:
        best_pts = hex_seed
        best_score = hex_score
    
    # Optimization Loop
    # We use L-BFGS-B which is fast.
    # We perform multiple restarts to escape local minima.
    
    current_pts = best_pts.copy()
    
    while elapsed() < DEADLINE - 5:
        # Perturb the current solution for restart
        # Use a small noise to avoid getting stuck in the exact same local minimum
        noise_scale = 0.005
        current_pts = current_pts + np.random.normal(0, noise_scale, current_pts.shape)
        
        # Ensure points are roughly inside (clip to [-1, 1] to help optimizer)
        # The score function handles the actual constraint, but keeping them
        # in bounds helps the numerical gradient.
        current_pts = np.clip(current_pts, -1.0, 1.0)
        
        x0 = current_pts.flatten()
        
        try:
            # Run optimizer
            # maxiter=100 is usually enough for a local polish
            res = minimize(
                lambda x: -score(x.reshape(N, 2)), 
                x0, 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
            )
            
            if res.x is not None:
                cand_pts = res.x.reshape(N, 2)
                
                # Validate
                if np.all(np.isfinite(cand_pts)):
                    cand_score = score(cand_pts)
                    
                    if cand_score > best_score:
                        best_score = cand_score
                        best_pts = cand_pts
                        # Update current to be the best found so far for next restart
                        # Or keep it as the result of this run?
                        # Using the best found is safer.
                        current_pts = best_pts.copy()
                        
        except Exception:
            pass

    return best_pts
>>>>>>> REPLACE

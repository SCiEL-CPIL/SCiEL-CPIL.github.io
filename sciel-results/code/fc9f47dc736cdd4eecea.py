"""Ring-based circle packing for 300 equal circles in unit circle."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit

    def compute_score(points):
        n = len(points)
        if n < 2:
            return 0.0
        # Compute all pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        # Compute minimum distance to boundary
        radii = np.sqrt(np.sum(points ** 2, axis=1))
        min_boundary_dist = np.min(1.0 - radii)
        return min(min_pair_dist / 2.0, min_boundary_dist)

    def ring_configuration(ring_params):
        """
        Generate points from ring parameters.
        ring_params: array of shape (n_rings, 3) where each row is [radius, n_points, phase]
        Returns array of shape (N, 2)
        """
        points = []
        for r, n_pts, phase in ring_params:
            if n_pts < 1:
                continue
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False) + phase
            xs = r * np.cos(angles)
            ys = r * np.sin(angles)
            points.append(np.column_stack([xs, ys]))
        
        if not points:
            return np.zeros((N, 2))
        
        all_points = np.vstack(points)
        
        # If we have more than N points, select the N that give best score
        if len(all_points) > N:
            # Select points closest to center first, then adjust
            radii = np.sqrt(np.sum(all_points ** 2, axis=1))
            order = np.argsort(radii, kind='stable')[:N]
            all_points = all_points[order]
        elif len(all_points) < N:
            # Pad with points near center
            deficit = N - len(all_points)
            pad_angles = np.linspace(0, 2 * np.pi, deficit, endpoint=False)
            pad_r = 0.01
            pad_pts = np.column_stack([pad_r * np.cos(pad_angles), pad_r * np.sin(pad_angles)])
            all_points = np.vstack([all_points, pad_pts])
        
        return all_points[:N]

    def ring_objective(params):
        """
        Objective function for ring optimization.
        params: flattened array of ring parameters [r1, n1, phi1, r2, n2, phi2, ...]
        Returns negative score (for minimization)
        """
        n_rings = len(params) // 3
        ring_params = params.reshape(n_rings, 3)
        
        # Ensure radii are in valid range
        ring_params[:, 0] = np.clip(ring_params[:, 0], 0.0, 0.999)
        
        # Ensure n_points are integers >= 1
        n_points = np.maximum(1, np.round(ring_params[:, 1])).astype(int)
        
        # Reconstruct ring params with integer n_points
        ring_params_int = ring_params.copy()
        ring_params_int[:, 1] = n_points
        
        points = ring_configuration(ring_params_int)
        score = compute_score(points)
        return -score

    # Initial ring configuration
    # Estimate number of rings and points per ring
    # For optimal packing, rings should have increasing number of points
    
    # Start with a reasonable initial guess
    # Total points N=300, try ~10-15 rings
    n_rings_init = 12
    
    # Distribute points across rings with increasing count
    # Use a formula that approximates optimal distribution
    # Ring i has roughly 6*i points (hexagonal-like)
    
    # Initialize ring parameters
    ring_params = np.zeros((n_rings_init, 3))
    
    # Radii: from small to large
    # Use a power law for better distribution
    for i in range(n_rings_init):
        # Radius increases with ring index
        ring_params[i, 0] = 0.1 + 0.85 * (i / (n_rings_init - 1))
        
        # Number of points: increases with ring radius
        # Approximate: n_i ~ 2*pi*r_i / (2*r_circle)
        # For r_circle ~ 0.053, spacing ~ 0.106
        # n_i ~ 2*pi*r_i / 0.106 ~ 60*r_i
        ring_params[i, 1] = max(1, int(2 * np.pi * ring_params[i, 0] / 0.106))
        
        # Phase: alternate for better packing
        ring_params[i, 2] = np.pi / ring_params[i, 1] if i % 2 == 1 else 0.0
    
    # Adjust to get exactly N points
    total_pts = int(np.sum(ring_params[:, 1]))
    if total_pts != N:
        # Adjust the largest ring to compensate
        diff = N - total_pts
        ring_params[-1, 1] += diff
        ring_params[-1, 1] = max(1, ring_params[-1, 1])
        
        # Re-check
        total_pts = int(np.sum(ring_params[:, 1]))
        if total_pts != N:
            # Force adjustment
            ring_params[:, 1] = np.maximum(1, np.round(ring_params[:, 1])).astype(int)
            diff = N - int(np.sum(ring_params[:, 1]))
            ring_params[0, 1] += diff
            ring_params[0, 1] = max(1, ring_params[0, 1])
    
    # Initial score
    points_init = ring_configuration(ring_params)
    best_score = compute_score(points_init)
    best_points = points_init.copy()
    
    # Optimization using scipy
    x0 = ring_params.flatten()
    
    # Bounds: radius in [0, 0.999], n_points in [1, 500], phase in [0, 2*pi]
    bounds = []
    for i in range(n_rings_init):
        bounds.append((0.0, 0.999))  # radius
        bounds.append((1.0, 500.0))  # n_points
        bounds.append((0.0, 2 * np.pi))  # phase
    
    # Try L-BFGS-B first
    try:
        result = minimize(ring_objective, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-8, 'disp': False})
        
        if result.fun < -best_score:
            new_params = result.x.reshape(n_rings_init, 3)
            new_params[:, 0] = np.clip(new_params[:, 0], 0.0, 0.999)
            new_points = ring_configuration(new_params)
            new_score = compute_score(new_points)
            if new_score > best_score:
                best_score = new_score
                best_points = new_points
    except Exception:
        pass
    
    # Try Nelder-Mead on a subset of parameters for refinement
    if time.time() < deadline - 30:
        # Only optimize radii and phases, keep n_points fixed
        n_params = n_rings_init * 2  # radius and phase only
        
        def reduced_objective(x):
            params = np.zeros((n_rings_init, 3))
            for i in range(n_rings_init):
                params[i, 0] = np.clip(x[i * 2], 0.0, 0.999)
                params[i, 1] = ring_params[i, 1]  # Keep n_points fixed
                params[i, 2] = x[i * 2 + 1]
            points = ring_configuration(params)
            return -compute_score(points)
        
        x0_reduced = np.zeros(n_params)
        for i in range(n_rings_init):
            x0_reduced[i * 2] = ring_params[i, 0]
            x0_reduced[i * 2 + 1] = ring_params[i, 2]
        
        bounds_reduced = []
        for i in range(n_rings_init):
            bounds_reduced.append((0.0, 0.999))  # radius
            bounds_reduced.append((0.0, 2 * np.pi))  # phase
        
        try:
            result2 = minimize(reduced_objective, x0_reduced, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False})
            
            if result2.fun < -best_score:
                new_params = np.zeros((n_rings_init, 3))
                for i in range(n_rings_init):
                    new_params[i, 0] = np.clip(result2.x[i * 2], 0.0, 0.999)
                    new_params[i, 1] = ring_params[i, 1]
                    new_params[i, 2] = result2.x[i * 2 + 1]
                new_points = ring_configuration(new_params)
                new_score = compute_score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points
        except Exception:
            pass
    
    # Final local optimization using force-directed method on the ring-generated points
    if time.time() < deadline - 10:
        points = best_points.copy()
        n = len(points)
        
        # Compute forces
        for iteration in range(500):
            if time.time() > deadline - 5:
                break
            
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dist, np.inf)
            
            # Find minimum distance
            min_dist = np.min(dist)
            
            # Compute repulsion forces
            forces = np.zeros_like(points)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    d = dist[i, j]
                    if d < 2 * min_dist:
                        # Strong repulsion for close pairs
                        force_mag = 1.0 / (d ** 2 + 1e-10)
                        forces[i] += force_mag * (points[i] - points[j]) / d
            
            # Boundary force
            radii = np.sqrt(np.sum(points ** 2, axis=1))
            for i in range(n):
                if radii[i] > 0.95:
                    forces[i] -= 0.5 * points[i] / radii[i]
            
            # Normalize and update
            force_norms = np.sqrt(np.sum(forces ** 2, axis=1))
            max_force = np.max(force_norms)
            if max_force > 1e-10:
                forces /= max_force
            
            step = 0.01 * min_dist
            points += step * forces
            
            # Project back inside
            radii = np.sqrt(np.sum(points ** 2, axis=1))
            mask_out = radii > 1.0
            if np.any(mask_out):
                points[mask_out] /= radii[mask_out, np.newaxis]
            
            current_score = compute_score(points)
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
            
            # Decrease step size
            step *= 0.99
        
        # Final check
        final_score = compute_score(best_points)
        if final_score > best_score:
            best_score = final_score
    
    return best_points

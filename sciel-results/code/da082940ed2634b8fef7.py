import numpy as np
import time
import random

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_centers = None
    best_score = -np.inf
    
    # Helper to compute the actual score (radius) for a given set of centers
    def compute_radius(centers):
        if len(centers) != N:
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        
        # Boundary check
        dists_boundary = 1.0 - np.hypot(centers[:, 0], centers[:, 1])
        r_boundary = np.min(dists_boundary)
        if r_boundary <= 0:
            return 0.0
        
        # Pairwise distances
        dx = centers[:, 0:1] - centers[:, 0:1].T
        dy = centers[:, 1:2] - centers[:, 1:2].T
        dists = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dists, np.inf)
        r_pair = np.min(dists) / 2.0
        
        return min(r_boundary, r_pair)

    # Fast energy function using squared distances (no sqrt)
    def calculate_energy_fast(centers, r):
        # Pairwise overlaps using squared distances
        dx = centers[:, 0:1] - centers[:, 0:1].T
        dy = centers[:, 1:2] - centers[:, 1:2].T
        dists_sq = dx**2 + dy**2
        np.fill_diagonal(dists_sq, np.inf)
        
        # Overlap: (2r - d)^2 where d = sqrt(dists_sq)
        # Use (2r)^2 - dists_sq as proxy, but need to be careful
        # Actually, let's use a smooth approximation:
        # For each pair, if dists_sq < (2r)^2, penalty = (2r)^2 - dists_sq
        # This is not exactly (2r-d)^2 but is smooth and monotonic
        min_dist_sq = (2*r)**2
        overlaps_sq = np.maximum(0, min_dist_sq - dists_sq)
        E_pair = np.sum(overlaps_sq)
        
        # Boundary violations
        dists_boundary = np.hypot(centers[:, 0], centers[:, 1])
        boundary_violations = np.maximum(0, dists_boundary + r - 1.0)
        E_bound = np.sum(boundary_violations**2)
        
        return E_pair + E_bound

    # Generate initial hexagonal lattice points
    def get_initial_points(r_init):
        s = 0.24
        k = int(1.0 / s) + 2
        points = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * s
                y = j * s * np.sqrt(3.0) / 2.0
                points.append((x, y))
        points = np.array(points)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - r_init]
        
        if len(points) < N:
            s *= 0.95
            return get_initial_points(r_init)
        
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx]

    # Directed move: push point away from nearest overlapping neighbor
    def directed_move(centers, idx, r):
        x, y = centers[idx]
        
        # Find nearest neighbor that is overlapping
        dx = centers[:, 0] - x
        dy = centers[:, 1] - y
        dists_sq = dx**2 + dy**2
        dists_sq[idx] = np.inf
        
        # Find points that are too close
        too_close = dists_sq < (2*r)**2
        if np.any(too_close):
            # Get the closest too-close point
            close_indices = np.where(too_close)[0]
            closest_idx = close_indices[np.argmin(dists_sq[close_indices])]
            
            # Direction away from that point
            dir_x = x - centers[closest_idx, 0]
            dir_y = y - centers[closest_idx, 1]
            norm = np.sqrt(dir_x**2 + dir_y**2)
            if norm > 1e-10:
                dir_x /= norm
                dir_y /= norm
            
            # Step size proportional to overlap
            overlap = 2*r - np.sqrt(dists_sq[closest_idx])
            step = np.clip(overlap * 0.5, 0.001, 0.05)
            
            new_x = x + dir_x * step
            new_y = y + dir_y * step
        else:
            # No overlap, try to move toward boundary or random
            if np.random.random() < 0.5:
                # Move toward boundary
                dist_to_center = np.sqrt(x**2 + y**2)
                if dist_to_center > 1e-10:
                    dir_x = x / dist_to_center
                    dir_y = y / dist_to_center
                    step = np.random.uniform(0.001, 0.01)
                    new_x = x + dir_x * step
                    new_y = y + dir_y * step
                else:
                    angle = np.random.uniform(0, 2*np.pi)
                    step = np.random.uniform(0.001, 0.01)
                    new_x = x + step * np.cos(angle)
                    new_y = y + step * np.sin(angle)
            else:
                angle = np.random.uniform(0, 2*np.pi)
                step = np.random.uniform(0.001, 0.02)
                new_x = x + step * np.cos(angle)
                new_y = y + step * np.sin(angle)
        
        # Clamp to stay within unit circle (with margin)
        dist = np.sqrt(new_x**2 + new_y**2)
        max_dist = 1.0 - r - 0.001
        if dist > max_dist:
            scale = max_dist / dist if dist > 1e-10 else 1.0
            new_x *= scale
            new_y *= scale
        
        return new_x, new_y

    # Final polish: maximize radius by adjusting critical points
    def polish(centers, time_limit):
        """
        Local optimization to maximize the minimum radius.
        Uses gradient-like steps on the critical constraints.
        """
        centers = centers.copy()
        end_time = time.time() + time_limit
        step_size = 0.005
        
        while time.time() < end_time:
            # Compute current radius and identify critical constraints
            r_current = compute_radius(centers)
            
            # Find which constraint is limiting
            dists_boundary = 1.0 - np.hypot(centers[:, 0], centers[:, 1])
            r_boundary = np.min(dists_boundary)
            
            dx = centers[:, 0:1] - centers[:, 0:1].T
            dy = centers[:, 1:2] - centers[:, 1:2].T
            dists = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(dists, np.inf)
            r_pair = np.min(dists) / 2.0
            
            # Identify critical points
            critical_boundary = np.where(np.isclose(dists_boundary, r_boundary, atol=1e-6))[0]
            critical_pairs = np.argwhere(np.isclose(dists, 2*r_pair, atol=1e-6))
            
            # Try to move critical points to increase the minimum
            moved = False
            
            # Move boundary-critical points inward slightly
            for idx in critical_boundary:
                x, y = centers[idx]
                dist = np.sqrt(x**2 + y**2)
                if dist > 1e-10:
                    # Move toward center
                    new_x = x * (1 - step_size / dist)
                    new_y = y * (1 - step_size / dist)
                    old_energy = calculate_energy_fast(centers, r_current)
                    centers[idx] = [new_x, new_y]
                    new_energy = calculate_energy_fast(centers, r_current)
                    if new_energy <= old_energy:
                        moved = True
                    else:
                        centers[idx] = [x, y]
            
            # Move pair-critical points apart
            for i, j in critical_pairs:
                if i == j:
                    continue
                xi, yi = centers[i]
                xj, yj = centers[j]
                dx_ij = xi - xj
                dy_ij = yi - yj
                dist_ij = np.sqrt(dx_ij**2 + dy_ij**2)
                if dist_ij > 1e-10:
                    dir_i = dx_ij / dist_ij
                    dir_j = -dir_i
                    step = step_size / 2
                    
                    old_energy = calculate_energy_fast(centers, r_current)
                    centers[i] = [xi + dir_i * step, yi + dir_i * step * dy_ij/dx_ij if abs(dx_ij)>1e-10 else yi]
                    # Simpler: just move along the line connecting them
                    centers[i, 0] = xi + (xi - xj) / dist_ij * step
                    centers[i, 1] = yi + (yi - yj) / dist_ij * step
                    centers[j, 0] = xj - (xi - xj) / dist_ij * step
                    centers[j, 1] = yj - (yi - yj) / dist_ij * step
                    
                    new_energy = calculate_energy_fast(centers, r_current)
                    if new_energy <= old_energy:
                        moved = True
                    else:
                        centers[i] = [xi, yi]
                        centers[j] = [xj, yj]
            
            if not moved:
                # Try random small moves on all points
                for idx in range(N):
                    x, y = centers[idx]
                    angle = np.random.uniform(0, 2*np.pi)
                    step = np.random.uniform(0.0005, 0.002)
                    new_x = x + step * np.cos(angle)
                    new_y = y + step * np.sin(angle)
                    
                    # Check bounds
                    if new_x**2 + new_y**2 > (1.0 - r_current)**2:
                        continue
                    
                    old_energy = calculate_energy_fast(centers, r_current)
                    centers[idx] = [new_x, new_y]
                    new_energy = calculate_energy_fast(centers, r_current)
                    if new_energy <= old_energy:
                        moved = True
                    else:
                        centers[idx] = [x, y]
            
            if not moved:
                break
        
        return centers

    # Simulated Annealing with adaptive target radius
    def anneal(initial_centers, time_limit):
        current_centers = initial_centers.copy()
        
        # Start with a modest target and increase
        r_target = 0.122
        current_energy = calculate_energy_fast(current_centers, r_target)
        
        best_centers = current_centers.copy()
        best_energy = current_energy
        best_r = r_target
        
        T = 0.03
        alpha = 0.995
        step_size = 0.015
        
        end_time = time.time() + time_limit
        last_improve_time = time.time()
        
        while time.time() < end_time:
            T *= alpha
            if T < 1e-7:
                break
            
            # Try a directed move
            idx = np.random.randint(N)
            old_x, old_y = current_centers[idx]
            
            if np.random.random() < 0.7:
                new_x, new_y = directed_move(current_centers, idx, r_target)
            else:
                new_x = old_x + np.random.normal(0, step_size)
                new_y = old_y + np.random.normal(0, step_size)
                dist = np.sqrt(new_x**2 + new_y**2)
                max_dist = 1.0 - r_target - 0.001
                if dist > max_dist:
                    scale = max_dist / dist if dist > 1e-10 else 1.0
                    new_x *= scale
                    new_y *= scale
            
            current_centers[idx] = [new_x, new_y]
            new_energy = calculate_energy_fast(current_centers, r_target)
            
            delta_E = new_energy - current_energy
            
            if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
                current_energy = new_energy
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_centers = current_centers.copy()
                    last_improve_time = time.time()
                    
                    # If energy is low enough, try to increase r_target
                    if best_energy < 1e-4 and r_target < 0.126:
                        r_target += 0.0005
                        current_energy = calculate_energy_fast(current_centers, r_target)
                        best_energy = current_energy
            else:
                current_centers[idx] = [old_x, old_y]
            
            # Reheat if stuck
            if time.time() - last_improve_time > 5.0:
                T = max(T, 0.02)
                last_improve_time = time.time()
            
            if best_energy < 1e-8:
                break
                
        return best_centers, best_r

    # Main loop with restarts
    num_restarts = 4
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        r_init = 0.118
        init_centers = get_initial_points(r_init)
        
        time_left = deadline - time.time() - 15.0  # Leave 15s for polish
        if time_left < 5.0:
            break
            
        centers, r_found = anneal(init_centers, time_left)
        
        actual_r = compute_radius(centers)
        
        if actual_r > best_score:
            best_score = actual_r
            best_centers = centers
        
        # If we have a good result, use remaining time for polish
        if best_score > 0.124:
            time_left_polish = deadline - time.time() - 5.0
            if time_left_polish > 3.0:
                polished = polish(best_centers, time_left_polish)
                polished_r = compute_radius(polished)
                if polished_r > best_score:
                    best_score = polished_r
                    best_centers = polished

    if best_centers is None:
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

    return best_centers

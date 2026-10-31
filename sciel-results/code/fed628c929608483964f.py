"""Circle packing in perimeter-4 rectangle using Simulated Annealing with penalty-based objective."""
import numpy as np
import time
import random

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def penalty_objective(W, H, centers, radii):
        """
        Compute penalty-based objective: sum of radii minus penalties for overlaps and boundary violations.
        Higher is better (we maximize this).
        """
        # Sum of radii (the actual objective)
        sum_r = np.sum(radii)
        
        # Boundary penalty: circles must be inside [0, W] x [0, H]
        boundary_pen = 0.0
        for i in range(N):
            x, y = centers[i]
            r = radii[i]
            if x < r:
                boundary_pen += (r - x) ** 2
            if x > W - r:
                boundary_pen += (x - (W - r)) ** 2
            if y < r:
                boundary_pen += (r - y) ** 2
            if y > H - r:
                boundary_pen += (y - (H - r)) ** 2
        
        # Overlap penalty
        overlap_pen = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    overlap_pen += (min_dist - dist) ** 2
        
        # Scale penalties to be comparable to sum_r
        pen_scale = 50.0
        return sum_r - pen_scale * (boundary_pen + overlap_pen)

    def is_valid(W, H, centers, radii, tol=1e-7):
        """Check if packing is valid: no overlaps, all inside rectangle."""
        for i in range(N):
            x, y = centers[i]
            r = radii[i]
            if r < -tol:
                return False
            if x < r - tol or x > W - r + tol:
                return False
            if y < r - tol or y > H - r + tol:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist_sq = dx * dx + dy * dy
                min_dist = radii[i] + radii[j]
                if dist_sq < min_dist * min_dist - tol:
                    return False
        return True

    def random_init(W, H):
        """Generate a random initial configuration."""
        centers = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(N):
            r = random.uniform(0.01, 0.15)
            x = random.uniform(r, W - r)
            y = random.uniform(r, H - r)
            centers[i] = [x, y]
            radii[i] = r
        return centers, radii

    def simulated_annealing(W, H, initial_centers, initial_radii, time_limit):
        """
        Run simulated annealing to maximize sum of radii with penalties.
        """
        centers = initial_centers.copy()
        radii = initial_radii.copy()
        
        best_centers = centers.copy()
        best_radii = radii.copy()
        best_obj = penalty_objective(W, H, centers, radii)
        
        # SA parameters
        T_initial = 0.5
        T_final = 0.001
        cooling_rate = 0.9995  # Slow cooling for thorough exploration
        
        T = T_initial
        step_size = 0.05
        
        t0 = time.time()
        
        while T > T_final and time.time() < time_limit:
            # Generate a perturbation
            # Randomly select which circles to perturb
            n_perturb = random.randint(1, 3)
            idxs = random.sample(range(N), n_perturb)
            
            new_centers = centers.copy()
            new_radii = radii.copy()
            
            for idx in idxs:
                # Perturb position
                angle = random.uniform(0, 2 * np.pi)
                mag = random.uniform(0, step_size)
                new_centers[idx][0] += mag * np.cos(angle)
                new_centers[idx][1] += mag * np.sin(angle)
                
                # Perturb radius
                new_radii[idx] *= (1 + random.uniform(-0.1, 0.1))
                new_radii[idx] = max(0.005, min(new_radii[idx], min(W, H) / 2))
                
                # Clamp to bounds
                r = new_radii[idx]
                new_centers[idx][0] = np.clip(new_centers[idx][0], r, W - r)
                new_centers[idx][1] = np.clip(new_centers[idx][1], r, H - r)
            
            new_obj = penalty_objective(W, H, new_centers, new_radii)
            
            # Acceptance criterion
            delta_obj = new_obj - penalty_objective(W, H, centers, radii)
            if delta_obj > 0 or random.random() < np.exp(delta_obj / T):
                centers = new_centers
                radii = new_radii
                current_obj = new_obj
                if current_obj > best_obj:
                    best_obj = current_obj
                    best_centers = centers.copy()
                    best_radii = radii.copy()
            
            # Cool down
            T *= cooling_rate
            # Gradually reduce step size
            step_size = max(0.005, step_size * 0.999)
        
        return best_centers, best_radii, best_obj

    # Main optimization
    random.seed(42)
    np.random.seed(42)
    
    best_sum_r = 0.0
    best_config = None
    best_W = 1.0
    best_H = 1.0
    
    # Try different aspect ratios
    aspect_ratios = [0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
    
    for W_ratio in aspect_ratios:
        if time.time() > deadline - 15:
            break
        
        W = W_ratio
        H = 2.0 - W
        if W <= 0.3 or H <= 0.3:
            continue
        
        # Multiple restarts per aspect ratio
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initial configuration
            centers, radii = random_init(W, H)
            
            # Run SA
            time_limit = deadline - 5
            best_centers, best_radii, best_obj = simulated_annealing(
                W, H, centers, radii, time_limit
            )
            
            # Validate
            if not is_valid(W, H, best_centers, best_radii, tol=1e-6):
                continue
            
            total_r = sum(best_radii)
            if total_r > best_sum_r:
                best_sum_r = total_r
                # Build the output array: (x, y, r) for each circle
                config = np.zeros((N, 3))
                for i in range(N):
                    config[i] = [best_centers[i][0], best_centers[i][1], best_radii[i]]
                best_config = config
                best_W = W
                best_H = H
    
    # Fallback: simple grid if nothing valid found
    if best_config is None:
        W = 1.0
        H = 1.0
        best_config = np.zeros((N, 3))
        count = 0
        for i in range(5):
            for j in range(5):
                if count >= N:
                    break
                x = (i + 0.5) / 5
                y = (j + 0.5) / 5
                r = 0.5 / 5 * 0.8
                best_config[count] = [x, y, r]
                count += 1
        best_W = W
        best_H = H
    
    return best_config

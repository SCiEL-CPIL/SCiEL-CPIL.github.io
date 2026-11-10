# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates of the 16 points.

    """
    start_time = time.time()
    deadline = start_time + 610.0  # Safety margin of 20 seconds

    n = 16

    def compute_score(points):
        if points.shape != (16, 2):
            return -np.inf
        d = pdist(points)
        if d.size == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def normalize(points):
        """Center and scale points to unit radius."""
        center = np.mean(points, axis=0)
        points = points - center
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points = points / max_dist
        return points

    def coordinate_ascent(points, step_sizes, max_rounds=50):
        """Coordinate ascent with random directions and decreasing step sizes."""
        best_points = points.copy()
        best_score = compute_score(best_points)
        
        for step_size in step_sizes:
            if time.time() >= deadline - 5:
                break
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for i in range(n):
                    if time.time() >= deadline - 5:
                        break
                    # Try moving point i in random directions
                    for _ in range(30):
                        if time.time() >= deadline - 5:
                            break
                        direction = np.random.randn(2)
                        norm = np.linalg.norm(direction)
                        if norm > 0:
                            direction /= norm
                        new_points = best_points.copy()
                        new_points[i] += direction * step_size
                        score = compute_score(new_points)
                        if score > best_score:
                            best_points = new_points
                            best_score = score
                            improved = True
                            break
        return best_points, best_score

    def gradient_ascent(points, lr=0.01, steps=1000):
        """Gradient ascent using finite differences."""
        best_points = points.copy()
        best_score = compute_score(points)
        
        for step in range(steps):
            if time.time() > deadline - 10:
                break
            
            # Compute gradient via central differences
            grad = np.zeros_like(points)
            eps = 1e-5
            for i in range(n):
                for j in range(2):
                    perturbed = points.copy()
                    perturbed[i, j] += eps
                    f_plus = compute_score(perturbed)
                    
                    perturbed = points.copy()
                    perturbed[i, j] -= eps
                    f_minus = compute_score(perturbed)
                    
                    if np.isfinite(f_plus) and np.isfinite(f_minus):
                        grad[i, j] = (f_plus - f_minus) / (2 * eps)
                    else:
                        grad[i, j] = 0.0
            
            # Update points
            points = points + lr * grad
            
            # Normalize
            points = normalize(points)
            
            current_score = compute_score(points)
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
            
            # Decay learning rate
            lr *= 0.999
        
        return best_points, best_score

    def simulated_annealing(points, max_iter=3000):
        """Simulated annealing to escape local optima."""
        best_points = points.copy()
        best_score = compute_score(points)
        current_points = points.copy()
        current_score = best_score
        temperature = 0.1
        cooling_rate = 0.9995
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Random perturbation
            perturbed = current_points + np.random.randn(n, 2) * temperature
            
            # Normalize
            perturbed = normalize(perturbed)
            
            new_score = compute_score(perturbed)
            
            if new_score > current_score or (temperature > 1e-6 and np.random.random() < np.exp((new_score - current_score) / temperature)):
                current_points = perturbed
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_points = perturbed.copy()
            
            temperature *= cooling_rate
        
        return best_points, best_score

    def local_optimize(points):
        """Multi-phase local optimization: coordinate ascent -> gradient ascent -> simulated annealing."""
        # Phase 1: Coordinate ascent
        ca_points, ca_score = coordinate_ascent(points, [0.1, 0.05, 0.02, 0.01, 0.005], max_rounds=30)
        
        # Phase 2: Gradient ascent from coordinate ascent result
        if time.time() < deadline - 15:
            ga_points, ga_score = gradient_ascent(ca_points, lr=0.01, steps=500)
            if ga_score > ca_score:
                ca_points, ca_score = ga_points, ga_score
        
        # Phase 3: Simulated annealing from gradient result
        if time.time() < deadline - 10:
            sa_points, sa_score = simulated_annealing(ca_points, max_iter=2000)
            if sa_score > ca_score:
                ca_points, ca_score = sa_points, sa_score
        
        # Phase 4: Final gradient polish
        if time.time() < deadline - 5:
            final_points, final_score = gradient_ascent(ca_points, lr=0.005, steps=300)
            if final_score > ca_score:
                ca_points, ca_score = final_points, final_score
        
        return ca_points, ca_score

    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Regular 16-gon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    config1 = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_configs.append(config1)
    
    # 2. Square grid 4x4
    xs = np.linspace(-1, 1, 4)
    ys = np.linspace(-1, 1, 4)
    grid_x, grid_y = np.meshgrid(xs, ys)
    config2 = np.column_stack([grid_x.flatten(), grid_y.flatten()])
    config2 = normalize(config2)
    initial_configs.append(config2)
    
    # 3. Two concentric rings (8 inner, 8 outer)
    inner_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    outer_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi/8
    inner_pts = 0.5 * np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    outer_pts = np.column_stack([np.cos(outer_angles), np.sin(outer_angles)])
    config3 = np.vstack([inner_pts, outer_pts])
    config3 = normalize(config3)
    initial_configs.append(config3)
    
    # 4. Three concentric rings
    ring1_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ring2_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi/5
    ring3_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    r1_pts = 0.4 * np.column_stack([np.cos(ring1_angles), np.sin(ring1_angles)])
    r2_pts = 0.7 * np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles)])
    r3_pts = 1.0 * np.column_stack([np.cos(ring3_angles), np.sin(ring3_angles)])
    config4 = np.vstack([r1_pts, r2_pts, r3_pts])
    config4 = normalize(config4)
    initial_configs.append(config4)
    
    # 5. Random with seed
    np.random.seed(42)
    config5 = np.random.randn(n, 2)
    config5 = normalize(config5)
    initial_configs.append(config5)
    
    # 6. Hexagonal lattice subset
    hex_pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i + 0.5 * j
            y = j * np.sqrt(3) / 2
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts[:16])
    config6 = normalize(hex_pts)
    initial_configs.append(config6)
    
    # Optimize each configuration using combined approach
    best_overall = None
    best_overall_score = -np.inf
    
    for i, config in enumerate(initial_configs):
        if time.time() > deadline - 20:
            break
        
        # Multi-phase local optimization
        opt_points, opt_score = local_optimize(config)
        
        if opt_score > best_overall_score:
            best_overall_score = opt_score
            best_overall = opt_points.copy()
    
    # Final normalization
    if best_overall is not None:
        center = np.mean(best_overall, axis=0)
        best_overall -= center
        max_dist = np.max(np.linalg.norm(best_overall, axis=1))
        if max_dist > 0:
            best_overall /= max_dist
    
    # Fallback to regular polygon if something went wrong
    if best_overall is None or not np.all(np.isfinite(best_overall)):
        best_overall = config1
    
    return best_overall


# EVOLVE-BLOCK-END

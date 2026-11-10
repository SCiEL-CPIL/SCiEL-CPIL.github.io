# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions to maximize (d_min/d_max)^2.
    
    Strategy: 
    1. Use Simulated Annealing (SA) to explore the configuration space.
    2. Initialize with diverse geometric structures (hexagonal, circular, random).
    3. Normalize points to unit scale (d_max = 1) to simplify the objective to d_min^2.
    4. Use a cooling schedule that balances exploration and exploitation.
    5. Restart from the best found solution if time permits.
    """
    n = 16
    start_time = time.time()
    deadline = start_time + 590  # Safety margin for 600s limit

    def time_exceeded():
        return time.time() > deadline

    def evaluate(points):
        """Calculate the actual score R = (d_min/d_max)^2"""
        dists = pdist(points, 'euclidean')
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def generate_hex_grid(radius, spacing=1.0):
        points = []
        for q in range(-radius, radius + 1):
            r_min = max(-radius, -q - radius)
            r_max = min(radius, -q + radius)
            for r in range(r_min, r_max + 1):
                x = spacing * (q + 0.5 * r)
                y = spacing * (np.sqrt(3) / 2 * r)
                points.append([x, y])
        return np.array(points)

    def generate_initializations():
        starts = []
        
        # 1. Hexagonal lattice: pick 16 closest to center from a larger grid
        for radius in [3, 4, 5]:
            hex_pts = generate_hex_grid(radius)
            hex_pts -= np.mean(hex_pts, axis=0)
            norm = np.max(np.linalg.norm(hex_pts, axis=1))
            if norm > 1e-10:
                hex_pts = hex_pts / norm
            dists_center = np.linalg.norm(hex_pts, axis=1)
            sorted_idx = np.argsort(dists_center)
            starts.append(hex_pts[sorted_idx[:16]])

        # 2. Circular arrangement
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        circle = np.column_stack((np.cos(angles), np.sin(angles)))
        starts.append(circle)

        # 3. Random starts with fixed seeds for reproducibility
        for seed in [1, 7, 13, 42, 99, 123, 456, 789]:
            np.random.seed(seed)
            pts = np.random.uniform(-1, 1, (n, 2))
            norm = np.max(np.linalg.norm(pts, axis=1))
            if norm > 1e-10:
                pts = pts / norm
            starts.append(pts)

        # 4. Square grid
        x = np.linspace(-1, 1, 4)
        y = np.linspace(-1, 1, 4)
        xx, yy = np.meshgrid(x, y)
        grid = np.column_stack((xx.ravel(), yy.ravel()))
        starts.append(grid)

        # 5. Two concentric circles (8 points each)
        angles1 = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        angles2 = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
        r1, r2 = 0.5, 1.0
        pts1 = np.column_stack((r1 * np.cos(angles1), r1 * np.sin(angles1)))
        pts2 = np.column_stack((r2 * np.cos(angles2), r2 * np.sin(angles2)))
        starts.append(np.vstack([pts1, pts2]))

        # 6. Triangular lattice (equilateral)
        tri_pts = []
        for i in range(4):
            for j in range(4):
                x = i + 0.5 * (j % 2)
                y = j * np.sqrt(3) / 2
                tri_pts.append([x, y])
        tri_pts = np.array(tri_pts)
        tri_pts -= np.mean(tri_pts, axis=0)
        norm = np.max(np.linalg.norm(tri_pts, axis=1))
        if norm > 1e-10:
            tri_pts = tri_pts / norm
        starts.append(tri_pts)

        return starts

    def simulated_annealing(points, max_iters=5000):
        """
        Perform Simulated Annealing on the given points.
        Objective: Maximize d_min / d_max.
        We normalize d_max to 1 by scaling points after each move, so we maximize d_min.
        """
        current_points = points.copy()
        current_score = evaluate(current_points)
        best_points = current_points.copy()
        best_score = current_score

        # Cooling schedule
        T_start = 0.5
        T_end = 0.001
        alpha = (T_end / T_start) ** (1.0 / max_iters)

        T = T_start
        step_size = 0.1

        for i in range(max_iters):
            if time_exceeded():
                break

            # Generate neighbor: perturb one random point
            new_points = current_points.copy()
            idx = np.random.randint(n)
            delta = np.random.normal(0, step_size, 2)
            new_points[idx] += delta

            # Normalize to keep d_max = 1 (approx)
            new_dists = pdist(new_points, 'euclidean')
            new_d_max = np.max(new_dists)
            if new_d_max > 1e-12:
                new_points = new_points / new_d_max
            
            new_score = evaluate(new_points)
            
            # Acceptance probability
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
            else:
                delta_e = current_score - new_score
                if delta_e < 1e-12:
                    accept = True
                else:
                    accept = np.random.random() < np.exp(-delta_e / T)
                
                if accept:
                    current_points = new_points
                    current_score = new_score

            T *= alpha
            # Reduce step size slightly
            step_size *= 0.9995

        return best_points, best_score

    best_points = None
    best_score = -1.0

    starts = generate_initializations()

    for start in starts:
        if time_exceeded():
            break
        
        # Run SA from this start
        pts, score = simulated_annealing(start, max_iters=10000)
        
        if score > best_score:
            best_score = score
            best_points = pts

    # Final polish: If we have time, run a tighter SA or local refinement on the best
    if not time_exceeded() and best_points is not None:
        # Run another round of SA with smaller steps and lower T
        def final_polish(points):
            current_points = points.copy()
            current_score = evaluate(current_points)
            best_points_local = current_points.copy()
            best_score_local = current_score

            T_start = 0.01
            T_end = 0.0001
            max_iters = 5000
            alpha = (T_end / T_start) ** (1.0 / max_iters)
            T = T_start
            step_size = 0.01

            for i in range(max_iters):
                if time_exceeded():
                    break

                new_points = current_points.copy()
                idx = np.random.randint(n)
                delta = np.random.normal(0, step_size, 2)
                new_points[idx] += delta

                new_dists = pdist(new_points, 'euclidean')
                new_d_max = np.max(new_dists)
                if new_d_max > 1e-12:
                    new_points = new_points / new_d_max
                
                new_score = evaluate(new_points)
                
                if new_score > current_score:
                    current_points = new_points
                    current_score = new_score
                    if current_score > best_score_local:
                        best_score_local = current_score
                        best_points_local = current_points.copy()
                else:
                    delta_e = current_score - new_score
                    if delta_e < 1e-12:
                        accept = True
                    else:
                        accept = np.random.random() < np.exp(-delta_e / T)
                    
                    if accept:
                        current_points = new_points
                        current_score = new_score

                T *= alpha
                step_size *= 0.999

            return best_points_local, best_score_local

        final_pts, final_score = final_polish(best_points)
        if final_score > best_score:
            best_score = final_score
            best_points = final_pts

    # Fallback
    if best_points is None:
        np.random.seed(0)
        best_points = np.random.randn(n, 2)

    return best_points

# EVOLVE-BLOCK-END

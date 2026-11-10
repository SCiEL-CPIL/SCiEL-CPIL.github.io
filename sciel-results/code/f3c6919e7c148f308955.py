import numpy as np
import time
from scipy.spatial.distance import pdist
import random

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Maximizes the ratio (d_min / d_max)^2 for 14 points in 3D.
    
    Strategy:
    1. Use Simulated Annealing to explore the space of 14 points in 3D.
       - Objective: Maximize (d_min / d_max)^2.
       - Moves: Random Gaussian perturbations to one or a few points.
       - Acceptance: Metropolis criterion with cooling schedule.
       - Constraints: Keep points within a bounded box (e.g., [-2, 2]^3) to prevent drift.
    2. Use a diverse set of initial configurations.
    3. Polish the best result found with a simple local search (hill climbing) to fine-tune.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 600  # 30s margin
    
    def true_score(points):
        """Calculate the actual metric: (d_min / d_max)^2"""
        if points.shape != (n, d):
            return 0.0
        if not np.all(np.isfinite(points)):
            return 0.0
        
        # Compute pairwise squared distances
        dists_sq = pdist(points, metric='sqeuclidean')
        
        if len(dists_sq) == 0:
            return 0.0
            
        d_min_sq = np.min(dists_sq)
        d_max_sq = np.max(dists_sq)
        
        if d_max_sq < 1e-12:
            return 0.0
            
        ratio_sq = d_min_sq / d_max_sq
        return ratio_sq

    def generate_initial_configs():
        configs = []
        
        # 1. Random points in a cube
        for _ in range(5):
            pts = np.random.uniform(-1, 1, (n, d))
            configs.append(pts)
            
        # 2. Icosahedron + 2 random points
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
        p1 = np.random.randn(1, 3)
        p1 /= np.linalg.norm(p1)
        p2 = np.random.randn(1, 3)
        p2 /= np.linalg.norm(p2)
        configs.append(np.vstack([ico_verts, p1, p2]))
        
        # 3. Fibonacci Sphere (approximation for uniform distribution)
        pts = []
        for i in range(n):
            z = 1 - (i / (n - 1)) * 2
            radius_z = np.sqrt(1 - z**2)
            theta = np.pi * (1 + 5) * i # Golden angle
            x = np.cos(theta) * radius_z
            y = np.sin(theta) * radius_z
            pts.append([x, y, z])
        configs.append(np.array(pts))
        
        # 4. Random sphere points
        for _ in range(5):
            pts = np.random.randn(n, d)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            configs.append(pts)
            
        return configs

    best_score = 0.0
    best_points = None
    
    initial_configs = generate_initial_configs()
    
    for i, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 5:
            break
            
        try:
            # Simulated Annealing
            current_pts = init_pts.copy()
            current_score = true_score(current_pts)
            
            best_pts_sa = current_pts.copy()
            best_score_sa = current_score
            
            # Cooling schedule
            T_start = 0.1
            T_end = 1e-6
            n_iterations = 5000
            alpha = (T_end / T_start) ** (1.0 / n_iterations)
            
            T = T_start
            step_size = 0.1
            
            for it in range(n_iterations):
                if time.time() > deadline - 10:
                    break
                    
                # Generate a move: perturb 1-3 random points
                n_perturb = np.random.randint(1, 4)
                indices = np.random.choice(n, n_perturb, replace=False)
                
                new_pts = current_pts.copy()
                for idx in indices:
                    # Gaussian perturbation
                    perturbation = np.random.randn(d) * step_size
                    new_pts[idx] += perturbation
                    # Keep within bounds
                    new_pts[idx] = np.clip(new_pts[idx], -2, 2)
                    
                new_score = true_score(new_pts)
                
                # Acceptance criterion
                if new_score > current_score:
                    current_pts = new_pts
                    current_score = new_score
                    if current_score > best_score_sa:
                        best_score_sa = current_score
                        best_pts_sa = current_pts.copy()
                else:
                    # Accept with probability exp((new_score - current_score) / T)
                    # Note: scores are in [0, 1], so differences are small.
                    # We scale the temperature or the score difference.
                    # Let's use a scaled difference to make T meaningful.
                    delta_score = (new_score - current_score) * 100.0
                    prob = np.exp(delta_score / T)
                    if np.random.rand() < prob:
                        current_pts = new_pts
                        current_score = new_score
                        
                T *= alpha
                # Adaptive step size: decrease slightly over time
                step_size = max(0.01, 0.1 * (1 - it / n_iterations))
                
            # Polish with simple local search
            # Try small random moves and accept only if better
            polish_steps = 500
            local_step = 0.01
            for _ in range(polish_steps):
                if time.time() > deadline - 5:
                    break
                idx = np.random.randint(n)
                new_pts = best_pts_sa.copy()
                new_pts[idx] += np.random.randn(d) * local_step
                new_pts[idx] = np.clip(new_pts[idx], -2, 2)
                new_score = true_score(new_pts)
                if new_score > best_score_sa:
                    best_pts_sa = new_pts
                    best_score_sa = new_score
                    
            if best_score_sa > best_score:
                best_score = best_score_sa
                best_points = best_pts_sa.copy()
                
        except Exception as e:
            continue

    # Fallback if all failed
    if best_points is None:
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
        p1 = np.array([0, 0, 1])
        p2 = np.array([0, 0, -1])
        best_points = np.vstack([ico_verts, p1, p2])
        best_score = true_score(best_points)

    # Final validation
    if not np.all(np.isfinite(best_points)):
        phi = (1 + np.sqrt(5)) / 2
        ico_verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
        best_points = np.vstack([ico_verts, np.array([0,0,0]), np.array([0,0,1])])

    return best_points

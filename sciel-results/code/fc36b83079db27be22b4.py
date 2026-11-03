import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    deadline = 145.0
    
    def hex_init(n_target):
        best_points = None
        min_diff = float('inf')
        for mult in np.linspace(0.8, 1.2, 10):
            a = 1.0 / np.sqrt(n_target) * 1.5 * mult
            b = a * np.sqrt(3) / 2.0
            points = []
            k = int(1.0 / a) + 5
            for j in range(-k, k + 1):
                y = j * b
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * a
                    if x*x + y*y <= 0.99:
                        points.append((x, y))
            if len(points) < n_target:
                continue
            points = np.array(points)
            dists = np.linalg.norm(points, axis=1)
            indices = np.argsort(dists)
            points = points[indices[:n_target]]
            diff = abs(len(points) - n_target)
            if diff < min_diff:
                min_diff = diff
                best_points = points
        if best_points is None:
            phi = np.pi * (3.0 - np.sqrt(5.0))
            i = np.arange(n_target)
            r = 0.95 * np.sqrt(i / n_target)
            theta = i * phi
            best_points = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        return best_points

    positions = hex_init(N)
    norms = np.linalg.norm(positions, axis=1)
    positions = positions / norms[:, np.newaxis] * 0.99
    
    def compute_r(pos):
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist) / 2.0
        norms = np.linalg.norm(pos, axis=1)
        min_b = np.min(1.0 - norms)
        return min(min_pair, min_b)
        
    best_positions = positions.copy()
    best_r = compute_r(positions)
    
    # Phase 1: Annealing
    T = 0.01
    T_min = 1e-6
    alpha = 0.999
    lr = 0.01
    max_steps = 5000
    
    for step in range(max_steps):
        if time.time() > deadline - 10:
            break
        pos = positions
        n = len(pos)
        forces = np.zeros_like(pos)
        
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        dist_safe = np.maximum(dist, 1e-6)
        inv_dist2 = 1.0 / (dist_safe ** 2)
        np.fill_diagonal(inv_dist2, 0)
        
        weighted_diff = diff * inv_dist2[:, :, np.newaxis]
        pair_forces = np.sum(weighted_diff, axis=1)
        
        norms = np.linalg.norm(pos, axis=1)
        norms_safe = np.maximum(norms, 1e-6)
        boundary_forces = -pos / norms_safe[:, np.newaxis]
        
        lambda_b = 0.5
        forces = pair_forces + lambda_b * boundary_forces
        
        noise = np.random.randn(n, 2) * T
        positions = pos + lr * (forces + noise)
        
        norms = np.linalg.norm(positions, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            positions[mask] = positions[mask] / norms[mask, np.newaxis]
            
        current_r = compute_r(positions)
        if current_r > best_r:
            best_r = current_r
            best_positions = positions.copy()
            
        T *= alpha
        if T < T_min:
            break
            
    # Phase 2: Gradient Polish
    # Use the best positions as start
    positions = best_positions.copy()
    lr = 0.001
    polish_steps = 2000
    
    for step in range(polish_steps):
        if time.time() > deadline - 2:
            break
        pos = positions
        n = len(pos)
        
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        
        norms = np.linalg.norm(pos, axis=1)
        boundary_dist = 1.0 - norms
        min_boundary_dist = np.min(boundary_dist)
        
        current_r = min(min_pair_dist / 2.0, min_boundary_dist)
        
        if current_r > best_r:
            best_r = current_r
            best_positions = pos.copy()
            
        # Gradient of min
        forces = np.zeros_like(pos)
        tol = 1e-5
        
        if min_pair_dist / 2.0 < min_boundary_dist + tol:
            # Active pairs
            mask = dist < (min_pair_dist + 2.0 * tol)
            np.fill_diagonal(mask, False)
            weights = np.zeros_like(dist)
            weights[mask] = 1.0 / dist[mask]
            sum_weights = np.sum(weights, axis=1)
            weighted_pos = weights @ pos
            pair_forces = pos * sum_weights[:, np.newaxis] - weighted_pos
            forces += pair_forces
            
        if min_boundary_dist < min_pair_dist / 2.0 + tol:
            mask_b = boundary_dist < (min_boundary_dist + tol)
            for i in np.where(mask_b)[0]:
                if norms[i] > 1e-10:
                    forces[i] += -pos[i] / norms[i]
                    
        positions = pos + lr * forces
        norms = np.linalg.norm(positions, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            positions[mask] = positions[mask] / norms[mask, np.newaxis]
            
        lr *= 0.999
        if lr < 1e-8:
            break
            
    return best_positions

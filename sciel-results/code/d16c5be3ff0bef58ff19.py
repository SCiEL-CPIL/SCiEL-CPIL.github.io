import numpy as np
import time

N = 150

def construct():
        start_time = time.time()
        deadline = start_time + 110.0 # 10s margin

        # 1. Initialization: Fibonacci Spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # Normalize to be sure
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        def energy_and_grad(pts):
            # Pairwise distances
            # D[i,j] = ||pts[i] - pts[j]||
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_sq = np.sum(diff * diff, axis=2)
            # Avoid division by zero for i==j
            np.fill_diagonal(dist_sq, 1.0)
            dist = np.sqrt(dist_sq)
            
            # Energy: sum_{i<j} log(1/d) = - sum_{i<j} log(d)
            # Using upper triangle
            iu = np.triu_indices(N, k=1)
            log_d = np.log(dist[iu])
            energy = -np.sum(log_d)
            
            # Gradient
            # grad_i = - sum_{j!=i} (x_i - x_j) / ||x_i - x_j||^2
            # Note: dist_sq[i,j] is ||x_i - x_j||^2
            # The term (x_i - x_j) / dist_sq[i,j]
            # We can compute the sum over j for each i.
            # inv_dist_sq = 1 / dist_sq
            # np.fill_diagonal(inv_dist_sq, 0)
            # grad = - np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
            
            inv_dist_sq = 1.0 / dist_sq
            np.fill_diagonal(inv_dist_sq, 0.0)
            
            # diff is (N, N, 3) where diff[i,j] = x_i - x_j
            # We want sum_j diff[i,j] * inv_dist_sq[i,j]
            grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
            
            return energy, grad

        # Initial energy
        e, g = energy_and_grad(points)
        
        # Gradient Descent with projection
        eta = 1e-4
        max_iter = 2000
        
        for k in range(max_iter):
            if time.time() > deadline:
                break
                
            e, g = energy_and_grad(points)
            
            # Project gradient to tangent plane
            # g_proj = g - (g . x) x
            dot = np.sum(g * points, axis=1, keepdims=True)
            g_proj = g - dot * points
            
            # Update
            points_new = points - eta * g_proj
            points_new /= np.linalg.norm(points_new, axis=1, keepdims=True)
            
            e_new, _ = energy_and_grad(points_new)
            
            if e_new < e:
                points = points_new
                e = e_new
                # Maybe increase eta slightly? Or keep constant.
                # For logarithmic potential, the force is strong at close distances.
                # Small eta is safer.
            else:
                eta *= 0.5
                if eta < 1e-10:
                    break

        return points

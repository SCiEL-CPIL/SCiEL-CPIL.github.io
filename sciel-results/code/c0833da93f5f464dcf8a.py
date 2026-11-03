import numpy as np
from scipy.optimize import minimize

N = 50

def construct():
    # We will try several initialization strategies and optimize each.
    # The optimization minimizes an energy function that penalizes close spheres and wall violations.
    
    def energy_and_grad(x):
        """
        x: (N, 3) array of centers.
        Returns energy and gradient.
        Energy = sum_{i<j} 1/d^2 + sum_{i,k} 1/w^2
        """
        centers = x
        # Pairwise distances
        # d^2_{ij}
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        d2 = np.sum(diff**2, axis=2)
        np.fill_diagonal(d2, 1e-9) # Avoid division by zero
        
        # Force between spheres: grad(1/d^2) wrt c_i = -2 * (c_i - c_j) / d^4
        # We want to minimize energy, so we move opposite to gradient?
        # Energy increases when d decreases.
        # dE/dc_i = sum_{j != i} -2 * (c_i - c_j) / d_{ij}^4
        # Actually, let's just compute the force vector F_i = sum_{j != i} (c_i - c_j) / d_{ij}^3
        # This is the repulsive force. We want to move along F to increase distances.
        # So we minimize -dot(F, delta)? No, just move c += alpha * F.
        # Let's define a cost function that is smooth.
        
        # Wall distances
        # w_ik = 0.5 - |c_ik|
        # Penalty if w_ik is small.
        # Let's use a soft penalty: sum 1/(0.5 - |c|)^2 is unstable near walls.
        # Better: Use the actual radius calculation in a loop or use a smooth approx.
        
        # For gradient descent on positions, let's use the repulsive force directly.
        # This is not a standard gradient descent on a scalar function, but a physical simulation.
        
        # Calculate repulsive forces
        # d3 = d2 ** 1.5
        # We need d, not d2, for the 1/d^2 force magnitude?
        # Force magnitude ~ 1/d^2. Direction ~ (c_i - c_j)/d.
        # So vector force ~ (c_i - c_j) / d^3.
        
        d = np.sqrt(d2)
        np.fill_diagonal(d, 1e-9)
        
        # Force matrix: F_{ij} = (c_i - c_j) / d_{ij}^3
        # c_i is repeated along axis 1, c_j along axis 0.
        # diff is c_i - c_j.
        # d3 = d ** 3
        d3 = d ** 3
        np.fill_diagonal(d3, 1e-9)
        
        # Force on i from j is diff[i,j] / d3[i,j]
        # Sum over j for each i.
        force_spheres = np.sum(diff / d3[:, :, np.newaxis], axis=1)
        
        # Wall forces
        # If c_ik > 0, wall is at 0.5. dist = 0.5 - c_ik. Force pushes left (negative).
        # If c_ik < 0, wall is at -0.5. dist = c_ik + 0.5. Force pushes right (positive).
        # Force magnitude ~ 1/dist^2.
        # Let's use a large constant K_wall to keep them inside.
        K_wall = 100.0
        force_walls = np.zeros_like(centers)
        
        for k in range(3):
            c = centers[:, k]
            dist_pos = 0.5 - c
            dist_neg = c + 0.5
            
            # Only apply if close to wall
            # Force = -K / dist^2 * sign(c) ?
            # If c > 0, force is negative. dist = 0.5 - c.
            # Force = -K / (0.5 - c)^2
            # If c < 0, force is positive. dist = c + 0.5.
            # Force = K / (c + 0.5)^2
            
            # Clip distances to avoid singularity
            eps = 1e-6
            dist_pos = np.maximum(dist_pos, eps)
            dist_neg = np.maximum(dist_neg, eps)
            
            f_pos = -K_wall / (dist_pos ** 2)
            f_neg = K_wall / (dist_neg ** 2)
            
            # Apply only when c is on the respective side
            # Actually, the formula works if we just sum them? No.
            # If c=0.4, dist_pos=0.1, f_pos = -100/0.01 = -10000. dist_neg=0.9, f_neg=100/0.81=123. Net -9877. Correct.
            # If c=-0.4, dist_pos=0.9, f_pos=-123. dist_neg=0.1, f_neg=10000. Net 9877. Correct.
            
            force_walls[:, k] = f_pos + f_neg

        total_force = force_spheres + force_walls
        
        # Calculate current radius for tracking
        d_min = np.min(d[np.triu_indices(N, k=1)])
        w_min = np.min(np.abs(0.5 - np.abs(centers)))
        r_current = min(d_min / 2.0, w_min)
        
        # We don't return gradient for standard minimize, we use this force for manual update
        return total_force, r_current

    def optimize(config, iterations=2000, lr_start=0.01, lr_end=0.001):
        centers = config.copy()
        best_r = 0.0
        best_centers = centers.copy()
        
        for i in range(iterations):
            # Decay learning rate
            lr = lr_start * (lr_end / lr_start) ** (i / iterations)
            
            force, r_current = energy_and_grad(centers)
            
            if r_current > best_r:
                best_r = r_current
                best_centers = centers.copy()
                
            # Update
            # Normalize force to prevent explosion
            norm_force = np.linalg.norm(force)
            if norm_force > 1e-6:
                force = force / norm_force * np.linalg.norm(force) # Keep magnitude? Or just scale?
                # Actually, force magnitude varies. Let's just use a fixed small step scaled by decay.
                # The force calculation above has 1/d^3 which can be huge.
                # Let's cap the step size.
                step = lr * force
                # Clip step
                max_step = 0.01
                step_norm = np.linalg.norm(step, axis=1)
                mask = step_norm > max_step
                if np.any(mask):
                    step[mask] = step[mask] * (max_step / step_norm[mask])[:, np.newaxis]
                
                centers += step
            
            # Clip to cube
            centers = np.clip(centers, -0.5, 0.5)
            
        # Final check
        force, r_current = energy_and_grad(centers)
        if r_current > best_r:
            best_r = r_current
            best_centers = centers.copy()
            
        return best_centers, best_r

    def generate_initializations():
        inits = []
        
        # 1. Jittered Grid
        m = 4 # 4x4x4 = 64
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        grid = grid[:N]
        jitter = np.random.normal(0, 0.01, grid.shape)
        grid_jittered = np.clip(grid + jitter, -0.49, 0.49)
        inits.append(grid_jittered)
        
        # 2. Random
        inits.append(np.random.uniform(-0.4, 0.4, (N, 3)))
        
        # 3. Spectral (Eigenvectors of random SPD matrix)
        # Generate random symmetric matrix
        A = np.random.randn(N, N)
        A = (A + A.T) / 2
        # Make it positive definite
        A = A @ A.T + np.eye(N) * 0.1
        # Eigenvectors
        eigvecs = np.linalg.eig(A)[1]
        # Take first N? We have N eigenvectors.
        # Project to 3D? The directive says "representing sphere centers as top N eigenvectors".
        # This implies the coordinates are the components of the eigenvectors?
        # If we take the first 3 components of each eigenvector, they are in R^3.
        # Let's normalize and scale them to fit in the cube.
        spectral_centers = eigvecs[:, :3]
        # Center and scale
        spectral_centers -= np.mean(spectral_centers, axis=0)
        spectral_centers /= np.max(np.abs(spectral_centers))
        spectral_centers *= 0.4 # Fit in [-0.4, 0.4]
        inits.append(spectral_centers)
        
        # 4. Another random
        inits.append(np.random.uniform(-0.4, 0.4, (N, 3)))
        
        return inits

    inits = generate_initializations()
    
    best_overall_r = 0.0
    best_overall_centers = None
    
    # Time management
    import time
    start_time = time.time()
    time_limit = 100.0 # Leave buffer for 120s
    
    for init in inits:
        if time.time() - start_time > time_limit:
            break
        centers, r = optimize(init, iterations=3000)
        if r > best_overall_r:
            best_overall_r = r
            best_overall_centers = centers
            
    # If we have time, do a final polish on the best one
    if time.time() - start_time < time_limit and best_overall_centers is not None:
        centers, r = optimize(best_overall_centers, iterations=2000, lr_start=0.005, lr_end=0.0001)
        if r > best_overall_r:
            best_overall_r = r
            best_overall_centers = centers
            
    return best_overall_centers

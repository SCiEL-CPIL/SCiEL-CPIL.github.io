# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Pack N equal circles inside the unit circle to maximize the common radius.
    
    Uses a physics-based relaxation approach (Langevin dynamics) with binary search on radius.
    1. Define an energy function based on overlap and boundary violations.
    2. Binary search for the maximum radius r such that a valid configuration exists.
    3. For each candidate r, minimize the energy using Langevin dynamics with cooling.
    4. Return the best configuration found.
    """
    N = 50
    
    def energy_and_grad(c, r):
        """
        Compute the penalty energy and its gradient.
        Energy = sum_{i<j} max(0, 2r - d_ij)^2 + sum_i max(0, |c_i| - (1-r))^2
        """
        c = c.reshape(N, 2)
        n = N
        energy = 0.0
        grad = np.zeros_like(c)
        
        # Pairwise interactions
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Overlap amount: overlap_ij = max(0, 2r - d_ij)
        overlap = np.maximum(0, 2 * r - dists)
        
        # Energy contribution: sum(overlap^2) / 2
        pair_energy = 0.5 * np.sum(overlap ** 2)
        energy += pair_energy
        
        # Gradient calculation
        safe_dists = np.where(dists > 1e-10, dists, 1.0)
        np.fill_diagonal(overlap, 0)
        
        # weights[i,j] = -2 * overlap[i,j] / safe_dists[i,j]
        weights = -2.0 * overlap / safe_dists
        np.fill_diagonal(weights, 0)
        
        grad_pairs = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
        grad += grad_pairs
        
        # Boundary interactions
        radii = np.linalg.norm(c, axis=1)
        bound_violation = np.maximum(0, radii - (1.0 - r))
        
        # E_bound = sum (bound_violation)^2
        energy += np.sum(bound_violation ** 2)
        
        # Gradient
        safe_radii = np.where(radii > 1e-10, radii, 1.0)
        grad_bound = 2.0 * bound_violation[:, np.newaxis] * c / safe_radii[:, np.newaxis]
        grad += grad_bound
        
        return energy, grad

    def relax(c_init, r, steps=2000, T_start=0.01, T_end=0.0001, dt=0.01, gamma=1.0):
        """
        Perform Langevin dynamics relaxation to minimize energy for fixed r.
        Returns (c_final, energy_final).
        """
        c = c_init.copy()
        v = np.zeros_like(c)
        
        # Cooling schedule
        temperatures = np.geomspace(T_start, T_end, steps)
        
        for i in range(steps):
            T = temperatures[i]
            # Compute force (negative gradient)
            _, grad = energy_and_grad(c, r)
            force = -grad
            
            # Langevin update
            # v_new = (v + force * dt) * exp(-gamma * dt) + sqrt(2 * k_B * T * dt) * noise
            # Simplified: v = (v + force * dt) * (1 - gamma * dt) + noise
            # More stable: Velocity Verlet-like with friction
            damping = np.exp(-gamma * dt)
            v = (v + force * dt) * damping
            noise = np.random.normal(0, np.sqrt(2 * T * dt), c.shape)
            v += noise
            
            c = c + v * dt
            
            # Keep inside container roughly (hard constraint for stability)
            norms = np.linalg.norm(c, axis=1)
            max_rad = 1.0 - r + 0.01 # small buffer
            mask = norms > max_rad
            if np.any(mask):
                c[mask] = c[mask] / norms[mask, np.newaxis] * max_rad
                v[mask] = 0.0 # damp velocity if hitting boundary
                
        _, e_final = energy_and_grad(c, r)
        return c, e_final

    def solve_for_radius(r, c_init, trials=3):
        """
        Try to find a low-energy configuration for radius r.
        Returns (c_opt, min_energy).
        """
        best_c = c_init
        best_e = np.inf
        
        for _ in range(trials):
            # Add some noise to initial config for diversity in trials
            c_noisy = c_init + np.random.normal(0, 0.01, c_init.shape)
            # Ensure inside
            norms = np.linalg.norm(c_noisy, axis=1)
            max_rad = 1.0 - r
            mask = norms > max_rad
            c_noisy[mask] = c_noisy[mask] / norms[mask, np.newaxis] * max_rad
            
            c_opt, e_opt = relax(c_noisy, r, steps=1000, T_start=0.05, T_end=0.001, dt=0.02, gamma=2.0)
            
            if e_opt < best_e:
                best_e = e_opt
                best_c = c_opt
                
        return best_c, best_e

    def generate_initial_configurations():
        """
        Generate a list of diverse initial configurations.
        """
        configs = []
        
        # 1. Hexagonal lattice (good for dense packing)
        scale = 0.2
        points = []
        for j in range(-10, 11):
            for i in range(-10, 11):
                x = i * scale + 0.5 * scale * (j % 2)
                y = j * scale * np.sqrt(3) / 2
                if np.sqrt(x * x + y * y) < 1.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) >= N:
            dists = np.linalg.norm(points, axis=1)
            indices = np.argsort(dists)[:N]
            configs.append(points[indices])
            
        # 2. Concentric rings
        rings = [1, 6, 12, 18, 13]
        radii = [0.0, 0.25, 0.5, 0.75, 0.9]
        points = []
        for k, (n_pts, rad) in enumerate(zip(rings, radii)):
            for i in range(n_pts):
                angle = 2 * np.pi * i / n_pts + (np.pi / n_pts if k % 2 == 0 else 0)
                points.append([rad * np.cos(angle), rad * np.sin(angle)])
        configs.append(np.array(points[:N]))
        
        # 3. Random
        c_rand = np.random.uniform(-0.8, 0.8, (N, 2))
        norms = np.linalg.norm(c_rand, axis=1)
        mask = norms > 0.8
        c_rand[mask] /= norms[mask, np.newaxis] * 0.8
        configs.append(c_rand)
        
        return configs

    np.random.seed(42)
    
    r_low = 0.1
    r_high = 0.13
    
    best_centers = None
    best_r = r_low
    
    configs = generate_initial_configurations()
    
    # Binary search
    for _ in range(15):
        r_mid = (r_low + r_high) / 2
        
        feasible = False
        best_c_for_r = None
        best_e_for_r = np.inf
        
        # Try a few initial configs
        for c_init in configs:
            c_opt, e_opt = solve_for_radius(r_mid, c_init, trials=2)
            if e_opt < best_e_for_r:
                best_e_for_r = e_opt
                best_c_for_r = c_opt
                
            if e_opt < 1e-4:
                feasible = True
                break # Found a good one, stop early for this r
        
        if feasible:
            r_low = r_mid
            best_r = r_mid
            best_centers = best_c_for_r
        else:
            r_high = r_mid
            
    # Final polish at best_r
    if best_centers is not None:
        # Use more steps for final polish
        c_final, e_final = relax(best_centers, best_r, steps=5000, T_start=0.01, T_end=0.00001, dt=0.01, gamma=5.0)
        if e_final < 1e-4:
            best_centers = c_final
        else:
            # Try slightly smaller radius
            r_retry = best_r - 0.0005
            c_retry, e_retry = relax(best_centers, r_retry, steps=2000, T_start=0.01, T_end=0.0001, dt=0.01, gamma=5.0)
            if e_retry < 1e-4:
                best_centers = c_retry
                best_r = r_retry

    if best_centers is None:
        best_centers = configs[0]
        
    return best_centers
# EVOLVE-BLOCK-END

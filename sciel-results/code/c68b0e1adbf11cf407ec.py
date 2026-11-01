# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Pack N equal circles inside the unit circle to maximize the common radius.
    
    Uses a physics-based relaxation (symplectic integrator) with binary search on radius.
    The circles are treated as particles with repulsive springs for overlaps and 
    boundary constraints. An adaptive cooling schedule helps escape local minima.
    """
    N = 50
    
    def compute_forces_and_energy(c, r):
        """
        Compute forces and energy for the configuration c at radius r.
        """
        c = c.reshape(N, 2)
        forces = np.zeros_like(c)
        energy = 0.0
        
        # Pairwise repulsion
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Overlap: max(0, 2r - d)
        overlap = np.maximum(0, 2 * r - dists)
        
        # Energy: 0.5 * sum(overlap^2)
        energy += 0.5 * np.sum(overlap ** 2)
        
        # Force: -dE/dc = sum (overlap / d) * diff
        # F_ij = (overlap_ij / d_ij) * (c_i - c_j)
        # Note: overlap is symmetric.
        safe_dists = np.where(dists > 1e-10, dists, 1.0)
        np.fill_diagonal(overlap, 0)
        
        weights = overlap / safe_dists
        np.fill_diagonal(weights, 0)
        
        # Sum of forces from all j on i
        # F_i = sum_j (overlap_ij / d_ij) * (c_i - c_j)
        # This is equivalent to: sum_j weights_ij * diff_ij
        forces += np.sum(weights[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary force
        radii = np.linalg.norm(c, axis=1)
        bound_violation = np.maximum(0, radii - (1.0 - r))
        
        energy += np.sum(bound_violation ** 2)
        
        # Force: -dE/dc = -2 * bound_violation * (c / |c|)
        safe_radii = np.where(radii > 1e-10, radii, 1.0)
        forces -= 2.0 * bound_violation[:, np.newaxis] * c / safe_radii[:, np.newaxis]
        
        return forces, energy

    def relax(c_init, r, steps=500, temp_init=0.01, temp_final=0.0001):
        """
        Relax the configuration using a symplectic-like integrator with annealing.
        """
        c = c_init.copy()
        v = np.zeros_like(c)
        
        # Adaptive cooling
        for step in range(steps):
            alpha = temp_init * (temp_final / temp_init) ** (step / steps)
            
            forces, energy = compute_forces_and_energy(c, r)
            
            # Update velocity (Verlet-like)
            # v = v * damping + force * dt
            # We use a fixed dt scaled by temperature to act as annealing
            dt = 0.01
            v = v * 0.9 + forces * dt
            c += v * dt
            
            # Keep inside a larger box to prevent flying off, but boundary force handles it
            # Optional: hard clip to unit circle * 1.5
            norms = np.linalg.norm(c, axis=1)
            mask = norms > 1.5
            if np.any(mask):
                c[mask] /= norms[mask, np.newaxis] * 1.5
                v[mask] = 0.0
                
        forces, energy = compute_forces_and_energy(c, r)
        return c, energy

    def generate_initial_configurations():
        """
        Generate diverse initial configurations.
        """
        configs = []
        
        # 1. Jittered Rings
        rings = [1, 6, 12, 18, 13]
        radii = [0.0, 0.25, 0.5, 0.75, 0.9]
        for jitter in [0.0, 0.05, 0.1]:
            points = []
            for k, (n_pts, rad) in enumerate(zip(rings, radii)):
                for i in range(n_pts):
                    angle = 2 * np.pi * i / n_pts + (np.pi / n_pts if k % 2 == 0 else 0)
                    if jitter > 0:
                        angle += np.random.uniform(-jitter, jitter)
                        rad_j = rad + np.random.uniform(-jitter, jitter)
                    else:
                        rad_j = rad
                    points.append([rad_j * np.cos(angle), rad_j * np.sin(angle)])
            configs.append(np.array(points[:N]))
            
        # 2. Random
        for _ in range(5):
            c_rand = np.random.uniform(-0.9, 0.9, (N, 2))
            norms = np.linalg.norm(c_rand, axis=1)
            mask = norms > 0.9
            c_rand[mask] /= norms[mask, np.newaxis] * 0.9
            configs.append(c_rand)
            
        return configs

    np.random.seed(42)
    
    r_low = 0.12
    r_high = 0.128 # Closer to known best 0.1258
    
    best_centers = None
    best_r = r_low
    
    configs = generate_initial_configurations()
    
    # Binary search
    for _ in range(15):
        r_mid = (r_low + r_high) / 2
        
        feasible = False
        best_c_for_r = None
        best_e_for_r = np.inf
        
        for c_init in configs:
            c_opt, e_opt = relax(c_init, r_mid, steps=300, temp_init=0.05, temp_final=0.001)
            if e_opt < best_e_for_r:
                best_e_for_r = e_opt
                best_c_for_r = c_opt
            
            if e_opt < 1e-6:
                feasible = True
        
        if feasible:
            r_low = r_mid
            best_r = r_mid
            best_centers = best_c_for_r
        else:
            r_high = r_mid
            
    # Final polish
    if best_centers is not None:
        c_final, e_final = relax(best_centers, best_r, steps=500, temp_init=0.01, temp_final=0.0001)
        if e_final < 1e-6:
            best_centers = c_final
        else:
            # Try slightly smaller
            r_retry = best_r - 0.001
            c_retry, e_retry = relax(best_centers, r_retry, steps=500, temp_init=0.01, temp_final=0.0001)
            if e_retry < 1e-6:
                best_centers = c_retry
                best_r = r_retry

    if best_centers is None:
        best_centers = configs[0]
        
    return best_centers
# EVOLVE-BLOCK-END

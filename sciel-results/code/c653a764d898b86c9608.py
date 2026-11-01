# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Pack N equal circles inside the unit circle to maximize the common radius.
    
    Uses a physics-based relaxation (Velocity Verlet with damping) to find 
    a valid configuration for a given radius, combined with binary search 
    to maximize the radius.
    """
    N = 50
    
    def compute_forces(c, r):
        """
        Compute forces on particles due to overlaps and boundary violations.
        """
        c = c.reshape(N, 2)
        forces = np.zeros_like(c)
        
        # Pairwise repulsive forces
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, 1.0) # Avoid division by zero
        
        # We want to push apart if dist < 2r
        # Force magnitude proportional to overlap
        overlap = np.maximum(0, 2 * r - dists)
        
        # Spring constant k
        k = 100.0
        # Force vector direction is along diff
        # F_ij = k * overlap_ij * (c_i - c_j) / dist_ij
        # Note: diff[i,j] = c_i - c_j
        # If overlap > 0, we want to push i and j apart.
        # Force on i from j is in direction (c_i - c_j).
        
        # weights[i,j] = k * overlap[i,j] / dists[i,j]
        weights = np.zeros_like(dists)
        mask = overlap > 0
        weights[mask] = k * overlap[mask] / dists[mask]
        np.fill_diagonal(weights, 0)
        
        # Sum forces
        # F_i = sum_j weights[i,j] * (c_i - c_j)
        forces += np.sum(weights[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary forces
        radii = np.linalg.norm(c, axis=1)
        bound_violation = np.maximum(0, radii - (1.0 - r))
        
        # Force pushing inward if outside (1-r)
        # Direction is -c_i / |c_i|
        safe_radii = np.where(radii > 1e-10, radii, 1.0)
        forces -= k * bound_violation[:, np.newaxis] * c / safe_radii[:, np.newaxis]
        
        return forces

    def relaxation(c_init, r, steps=5000, dt_init=0.001, damping=0.9):
        """
        Relax the configuration using Velocity Verlet with damping.
        """
        c = c_init.copy()
        v = np.zeros_like(c)
        f = compute_forces(c, r)
        
        dt = dt_init
        max_dt = 0.01
        min_dt = 1e-5
        
        for i in range(steps):
            # Adaptive step size based on max force
            max_f = np.max(np.linalg.norm(f, axis=1))
            if max_f > 0:
                # Limit acceleration to prevent explosion
                # a = f/m, m=1
                # dt ~ sqrt(dx / a)
                # Heuristic: dt = min(max_dt, 0.1 / max_f)
                dt = min(max_dt, max(min_dt, 0.05 / (max_f + 1e-6)))
            
            # Velocity Verlet
            # v_half = v + 0.5 * dt * f
            v_half = v + 0.5 * dt * f
            
            # c_new = c + dt * v_half
            c_new = c + dt * v_half
            
            # f_new = compute_forces(c_new, r)
            f_new = compute_forces(c_new, r)
            
            # v_new = (v_half + 0.5 * dt * f_new) * damping
            v = (v_half + 0.5 * dt * f_new) * damping
            
            c = c_new
            f = f_new
            
            # Check convergence
            if max_f < 1e-4:
                break
                
        return c

    def energy(c, r):
        """
        Compute the total energy (overlap + boundary).
        """
        c = c.reshape(N, 2)
        energy = 0.0
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, 1.0)
        
        overlap = np.maximum(0, 2 * r - dists)
        energy += 0.5 * 100.0 * np.sum(overlap ** 2)
        
        radii = np.linalg.norm(c, axis=1)
        bound_violation = np.maximum(0, radii - (1.0 - r))
        energy += 0.5 * 100.0 * np.sum(bound_violation ** 2)
        
        return energy

    def solve_for_radius(r, c_init):
        """
        Relax the configuration for a fixed radius r.
        """
        c_relaxed = relaxation(c_init, r, steps=2000, dt_init=0.005, damping=0.95)
        
        # Polish with L-BFGS-B
        def obj(x):
            return energy(x, r)
            
        # Compute gradient for L-BFGS-B
        def jac(x):
            c = x.reshape(N, 2)
            # Numerical gradient or analytical?
            # Let's use numerical for simplicity and robustness in the polish step, 
            # or reuse the force logic as gradient (since F = -dE/dx for spring potential)
            # Note: compute_forces returns the force, which is -grad(E).
            # So grad(E) = -force.
            f = compute_forces(c, r)
            return -f.flatten()
            
        try:
            res = minimize(obj, c_relaxed.flatten(), jac=jac, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-8})
            return res.x.reshape(N, 2), res.fun
        except Exception:
            return c_relaxed, energy(c_relaxed, r)

    def generate_initial_configurations():
        """
        Generate a list of diverse initial configurations.
        """
        configs = []
        
        # 1. Hexagonal lattice
        for scale in [0.12, 0.14, 0.16]:
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
            else:
                while len(points) < N:
                    angle = np.random.uniform(0, 2 * np.pi)
                    radius = np.random.uniform(0, 0.9)
                    points.append([radius * np.cos(angle), radius * np.sin(angle)])
                configs.append(np.array(points))
        
        # 2. Concentric rings
        rings = [1, 6, 12, 18, 13]
        radii = [0.0, 0.25, 0.5, 0.75, 0.9]
        points = []
        for k, (n_pts, rad) in enumerate(zip(rings, radii)):
            for i in range(n_pts):
                angle = 2 * np.pi * i / n_pts + (np.pi / n_pts if k % 2 == 0 else 0)
                points.append([rad * np.cos(angle), rad * np.sin(angle)])
        configs.append(np.array(points[:N]))
        
        # 3. Random restarts
        for _ in range(10):
            c_rand = np.random.uniform(-0.9, 0.9, (N, 2))
            norms = np.linalg.norm(c_rand, axis=1)
            mask = norms > 0.9
            c_rand[mask] /= norms[mask, np.newaxis] * 0.9
            configs.append(c_rand)
            
        return configs

    # Binary search for the best radius
    np.random.seed(42)
    
    r_low = 0.1
    r_high = 0.13
    
    best_centers = None
    best_r = r_low
    
    configs = generate_initial_configurations()
    
    # Binary search
    for _ in range(25):
        r_mid = (r_low + r_high) / 2
        
        feasible = False
        best_c_for_r = None
        best_e_for_r = np.inf
        
        for c_init in configs:
            c_opt, e_opt = solve_for_radius(r_mid, c_init)
            if e_opt < best_e_for_r:
                best_e_for_r = e_opt
                best_c_for_r = c_opt
            
            if e_opt < 1e-4:
                feasible = True
        
        if feasible:
            r_low = r_mid
            best_r = r_mid
            best_centers = best_c_for_r
        else:
            r_high = r_mid
            
    # Final polish at best_r
    if best_centers is not None:
        c_final, e_final = solve_for_radius(best_r, best_centers)
        if e_final < 1e-4:
            best_centers = c_final
        else:
            # Try slightly smaller radius
            r_retry = best_r - 0.001
            c_retry, e_retry = solve_for_radius(r_retry, best_centers)
            if e_retry < 1e-4:
                best_centers = c_retry
                best_r = r_retry

    if best_centers is None:
        best_centers = configs[0]
        
    return best_centers
# EVOLVE-BLOCK-END

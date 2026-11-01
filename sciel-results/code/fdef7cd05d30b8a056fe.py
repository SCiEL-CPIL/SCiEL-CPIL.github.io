# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Pack N equal circles inside the unit circle to maximize the common radius.
    
    Uses a hybrid Adam + L-BFGS-B optimization approach with binary search on radius.
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

    def adam_optimize(c_init, r, lr=0.005, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=300):
        """
        Adam optimizer for the penalty energy.
        """
        c = c_init.copy().flatten()
        m = np.zeros_like(c)
        v = np.zeros_like(c)
        
        for t in range(1, max_iter + 1):
            e, g = energy_and_grad(c, r)
            grad = g.flatten()
            
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            c -= lr * m_hat / (np.sqrt(v_hat) + eps)
            
            if np.linalg.norm(grad) < 1e-7:
                break
                
        e_final, _ = energy_and_grad(c, r)
        return c.reshape(N, 2), e_final

    def solve_for_radius(r, c_init):
        """
        Minimize energy for a fixed radius r starting from c_init.
        Uses Adam followed by L-BFGS-B for polishing.
        Returns (c_opt, min_energy).
        """
        # Stage 1: Adam
        c_adam, e_adam = adam_optimize(c_init, r)
        
        # Stage 2: L-BFGS-B Polish
        def obj(x):
            e, _ = energy_and_grad(x, r)
            return e
            
        def jac(x):
            _, g = energy_and_grad(x, r)
            return g.flatten()
            
        try:
            res = minimize(obj, c_adam.flatten(), jac=jac, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
            return res.x.reshape(N, 2), res.fun
        except Exception:
            return c_adam, e_adam

    def generate_initial_configurations():
        """
        Generate a list of diverse initial configurations.
        """
        configs = []
        
        # 1. Fermat Spiral
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        points = []
        for i in range(N):
            r = np.sqrt(i / N) * 0.95 # Scale down slightly to stay inside
            theta = i * golden_angle
            points.append([r * np.cos(theta), r * np.sin(theta)])
        configs.append(np.array(points))

        # 2. Hexagonal lattice
        for scale in [0.15, 0.18]:
            points = []
            for j in range(-10, 11):
                for i in range(-10, 11):
                    x = i * scale + 0.5 * scale * (j % 2)
                    y = j * scale * np.sqrt(3) / 2
                    if np.sqrt(x * x + y * y) < 0.95:
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
        
        # 3. Concentric rings
        rings = [1, 6, 12, 18, 13]
        radii = [0.0, 0.25, 0.5, 0.75, 0.9]
        points = []
        for k, (n_pts, rad) in enumerate(zip(rings, radii)):
            for i in range(n_pts):
                angle = 2 * np.pi * i / n_pts + (np.pi / n_pts if k % 2 == 0 else 0)
                points.append([rad * np.cos(angle), rad * np.sin(angle)])
        configs.append(np.array(points[:N]))
        
        # 4. Random restarts
        for _ in range(3):
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
    
    for _ in range(20):
        r_mid = (r_low + r_high) / 2
        
        feasible = False
        best_c_for_r = None
        best_e_for_r = np.inf
        
        for c_init in configs:
            c_opt, e_opt = solve_for_radius(r_mid, c_init)
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
            
    # Final polish at best_r
    if best_centers is not None:
        c_final, e_final = solve_for_radius(best_r, best_centers)
        if e_final < 1e-6:
            best_centers = c_final
        else:
            r_retry = best_r - 0.001
            c_retry, e_retry = solve_for_radius(r_retry, best_centers)
            if e_retry < 1e-6:
                best_centers = c_retry
                best_r = r_retry

    if best_centers is None:
        best_centers = configs[0]
        
    return best_centers
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Pack N equal circles inside the unit circle to maximize the common radius.
    
    Uses a penalty-based energy minimization approach:
    1. Define an energy function based on overlap and boundary violations.
    2. Iteratively increase the target radius r.
    3. For each r, minimize the energy with respect to center positions using L-BFGS-B.
    4. If a valid configuration is found (low energy), accept the new radius.
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
        # d_ij = |c_i - c_j|
        # Penalty if d_ij < 2r
        # We use squared hinge loss: max(0, 2r - d)^2
        
        # Compute pairwise differences
        # To save memory, we can do it in chunks or use broadcasting if N is small enough.
        # N=50 is small, broadcasting is fine.
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Overlap amount: overlap_ij = max(0, 2r - d_ij)
        overlap = np.maximum(0, 2 * r - dists)
        
        # Energy contribution: sum(overlap^2)
        # Note: each pair counted twice in sum, so divide by 2? 
        # The sum over i<j is half the sum over all i!=j.
        # Let's just sum all i!=j and divide by 2.
        pair_energy = 0.5 * np.sum(overlap ** 2)
        energy += pair_energy
        
        # Gradient with respect to c_i
        # d/dc_i sum_{j!=i} 0.5 * max(0, 2r - d_ij)^2
        # Let f(d) = 0.5 * max(0, 2r - d)^2
        # df/dd = max(0, 2r - d) * (-(2r - d)) = -overlap * (2r - d)
        # But wait, d = |c_i - c_j|. d c_i / d d = (c_i - c_j) / d
        # So grad_i += sum_j df/dd * (c_i - c_j) / d
        # df/dd = -overlap * (2r - d) ? No.
        # Let g(d) = max(0, 2r - d). E = 0.5 * g(d)^2.
        # dE/dd = g(d) * g'(d).
        # g'(d) = -1 if d < 2r, else 0.
        # So dE/dd = -overlap if d < 2r, else 0.
        # grad_i = sum_j dE/dd * (c_i - c_j) / d_ij
        # Note: The factor 0.5 in energy definition accounts for double counting.
        # If we sum over all i!=j, we get 2 * (sum i<j).
        # Let's define E_pair = sum_{i<j} (max(0, 2r - d_ij))^2.
        # Then grad_i = sum_{j!=i} max(0, 2r - d_ij) * (c_i - c_j) / d_ij * (-1) * (1/2 * 2? No)
        
        # Let's restart the gradient derivation carefully.
        # E = sum_{i<j} h(d_ij)^2 where h(d) = max(0, 2r-d).
        # dE/dc_i = sum_{j!=i, j>i} d(h(d_ij)^2)/dc_i + sum_{j<i} d(h(d_ji)^2)/dc_i
        # Since d_ij = d_ji, and h is same:
        # dE/dc_i = sum_{j!=i} d(h(d_ij)^2)/dc_i
        # Let's just compute for all j != i.
        
        # h(d) = max(0, 2r - d)
        # d(h^2)/dd = 2 h dh/dd = 2 h (-1) = -2h (if d<2r)
        # dd/dc_i = (c_i - c_j) / d
        # d(h^2)/dc_i = -2 h (c_i - c_j) / d
        
        # So grad_i = sum_{j!=i} -2 * overlap_ij * (c_i - c_j) / d_ij
        
        # We have `overlap` matrix and `dists` matrix.
        # overlap[i,j] is max(0, 2r - d_ij).
        # term_ij = -2 * overlap[i,j] * diff[i,j] / dists[i,j]
        
        safe_dists = np.where(dists > 1e-10, dists, 1.0)
        # Set diagonal to 0 to avoid self-interaction
        np.fill_diagonal(overlap, 0)
        
        # Calculate force/gradient contribution from pairs
        # F_i = sum_j -2 * overlap_ij * (c_i - c_j) / d_ij
        # Note: diff[i,j] = c_i - c_j
        # We need to sum over j for each i.
        
        # Vectorized calculation:
        # weights[i,j] = -2 * overlap[i,j] / safe_dists[i,j]
        # grad_i = sum_j weights[i,j] * diff[i,j]
        
        weights = -2.0 * overlap / safe_dists
        # Zero out diagonal just in case
        np.fill_diagonal(weights, 0)
        
        grad_pairs = np.sum(weights[:, :, np.newaxis] * diff, axis=1)
        grad += grad_pairs
        
        # Boundary interactions
        # Penalty if |c_i| > 1 - r
        # h_bound(d) = max(0, d - (1-r))
        # E_bound = sum_i h_bound(|c_i|)^2
        
        radii = np.linalg.norm(c, axis=1)
        bound_violation = np.maximum(0, radii - (1.0 - r))
        
        # E_bound = sum (bound_violation)^2
        energy += np.sum(bound_violation ** 2)
        
        # Gradient
        # dE/dc_i = 2 * bound_violation_i * d(|c_i|)/dc_i
        # d(|c_i|)/dc_i = c_i / |c_i|
        # So grad_i += 2 * bound_violation_i * c_i / radii_i
        
        safe_radii = np.where(radii > 1e-10, radii, 1.0)
        grad_bound = 2.0 * bound_violation[:, np.newaxis] * c / safe_radii[:, np.newaxis]
        # Only apply where violation exists (bound_violation > 0), but the formula handles it since bound_violation is 0 otherwise.
        grad += grad_bound
        
        return energy, grad

    def solve_for_radius(r, c_init):
        """
        Minimize energy for a fixed radius r starting from c_init.
        Returns (c_opt, min_energy).
        """
        c_flat = c_init.flatten()
        
        def obj(x):
            e, _ = energy_and_grad(x, r)
            return e
            
        def jac(x):
            _, g = energy_and_grad(x, r)
            return g.flatten()
            
        try:
            res = minimize(obj, c_flat, jac=jac, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            return res.x.reshape(N, 2), res.fun
        except Exception:
            return c_init, np.inf

    def make_initial_hexagonal(n, scale=0.15):
        points = []
        for j in range(-10, 11):
            for i in range(-10, 11):
                x = i * scale + 0.5 * scale * (j % 2)
                y = j * scale * np.sqrt(3) / 2
                if np.sqrt(x * x + y * y) < 1.0:
                    points.append([x, y])
        points = np.array(points)
        if len(points) > n:
            dists = np.linalg.norm(points, axis=1)
            indices = np.argsort(dists)[:n]
            points = points[indices]
        elif len(points) < n:
            while len(points) < n:
                angle = np.random.uniform(0, 2 * np.pi)
                radius = np.random.uniform(0, 0.9)
                points = np.vstack([points, [radius * np.cos(angle), radius * np.sin(angle)]])
        return points

    # Try multiple initial configurations
    best_centers = None
    best_r = -np.inf
    
    np.random.seed(42)
    
    initial_configs = []
    # Hexagonal
    for scale in [0.12, 0.14, 0.16]:
        initial_configs.append(make_initial_hexagonal(N, scale=scale))
    
    # Random
    for _ in range(3):
        c_rand = np.random.uniform(-0.9, 0.9, (N, 2))
        # Ensure inside
        norms = np.linalg.norm(c_rand, axis=1)
        mask = norms > 0.9
        c_rand[mask] /= norms[mask, np.newaxis] * 0.9
        initial_configs.append(c_rand)
        
    for c_init in initial_configs:
        # Start with a small radius
        r = 0.05
        c_current = c_init.copy()
        
        # Iterate increasing r
        for _ in range(100): # Max 100 steps of 0.005 -> up to r=0.5
            c_opt, min_energy = solve_for_radius(r, c_current)
            
            if min_energy < 1e-6:
                # Feasible, increase r
                c_current = c_opt
                r += 0.002
            else:
                # Not feasible, try smaller step or break
                # If we can't even satisfy current r, break
                if r > 0.05: # If we've increased before, we are stuck
                    break
                else:
                    break
            
            if r > 0.15: # Heuristic upper bound for this task
                break
                
        # After the loop, r is the largest feasible radius we found + 0.002
        # So the actual feasible r is r - 0.002
        # Let's verify and maybe do one more polish at r - 0.002
        
        r_final = r - 0.002
        if r_final > best_r:
            # Polish at r_final
            c_polished, e_polished = solve_for_radius(r_final, c_current)
            if e_polished < 1e-6:
                best_r = r_final
                best_centers = c_polished
            else:
                # If polish failed, use previous
                best_r = r_final - 0.002 # Be conservative
                best_centers = c_current

    if best_centers is None:
        # Fallback
        best_centers = make_initial_hexagonal(N, 0.15)
        
    return best_centers
# EVOLVE-BLOCK-END

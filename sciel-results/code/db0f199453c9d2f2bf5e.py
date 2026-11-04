import numpy as np
import time

N = 100

def construct():
    # Start with a grid initialization
    # 5x5x4 grid
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.375, 0.375, nz)
    
    # Create the grid
    centers = np.array([(x, y, z) for x in xs for y in ys for z in zs])
    
    def get_score(c):
        # Calculate min pair distance
        # Using broadcasting for N=100 is fast enough
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        # Calculate min boundary distance
        # Distance to closest face
        # Faces are at +/- 0.5
        # dist to +0.5 is 0.5 - c
        # dist to -0.5 is c - (-0.5) = c + 0.5
        # min dist for sphere i is min(0.5 - c_i, c_i + 0.5)
        # Actually, simpler: 0.5 - max(abs(c_i))
        max_abs_coord = np.max(np.abs(c), axis=1)
        min_boundary = 0.5 - np.max(max_abs_coord)
        
        return min(min_pair / 2.0, min_boundary)

    best_centers = centers.copy()
    best_score = get_score(best_centers)
    
    # Simulated Annealing Parameters
    # We have ~120s. Let's aim for 100,000 - 500,000 iterations.
    # Each iteration involves O(N^2) distance calc.
    # N=100, N^2=10,000. 
    # 500,000 * 10,000 = 5e9 ops. Might be too slow in Python/NumPy overhead.
    # Let's estimate: 100x100 distance calc in NumPy is ~1ms?
    # 500,000 * 1ms = 500s. Too slow.
    # We need to optimize the score calculation or reduce iterations.
    
    # Optimization: 
    # We can cache the distance matrix? No, it changes every step.
    # We can use a faster move strategy?
    # Maybe reduce N^2 cost by only checking neighbors? Hard with random moves.
    
    # Let's try a different approach: Gradient Descent with adaptive step size.
    # It converges faster locally.
    
    # Let's combine: SA for a while to escape bad local minima, then GD for polish.
    # Or just a very well-tuned GD with multiple restarts from perturbed grid.
    
    # Let's try a "Repulsion-based" gradient descent which is similar to the current one but more robust.
    # The current one increases r. A better way is to minimize a potential energy.
    
    def potential_energy(c, r_target):
        # Energy = sum of overlaps + sum of boundary violations
        # Overlap: max(0, 2*r_target - d_ij)^2
        # Boundary: max(0, r_target - (0.5 - |c_ik|))^2
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Overlap penalty
        overlaps = np.maximum(0, 2 * r_target - dists)
        energy = np.sum(overlaps**2)
        
        # Boundary penalty
        # dist to boundary for each coord
        # For each sphere i and coord k:
        # dist_pos = 0.5 - c[i,k]
        # dist_neg = c[i,k] + 0.5
        # min_dist = min(dist_pos, dist_neg) = 0.5 - abs(c[i,k])
        # violation = max(0, r_target - min_dist)
        
        min_dists = 0.5 - np.abs(c)
        violations = np.maximum(0, r_target - min_dists)
        energy += np.sum(violations**2)
        
        return energy

    # Strategy:
    # 1. Start with grid.
    # 2. Run Gradient Descent to minimize energy for a fixed r.
    # 3. If energy is 0, increase r.
    # 4. If energy > 0, decrease r or break.
    # This is similar to the current one but using a smooth potential (squared) instead of linear forces, which is often better for GD.
    
    # Actually, let's use a **L-BFGS-B** or similar optimizer from SciPy?
    # "SciPy are available".
    # We can define the objective function as -min_radius? No, min_radius is non-differentiable.
    # We can use the potential energy approach and increase r.
    
    # Let's try to use `scipy.optimize.minimize` with L-BFGS-B.
    # We need to flatten the centers.
    
    from scipy.optimize import minimize
    
    def objective_flat(x_flat, r_target):
        c = x_flat.reshape(N, 3)
        return potential_energy(c, r_target)
    
    def grad_flat(x_flat, r_target):
        c = x_flat.reshape(N, 3)
        # Gradient of potential energy
        # Overlap part:
        # d/dc_i sum_j max(0, 2r - d_ij)^2
        # Let u_ij = 2r - d_ij. If u_ij > 0, term is u_ij^2.
        # d/dc_i u_ij^2 = 2 * u_ij * d/dc_i (2r - d_ij)
        # d/dc_i d_ij = - (c_i - c_j) / d_ij
        # So d/dc_i term = 2 * (2r - d_ij) * (c_i - c_j) / d_ij
        
        grad = np.zeros_like(c)
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        mask = dists < 2 * r_target
        # For each i, sum over j where mask[i,j] is True
        # term_ij = 2 * (2r - dists[i,j]) * diff[i,j] / dists[i,j]
        
        # We can compute this vectorized
        # factor[i,j] = 2 * (2r - dists[i,j]) / dists[i,j] if mask else 0
        factor = np.zeros_like(dists)
        factor[mask] = 2 * (2 * r_target - dists[mask]) / dists[mask]
        
        # grad[i] = sum_j factor[i,j] * diff[i,j]
        # Note: diff[i,j] = c[i] - c[j]
        # grad[i] = sum_j factor[i,j] * (c[i] - c[j])
        
        # This can be done as:
        # grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
        
        grad_overlap = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
        
        # Boundary part:
        # E_bound = sum_{i,k} max(0, r - (0.5 - |c_ik|))^2
        # Let v_ik = r - (0.5 - |c_ik|). If v_ik > 0, term is v_ik^2.
        # d/dc_ik v_ik^2 = 2 * v_ik * d/dc_ik (r - 0.5 + |c_ik|)
        # d/dc_ik |c_ik| = sign(c_ik)
        # So d/dc_ik term = 2 * (r - 0.5 + |c_ik|) * sign(c_ik)
        
        min_dists = 0.5 - np.abs(c)
        violations = r_target - min_dists
        mask_bound = violations > 0
        
        # factor_bound[i,k] = 2 * violations[i,k] * sign(c[i,k]) if mask else 0
        # Note: sign(0) is 0, which is fine.
        
        grad_bound = np.zeros_like(c)
        if np.any(mask_bound):
             # 2 * (r - 0.5 + abs(c)) * sign(c)
             # Note: r - min_dists = r - (0.5 - abs(c)) = r - 0.5 + abs(c)
             # So factor is 2 * (r_target - 0.5 + np.abs(c)) * np.sign(c)
             # But only where mask_bound is True.
             
             temp = 2 * (r_target - 0.5 + np.abs(c)) * np.sign(c)
             grad_bound[mask_bound] = temp[mask_bound]
        
        total_grad = grad_overlap + grad_bound
        return total_grad.flatten()

    # Algorithm:
    # 1. Init grid.
    # 2. r = 0.1
    # 3. Loop:
    #    a. Use L-BFGS-B to minimize energy for current r.
    #    b. If min energy is close to 0, increase r.
    #    c. Else, break.
    
    x0 = centers.flatten()
    
    # Bounds: [-0.5, 0.5] for each coord
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    r = 0.10
    dr = 0.001
    
    # Time limit check
    start_time = time.time()
    timeout = 110.0 # Leave 10s buffer
    
    while time.time() - start_time < timeout:
        # Optimize for current r
        res = minimize(objective_flat, x0, args=(r,), method='L-BFGS-B', jac=grad_flat, bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10})
        
        if res.success or res.fun < 1e-4:
            # Check if we actually achieved the radius
            # The energy is sum of squares. If it's small, it's good.
            # But we need to ensure the *actual* min radius is at least r.
            # The optimization minimizes the penalty. If penalty is 0, constraints are met.
            
            # Update x0 to the result
            x0 = res.x
            
            # Increase r
            r += dr
        else:
            # If optimization fails or energy is high, we might be stuck.
            # Try a smaller dr? Or break?
            # If res.fun is not small, we didn't satisfy constraints for r.
            # We should probably reduce r or break.
            # Since we are increasing r, if we fail, we are at the limit.
            # Let's try to recover by reducing r slightly and re-optimizing?
            # Or just break.
            
            # Check if the current solution is valid for a slightly smaller r?
            # Let's just break to be safe, as we are likely at a local optimum.
            break
            
    # Final polish with Hill Climbing / SA on the best found
    best_centers = x0.reshape(N, 3)
    best_score = get_score(best_centers)
    
    # Quick SA to polish
    T = 0.001
    for i in range(50000):
        if time.time() - start_time > timeout:
            break
        idx = np.random.randint(N)
        delta = np.random.normal(0, 0.005, 3)
        new_c = best_centers.copy()
        new_c[idx] = np.clip(best_centers[idx] + delta, -0.5, 0.5)
        new_score = get_score(new_c)
        
        if new_score > best_score:
            best_centers = new_c
            best_score = new_score
        else:
            # Accept with probability
            if np.random.rand() < np.exp((new_score - best_score) / T):
                best_centers = new_c
                best_score = new_score
        
        T *= 0.999 # Cool down
        
    return best_centers

import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 100
HALF = 0.5

def compute_min_dist(pos):
    """Compute the minimum radius for a given configuration."""
    if not np.all(np.isfinite(pos)):
        return 0.0
    
    # Vectorized distance calculation
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(dists_sq)
    np.fill_diagonal(dists, np.inf)
    
    min_pair_val = np.min(dists)
    
    d_walls = HALF - np.abs(pos)
    min_wall_val = np.min(d_walls)
    
    r_pair = min_pair_val / 2.0
    r_wall = min_wall_val
    
    return min(r_pair, r_wall)

def energy_and_gradient(x, R):
    """
    Calculates the repulsion energy and gradient for a target radius R.
    Energy is the sum of squared violations of the distance constraints.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # 1. Pairwise terms
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    d_ij = dists[i_idx, j_idx]
    mask_pairs = d_ij < 2.0 * R
    
    if np.any(mask_pairs):
        v_ij = 2.0 * R - d_ij[mask_pairs]
        d_valid = d_ij[mask_pairs]
        i_valid = i_idx[mask_pairs]
        j_valid = j_idx[mask_pairs]
        
        energy += np.sum(v_ij**2)
        
        # Gradient of 0.5 * (2R - d)^2 w.r.t centers
        # dE/dx_i = sum_j -2(2R - d_ij) * (x_i - x_j)/d_ij
        factors = 2.0 * v_ij / d_valid
        vecs = centers[i_valid] - centers[j_valid]
        forces = factors[:, np.newaxis] * vecs
        
        np.add.at(grad, i_valid, -forces)
        np.add.at(grad, j_valid, forces)

    # 2. Wall terms
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    
    mask_wall = wall_dists < R
    if np.any(mask_wall):
        violations_wall = R - wall_dists[mask_wall]
        i_valid_wall = np.where(mask_wall)[0]
        
        energy += np.sum(violations_wall**2)
        
        for idx in i_valid_wall:
            v = violations_wall[np.where(i_valid_wall == idx)[0][0]]
            max_val = max_abs[idx]
            max_k = np.argmax(abs_centers[idx])
            
            # Gradient of (R - (0.5 - |x_k|))^2
            # d/dx_k (R - 0.5 + |x_k|)^2 = 2(R - 0.5 + |x_k|) * sign(x_k)
            grad[idx, max_k] += 2.0 * v * np.sign(centers[idx, max_k])
            
    return energy, grad.flatten()

def _minimize_energy(init_centers, target_R, max_iter=500, ftol=1e-14, gtol=1e-12):
    """
    Minimizes the energy for a given target radius.
    """
    def objective_and_grad(x):
        val, g = energy_and_gradient(x, target_R)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad,
            init_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False}
        )
        new_centers = res.x.reshape(N, 3)
        final_energy, _ = energy_and_gradient(new_centers, target_R)
        return new_centers, final_energy, res.success
    except Exception:
        return init_centers, np.inf, False

def _force_directed_relaxation(pos, target_r, max_iter=200, time_limit=10.0):
    """
    Structural approach: Model centers as vertices of a 3D Delaunay triangulation.
    Optimize a discrete set of edge lengths using a force-directed graph relaxation.
    This leverages the topological stability of the contact graph.
    """
    start_time = time.time()
    pos = pos.copy()
    
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(pos)
            simplices = tri.simplices
        except Exception:
            # Fallback to simple repulsion if triangulation fails
            break
            
        # Build contact graph from Delaunay simplices
        # Edges are shared between simplices
        edge_set = set()
        for simplex in simplices:
            for i in range(4):
                for j in range(i+1, 4):
                    edge = (simplex[i], simplex[j])
                    edge_set.add((min(edge), max(edge)))
        
        edges = list(edge_set)
        
        # Compute forces
        forces = np.zeros_like(pos)
        
        # Target distance is 2 * target_r
        target_d = 2.0 * target_r
        
        for i, j in edges:
            vec = pos[i] - pos[j]
            d = np.linalg.norm(vec)
            if d < 1e-10:
                continue
            # Spring force: k * (d - target_d)
            # We want to push apart if d < target_d, pull together if d > target_d
            # But we also have wall constraints.
            # Simple spring force to enforce uniform edge lengths
            k = 1.0
            f = k * (d - target_d) / d * vec
            forces[i] += f
            forces[j] -= f
        
        # Wall forces
        for i in range(N):
            for k in range(3):
                x = pos[i, k]
                if x > HALF - target_r:
                    forces[i, k] -= (x - (HALF - target_r))
                elif x < -HALF + target_r:
                    forces[i, k] += (-HALF + target_r - x)
        
        # Update positions
        # Simple gradient descent with adaptive step size
        step = 0.01
        pos_new = pos + step * forces
        
        # Keep inside cube
        pos_new = np.clip(pos_new, -HALF + target_r, HALF - target_r)
        
        # Check if we improved
        r_new = compute_min_dist(pos_new)
        if r_new > compute_min_dist(pos):
            pos = pos_new
        else:
            # Reduce step size
            step *= 0.9
            if step < 1e-6:
                break
    
    return pos

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # 1. Initialization: 5x4x5 Grid with symmetry breaking
    nx, ny, nz = 5, 4, 5
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    
    rng = np.random.default_rng(42)
    # Add small noise to break symmetry
    best_pos = grid[:N].copy() + rng.normal(0, 0.01, size=(N, 3))
    best_pos = np.clip(best_pos, -0.49, 0.49)
    
    # Initial optimization to get a good baseline
    # Start with a conservative radius
    initial_r = 0.1
    best_pos, energy, _ = _minimize_energy(best_pos, initial_r, max_iter=200)
    best_r = compute_min_dist(best_pos)
    
    if best_r <= 0:
         best_pos = rng.uniform(-0.45, 0.45, size=(N, 3))
         best_r = compute_min_dist(best_pos)

    # 2. Iterative Radius Increase
    current_r = best_r
    step_size = 0.0005  # Smaller steps for stability
    
    consecutive_failures = 0
    max_failures = 5
    
    rng_search = np.random.default_rng(123)
    
    while time.time() < deadline - 15.0:
        target_r = current_r + step_size
        
        if consecutive_failures >= max_failures:
            step_size *= 0.7
            consecutive_failures = 0
            if step_size < 1e-7:
                break
        
        candidates = []
        candidates.append(best_pos)
        candidates.append(best_pos + rng_search.normal(0, 0.005, size=(N, 3)))
        candidates.append(best_pos + rng_search.normal(0, 0.02, size=(N, 3)))
        
        success = False
        for start_pos in candidates:
            start_pos = np.clip(start_pos, -0.5, 0.5)
            
            # Increase max_iter for higher precision as we get closer to optimum
            max_iter = 300 if current_r < 0.11 else 500
            
            new_pos, energy, opt_success = _minimize_energy(start_pos, target_r, max_iter=max_iter)
            
            if energy < 1e-10:
                actual_r = compute_min_dist(new_pos)
                if actual_r >= target_r - 1e-6:
                    best_pos = new_pos
                    best_r = actual_r
                    current_r = actual_r
                    success = True
                    consecutive_failures = 0
                    break
            else:
                # Even if energy is not perfectly zero, if we improved the radius, we can take it
                # But we must ensure constraints are met for the new radius
                actual_r = compute_min_dist(new_pos)
                if actual_r > current_r:
                    best_pos = new_pos
                    best_r = actual_r
                    current_r = actual_r
                    # Don't mark as full success for step size increase, but accept the position
                    # This allows gradual improvement even if target_r wasn't fully met
                    # However, for the loop logic, 
                    # we treat this as a partial success
                    success = True
                    consecutive_failures = 0
                    break
        
        if not success:
            consecutive_failures += 1
    
    # 3. Final Polish with Force-Directed Relaxation
    if time.time() < deadline - 5.0:
        # Try force-directed relaxation to exploit topological structure
        polished_pos = _force_directed_relaxation(best_pos, best_r, max_iter=100, time_limit=5.0)
        final_r = compute_min_dist(polished_pos)
        if final_r > best_r:
            best_r = final_r
            best_pos = polished_pos
        
        # Try a few hard-min polishes with slightly perturbed starts to escape any flat spots
        for _ in range(3):
            polished_pos = _minimize_energy(best_pos, best_r, max_iter=300)
            final_r = compute_min_dist(polished_pos)
            if final_r > best_r:
                best_r = final_r
                best_pos = polished_pos
            else:
                # Try a tiny perturbation before next polish
                perturbed = best_pos + rng_search.normal(0, 0.001, size=(N, 3))
                perturbed = np.clip(perturbed, -0.5, 0.5)
                best_pos = perturbed
                
    return best_pos

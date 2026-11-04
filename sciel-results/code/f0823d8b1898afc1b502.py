"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """
    Computes the maximum common radius for a given set of centers.
    Centers: (N, 3) array.
    Cube: [-0.5, 0.5]^3.
    """
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
        
    # Wall constraints: distance from center to nearest wall
    max_abs = np.max(np.abs(centers), axis=1)
    wall_dists = 0.5 - max_abs
    min_wall_dist = np.min(wall_dists)
    
    if min_wall_dist < 0:
        return 0.0

    # Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to infinity to exclude self-distance
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # The radius is half the minimum pairwise distance, limited by wall distance
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _energy_and_gradient(x, R):
    """
    Calculates the repulsion energy and gradient for a target radius R.
    Energy is the sum of squared violations of the distance constraints.
    Uses vectorized operations for speed.
    """
    centers = x.reshape(-1, 3)
    n = len(centers)
    
    energy = 0.0
    grad = np.zeros_like(centers)
    
    # 1. Pairwise terms
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-15)
    np.fill_diagonal(dists, np.inf)
    
    # Mask for pairs closer than 2R
    i_idx, j_idx = np.triu_indices(n, k=1)
    d_ij = dists[i_idx, j_idx]
    mask_pairs = d_ij < 2.0 * R
    
    if np.any(mask_pairs):
        v_ij = 2.0 * R - d_ij[mask_pairs]
        d_valid = d_ij[mask_pairs]
        i_valid = i_idx[mask_pairs]
        j_valid = j_idx[mask_pairs]
        
        energy += np.sum(v_ij**2)
        
        factors = 2.0 * v_ij / d_valid
        vecs = centers[i_valid] - centers[j_valid]
        forces = factors[:, np.newaxis] * vecs
        
        np.add.at(grad, i_valid, -forces)
        np.add.at(grad, j_valid, forces)

    # 2. Wall terms - vectorized
    abs_centers = np.abs(centers)
    max_abs = np.max(abs_centers, axis=1)
    wall_dists = 0.5 - max_abs
    
    mask_wall = wall_dists < R
    if np.any(mask_wall):
        violations_wall = R - wall_dists[mask_wall]
        i_valid_wall = np.where(mask_wall)[0]
        
        energy += np.sum(violations_wall**2)
        
        # Vectorized wall gradient
        for idx in i_valid_wall:
            v = violations_wall[list(i_valid_wall).index(idx)]
            max_k = np.argmax(abs_centers[idx])
            grad[idx, max_k] += 2.0 * v * np.sign(centers[idx, max_k])
            
    return energy, grad.flatten()

def _generate_grid(nx, ny, nz):
    """Generate a grid initialization."""
    xs = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
    ys = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
    zs = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
    grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return grid

def _generate_bcc():
    """Generate a BCC lattice initialization for better packing."""
    # BCC: simple cubic lattice + body center
    # For 100 spheres, try a 5x4x4 grid with BCC offset
    positions = []
    # Use a 5x4x4 grid = 80 points, plus 20 body centers
    nx, ny, nz = 5, 4, 4
    spacing = 1.0 / max(nx, ny, nz)
    
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                x = -0.5 + (ix + 0.5) * spacing
                y = -0.5 + (iy + 0.5) * spacing
                z = -0.5 + (iz + 0.5) * spacing
                positions.append([x, y, z])
                # Add body center if we need more points
                if len(positions) < N:
                    bx = x + spacing/2
                    by = y + spacing/2
                    bz = z + spacing/2
                    if abs(bx) < 0.49 and abs(by) < 0.49 and abs(bz) < 0.49:
                        positions.append([bx, by, bz])
    
    positions = np.array(positions[:N])
    return positions

def _local_optimize_violation(centers, R, max_iter=500, ftol=1e-14, gtol=1e-12):
    """Minimize violation energy for target radius R."""
    def objective_and_grad(x):
        val, g = _energy_and_gradient(x, R)
        return val, g

    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad,
            centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False}
        )
        new_centers = res.x.reshape(N, 3)
        final_energy, _ = _energy_and_gradient(new_centers, R)
        return new_centers, final_energy
    except Exception:
        return centers, np.inf

def _contact_graph_refine(centers, R, max_iter=2000, ftol=1e-16, gtol=1e-14):
    """
    Refine using contact graph: identify active constraints and do targeted optimization.
    """
    # First pass: standard optimization
    new_centers, _ = _local_optimize_violation(centers, R, max_iter=max_iter)
    
    # Identify contact pairs (distance close to 2R) and wall contacts
    diff = new_centers[:, np.newaxis, :] - new_centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    d_ij = dists[i_idx, j_idx]
    
    # Contacts: pairs within 1% of 2R
    contact_threshold = 2.0 * R * 1.01
    contact_mask = d_ij < contact_threshold
    
    # Wall contacts
    abs_c = np.abs(new_centers)
    max_abs = np.max(abs_c, axis=1)
    wall_dists = 0.5 - max_abs
    wall_contact_mask = wall_dists < R * 1.01
    
    # Second pass: high-precision polish
    def objective_and_grad(x):
        val, g = _energy_and_gradient(x, R)
        return val, g
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            objective_and_grad,
            new_centers.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol, 'disp': False}
        )
        refined_centers = res.x.reshape(N, 3)
    except Exception:
        refined_centers = new_centers
    
    return refined_centers

def _multi_start_search(centers, current_r, deadline, rng):
    """
    Multi-start search with various initializations and strategies.
    """
    best_pos = centers.copy()
    best_r = current_r
    step_size = 0.0005
    
    consecutive_failures = 0
    max_failures = 5
    
    while time.time() < deadline - 20.0:
        target_r = best_r + step_size
        
        if consecutive_failures >= max_failures:
            step_size *= 0.7
            consecutive_failures = 0
            if step_size < 1e-7:
                break
        
        success = False
        
        # Strategy 1: Direct energy minimization from current best
        max_iter = 400 if best_r < 0.11 else 600
        new_pos, energy = _local_optimize_violation(best_pos, target_r, max_iter=max_iter)
        actual_r = _compute_radius(new_pos)
        if actual_r >= target_r - 1e-6:
            best_pos = new_pos
            best_r = actual_r
            success = True
            consecutive_failures = 0
        
        # Strategy 2: Perturbation + optimization
        if not success and time.time() < deadline - 25.0:
            start_pos = best_pos + rng.normal(0, 0.005, size=(N, 3))
            start_pos = np.clip(start_pos, -0.49, 0.49)
            new_pos, energy = _local_optimize_violation(start_pos, target_r, max_iter=max_iter)
            actual_r = _compute_radius(new_pos)
            if actual_r >= target_r - 1e-6:
                best_pos = new_pos
                best_r = actual_r
                success = True
                consecutive_failures = 0
            elif actual_r > best_r + 1e-6:
                best_pos = new_pos
                best_r = actual_r
                consecutive_failures = 0
        
        # Strategy 3: BCC-inspired restart
        if not success and time.time() < deadline - 30.0 and consecutive_failures % 3 == 0:
            bcc_pos = _generate_bcc()
            # Blend with current best to preserve good structure
            blended = 0.5 * bcc_pos + 0.5 * best_pos
            blended = np.clip(blended, -0.49, 0.49)
            new_pos, energy = _local_optimize_violation(blended, target_r, max_iter=max_iter)
            actual_r = _compute_radius(new_pos)
            if actual_r >= target_r - 1e-6:
                best_pos = new_pos
                best_r = actual_r
                success = True
                consecutive_failures = 0
            elif actual_r > best_r + 1e-6:
                best_pos = new_pos
                best_r = actual_r
                consecutive_failures = 0
        
        # Strategy 4: Simulated annealing escape
        if not success and time.time() < deadline - 35.0:
            T = 0.01 * (0.5 ** (consecutive_failures / max_failures))
            sa_pos = best_pos.copy()
            current_energy, _ = _energy_and_gradient(sa_pos, best_r)
            for _ in range(100):
                perturbation = rng.normal(0, 0.02, size=sa_pos.shape)
                proposed = sa_pos + perturbation
                proposed = np.clip(proposed, -0.49, 0.49)
                proposed_energy, _ = _energy_and_gradient(proposed, best_r)
                delta_E = proposed_energy - current_energy
                if delta_E < 0 or rng.random() < np.exp(-delta_E / max(T, 1e-10)):
                    sa_pos = proposed
                    current_energy = proposed_energy
            
            new_pos, energy = _local_optimize_violation(sa_pos, target_r, max_iter=max_iter)
            actual_r = _compute_radius(new_pos)
            if actual_r >= target_r - 1e-6:
                best_pos = new_pos
                best_r = actual_r
                success = True
                consecutive_failures = 0
            elif actual_r > best_r + 1e-6:
                best_pos = new_pos
                best_r = actual_r
                consecutive_failures = 0
        
        if not success:
            consecutive_failures += 1
            
            # Periodic contact graph refinement
            if time.time() < deadline - 40.0 and consecutive_failures % 4 == 0:
                refined = _contact_graph_refine(best_pos, best_r, max_iter=1000)
                refined_r = _compute_radius(refined)
                if refined_r > best_r:
                    best_pos = refined
                    best_r = refined_r

    return best_pos, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    # Try multiple initializations and keep the best
    initializations = []
    
    # Grid initialization
    grid = _generate_grid(5, 4, 5)
    initializations.append(grid[:N].copy() + rng.normal(0, 0.01, size=(N, 3)))
    
    # BCC initialization
    bcc = _generate_bcc()
    initializations.append(bcc + rng.normal(0, 0.01, size=(N, 3)))
    
    # Random initialization
    random_init = rng.uniform(-0.45, 0.45, size=(N, 3))
    initializations.append(random_init)
    
    # Optimize each initialization and keep the best
    best_pos = None
    best_r = -1
    
    for init in initializations:
        if time.time() > deadline - 30.0:
            break
        
        init = np.clip(init, -0.49, 0.49)
        pos, energy = _local_optimize_violation(init, 0.1, max_iter=300)
        r = _compute_radius(pos)
        
        if r > best_r:
            best_r = r
            best_pos = pos
    
    if best_pos is None:
        best_pos = initializations[0]
        best_r = _compute_radius(best_pos)
    
    # Main search loop
    rng_search = np.random.default_rng(123)
    best_pos, best_r = _multi_start_search(best_pos, best_r, deadline, rng_search)
    
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        for polish_iter in range(3):
            if time.time() > deadline - 5.0:
                break
            
            # Try to push slightly beyond current radius
            target_r = best_r + 5e-6 * (polish_iter + 1)
            max_iter = 1500 + 500 * polish_iter
            
            refined = _contact_graph_refine(best_pos, target_r, max_iter=max_iter)
            actual_r = _compute_radius(refined)
            
            if actual_r > best_r:
                best_pos = refined
                best_r = actual_r
            else:
                # Polish at current radius
                refined = _contact_graph_refine(best_pos, best_r, max_iter=2000)
                actual_r = _compute_radius(refined)
                if actual_r > best_r:
                    best_pos = refined
                    best_r = actual_r
                else:
                    break
    
    # Final validation
    final_pos = np.clip(best_pos, -0.5, 0.5)
    final_r = _compute_radius(final_pos)
    
    return final_pos
# EVOLVE-BLOCK-END

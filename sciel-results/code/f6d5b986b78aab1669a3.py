import numpy as np
import time
import math
from scipy.optimize import minimize

N = 38
DEADLINE = 145.0  # Safety margin for 150s limit
START_TIME = time.time()

def lj_energy(positions):
    """Compute total Lennard-Jones energy: 4 * sum(r^-12 - r^-6) over all pairs."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r2 = float(np.dot(rij, rij))
            if r2 < 1e-12:
                return 1e20
            inv_r2 = 1.0 / r2
            inv_r6 = inv_r2 * inv_r2 * inv_r2
            inv_r12 = inv_r6 * inv_r6
            energy += 4.0 * (inv_r12 - inv_r6)
    return energy

def lj_energy_vectorized(positions):
    """Vectorized LJ energy computation."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy

def check_validity(positions, min_dist=0.5):
    """Check that all atoms are at least min_dist apart."""
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r2 = np.dot(rij, rij)
            if r2 < min_dist * min_dist:
                return False
    return True

_FCC_CACHE = {}

def _get_fcc_shell_structure(n_atoms=38):
    """
    Generate the 38 closest atoms in an FCC lattice.
    Returns base coordinates (unscaled), shell IDs, and unique radii.
    """
    if n_atoms in _FCC_CACHE:
        return _FCC_CACHE[n_atoms]
    
    scale = 1.0 / math.sqrt(2.0)
    max_coord = 10
    points = []
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = i * scale, j * scale, k * scale
                    r2 = x*x + y*y + z*z
                    points.append((x, y, z, r2))
    
    points.sort(key=lambda p: (p[3], abs(p[0]), abs(p[1]), abs(p[2])))
    
    selected = [p[:3] for p in points[:n_atoms]]
    coords = np.array(selected, dtype=float)
    
    radii = np.linalg.norm(coords, axis=1)
    unique_r = np.sort(np.unique(np.round(radii, 4)))
    
    shell_ids = np.zeros(n_atoms, dtype=int)
    for i, r in enumerate(radii):
        for sid, ur in enumerate(unique_r):
            if abs(r - ur) < 0.05:
                shell_ids[i] = sid
                break
    
    _FCC_CACHE[n_atoms] = (coords, shell_ids, unique_r)
    return coords, shell_ids, unique_r

def _make_truncated_octahedron():
    """
    Build an explicit FCC truncated octahedron for 38 atoms.
    Uses the actual FCC lattice points sorted by radius.
    """
    base_coords, shell_ids, unique_r = _get_fcc_shell_structure(N)
    # Scale to LJ equilibrium: nearest neighbor should be 2^(1/6)
    # In our base coords, NN distance is 1.0 (after dividing by sqrt(2))
    # We want NN distance = 2^(1/6), so global scale = 2^(1/6)
    scale = 2.0 ** (1.0/6.0)
    return base_coords * scale

def _lj_energy_and_grad(pos_flat):
    """Compute LJ energy and gradient for full Cartesian coordinates."""
    pos = pos_flat.reshape(N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    
    inv_r14 = inv_r12 * inv_r2
    inv_r8 = inv_r6 * inv_r2
    
    coeff = -24.0 * inv_r14 + 12.0 * inv_r8
    grad_matrix = coeff[:, :, np.newaxis] * diff
    grad_flat = np.sum(grad_matrix, axis=1).flatten()
    
    return energy, grad_flat

def _lj_energy_only(pos_flat):
    """Compute LJ energy only (no gradient) - faster for basin hopping."""
    pos = pos_flat.reshape(N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy

def relax(positions, maxiter=500, gtol=1e-8):
    """Relax using L-BFGS-B with analytic gradient."""
    x0 = positions.flatten()
    try:
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B', 
                          jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def _basin_hop(x0, current_energy, rng, step_size, n_iter, deadline_margin=20, temp_start=0.15, temp_end=0.02):
    """
    Basin hopping with adaptive step size and annealing Metropolis acceptance.
    Returns (best_energy, best_x).
    """
    best_bh_energy = current_energy
    best_bh_x = x0.copy()
    
    # Adaptive step size
    base_step = step_size
    
    for i in range(n_iter):
        if time.time() - START_TIME > DEADLINE - deadline_margin:
            break
        
        # Annealing temperature
        progress = i / max(n_iter - 1, 1)
        temp = temp_start + (temp_end - temp_start) * progress
        
        # Adaptive step size: occasionally increase or decrease
        if i % 8 == 4:
            if rng.uniform() < 0.35:
                step_size = base_step * rng.uniform(1.5, 3.0)
            elif rng.uniform() < 0.3:
                step_size = base_step * rng.uniform(0.3, 0.7)
            else:
                step_size = base_step
        
        noise = rng.normal(0, step_size, x0.shape)
        perturbed = x0 + noise
        
        try:
            result = minimize(_lj_energy_and_grad, perturbed, method='L-BFGS-B', 
                            jac=True,
                            options={'maxiter': 150, 'gtol': 1e-6, 'maxls': 50})
            new_x = result.x
            new_energy = _lj_energy_only(new_x)
            
            if new_energy < current_energy:
                x0 = new_x
                current_energy = new_energy
                if new_energy < best_bh_energy:
                    best_bh_energy = new_energy
                    best_bh_x = new_x.copy()
            else:
                delta = new_energy - current_energy
                if delta > 0:
                    accept = math.exp(-delta / max(temp, 1e-10))
                    if rng.uniform() < accept:
                        x0 = new_x
                        current_energy = new_energy
        except Exception:
            continue
    
    return best_bh_energy, best_bh_x

def _make_icosahedral_structure():
    """
    Generate an icosahedral structure for 38 atoms.
    1 center + 12 inner shell + 25 outer shell
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    
    min_dist = float('inf')
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(vertices[i] - vertices[j])
            if d < min_dist:
                min_dist = d
    
    scale = 1.0 / min_dist
    vertices = vertices * scale
    
    positions = np.zeros((38, 3))
    positions[0] = [0.0, 0.0, 0.0]
    positions[1:13] = vertices
    
    # Outer 25 atoms: place on a larger sphere with good angular distribution
    n_outer = 25
    r_outer = 1.8
    
    for i in range(n_outer):
        z = 1.0 - 2.0 * (i + 0.5) / n_outer
        r = math.sqrt(1.0 - z*z)
        theta = math.pi * (1.0 + math.sqrt(5.0)) * i
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        positions[13 + i] = np.array([x, y, z]) * r_outer
    
    return positions

def _topology_loss_and_grad(positions):
    """
    Compute a smooth topology-forcing loss that penalizes deviation from the
    FCC truncated octahedron shell radii. Returns (loss, gradient).
    
    The loss is based on comparing each atom's radial distance to its expected
    shell radius in the ideal FCC structure.
    """
    n_atoms = N
    base_coords, shell_ids, unique_r = _get_fcc_shell_structure(n_atoms)
    n_shells = len(unique_r)
    
    # Ideal radii for each shell (scaled to LJ equilibrium distance)
    ideal_scale = 2.0 ** (1.0/6.0)
    ideal_radii = unique_r * ideal_scale
    
    # Compute current radial distances
    radii = np.linalg.norm(positions, axis=1)
    
    # For each atom, find its target radius based on its shell ID
    target_radii = np.zeros(n_atoms)
    for i in range(n_atoms):
        s = shell_ids[i]
        if s < len(ideal_radii):
            target_radii[i] = ideal_radii[s]
        else:
            target_radii[i] = ideal_radii[-1]
    
    # Smooth loss: sum of squared deviations from target radii
    # Weight by shell to prioritize inner shells
    shell_weights = np.ones(n_atoms)
    for i in range(n_atoms):
        s = shell_ids[i]
        shell_weights[i] = 1.0 + 0.5 * (n_shells - s)  # Higher weight for inner shells
    
    diff = radii - target_radii
    loss = np.sum(shell_weights * diff ** 2)
    
    # Gradient: d(loss)/d(positions)
    # d(loss)/d(r_i) = 2 * shell_weights[i] * (radii[i] - target_radii[i])
    # d(r_i)/d(positions[i]) = positions[i] / r_i
    grad = np.zeros_like(positions)
    for i in range(n_atoms):
        r = radii[i]
        if r > 1e-8:
            grad[i] = 2.0 * shell_weights[i] * diff[i] * positions[i] / r
    
    return loss, grad

def _topology_forcing_optimize(initial_positions, max_iter=500, gtol=1e-6):
    """
    Optimize positions to match FCC topology using the topology loss.
    This helps navigate toward the FCC truncated octahedron structure.
    """
    x0 = initial_positions.flatten()
    
    def topology_obj_and_grad(x_flat):
        pos = x_flat.reshape(N, 3)
        loss, grad = _topology_loss_and_grad(pos)
        return loss, grad.flatten()
    
    try:
        result = minimize(topology_obj_and_grad, x0, method='L-BFGS-B',
                         jac=True,
                         options={'maxiter': max_iter, 'gtol': gtol, 'maxls': 100})
        return result.x.reshape(N, 3)
    except Exception:
        return initial_positions

def _optimize_fcc_with_shell_breathing():
    """
    Optimize FCC structure with per-shell breathing parameters.
    Returns (positions, energy) or (None, inf) on failure.
    """
    base_coords, shell_ids, unique_r = _get_fcc_shell_structure(N)
    n_shells = len(unique_r)
    
    initial_scale = 2.0 ** (1.0/6.0)
    x0 = np.array([initial_scale] + [0.0] * n_shells)
    
    try:
        res = minimize(_symmetry_adapted_energy_and_grad, x0, method='L-BFGS-B',
                       jac=True,
                       options={'maxiter': 2000, 'gtol': 1e-12, 'maxls': 100, 'ftol': 1e-16})
        
        global_scale = res.x[0]
        shell_params = res.x[1:]
        
        positions = np.zeros((N, 3))
        for i in range(N):
            s = shell_ids[i]
            if s < len(shell_params):
                breath = 1.0 + shell_params[s]
            else:
                breath = 1.0
            positions[i] = base_coords[i] * breath * global_scale
        
        if check_validity(positions):
            energy = lj_energy_vectorized(positions)
            return positions, energy
    except Exception:
        pass
    
    return None, float('inf')

def _two_stage_topology_optimize(start_positions):
    """
    Two-stage optimization:
    1. Use topology-forcing loss to guide toward FCC structure
    2. Then do full LJ energy optimization
    
    Returns (positions, energy) or (None, inf) on failure.
    """
    # Stage 1: Topology forcing - guide positions toward FCC shell radii
    try:
        topo_positions = _topology_forcing_optimize(start_positions, max_iter=300, gtol=1e-5)
        
        # Stage 2: Full LJ energy relaxation from topology-guided positions
        lj_positions = relax(topo_positions, maxiter=1000, gtol=1e-10)
        
        if check_validity(lj_positions):
            energy = lj_energy_vectorized(lj_positions)
            return lj_positions, energy
    except Exception:
        pass
    
    return None, float('inf')

def _high_precision_polish(positions):
    """
    Final high-precision local optimization with very tight tolerances.
    """
    x0 = positions.flatten()
    try:
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B', 
                          jac=True,
                          options={'maxiter': 10000, 'gtol': 1e-14, 'maxls': 200, 'ftol': 1e-18})
        polished = result.x.reshape(N, 3)
        if check_validity(polished):
            energy = lj_energy_vectorized(polished)
            return polished, energy
    except Exception:
        pass
    
    return positions, lj_energy_vectorized(positions)

def _symmetry_adapted_energy_and_grad(params):
    """
    Energy and gradient for symmetry-adapted parameters.
    params[0] = global scale
    params[1:] = shell breathing parameters
    """
    global_scale = params[0]
    shell_params = params[1:]
    
    base_coords, shell_ids, unique_r = _get_fcc_shell_structure(N)
    n_shells = len(unique_r)
    
    # Build positions
    positions = np.zeros((N, 3))
    for i in range(N):
        s = shell_ids[i]
        if s < len(shell_params):
            breath = 1.0 + shell_params[s]
        else:
            breath = 1.0
        positions[i] = base_coords[i] * breath * global_scale
    
    # Compute energy
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    
    # Compute gradient w.r.t. positions
    inv_r14 = inv_r12 * inv_r2
    inv_r8 = inv_r6 * inv_r2
    
    coeff = -24.0 * inv_r14 + 12.0 * inv_r8
    grad_pos_matrix = coeff[:, :, np.newaxis] * diff
    grad_positions = np.sum(grad_pos_matrix, axis=1)
    
    # Chain rule: dE/d(params)
    grad = np.zeros_like(params)
    
    # dE/d(global_scale) = sum_i grad_positions[i] . (base_coords[i] * breath_i)
    for i in range(N):
        s = shell_ids[i]
        if s < len(shell_params):
            breath = 1.0 + shell_params[s]
        else:
            breath = 1.0
        grad[0] += np.dot(grad_positions[i], base_coords[i] * breath)
    
    # dE/d(shell_params[s]) = sum_{i in shell s} grad_positions[i] . (base_coords[i] * global_scale)
    for s in range(len(shell_params)):
        mask = (shell_ids == s)
        if np.any(mask):
            grad[s+1] = np.sum([np.dot(grad_positions[i], base_coords[i] * global_scale) for i in range(N) if mask[i]])
    
    return energy, grad

def _distance_matching_loss_and_grad(positions, target_distances, weights):
    """
    Compute smooth distance-matching loss and gradient.
    
    The loss penalizes deviation from target inter-atomic distances.
    This is used for the "folding" approach where we guide the structure
    toward the FCC truncated octahedron geometry.
    
    Returns (loss, gradient).
    """
    n = len(positions)
    
    # Compute current pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r = np.sqrt(np.maximum(r2, 1e-12))
    
    # Loss: sum of weighted squared deviations
    diff_r = r - target_distances
    loss = np.sum(weights * diff_r ** 2)
    
    # Gradient: d(loss)/d(positions[i])
    # d(loss)/d(r_ij) = 2 * weights[i,j] * (r_ij - target_ij)
    # d(r_ij)/d(positions[i]) = (positions[i] - positions[j]) / r_ij
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            wij = weights[i, j]
            rij = r[i, j]
            if rij > 1e-8:
                coeff = 2.0 * wij * (rij - target_distances[i, j]) / rij
                grad[i] += coeff * diff[i, j]
    
    return loss, grad

def _folding_optimize(start_positions, target_positions, max_iter=500, gtol=1e-6):
    """
    Folding optimization: minimize distance-matching loss to guide toward target structure.
    This helps "fold" a 1D chain or perturbed structure into the 3D global minimum.
    
    Returns (positions, energy) or (None, inf) on failure.
    """
    # Compute target pairwise distances
    target_diff = target_positions[:, np.newaxis, :] - target_positions[np.newaxis, :, :]
    target_r2 = np.sum(target_diff * target_diff, axis=-1)
    np.fill_diagonal(target_r2, 1.0)
    target_distances = np.sqrt(np.maximum(target_r2, 1e-12))
    
    # Weights: prioritize close pairs (they are more structurally important)
    weights = 1.0 / (target_distances + 1.0)
    weights[weights < 0.1] = 0.1
    np.fill_diagonal(weights, 0.0)
    
    x0 = start_positions.flatten()
    
    def folding_obj_and_grad(x_flat):
        pos = x_flat.reshape(N, 3)
        loss, grad = _distance_matching_loss_and_grad(pos, target_distances, weights)
        return loss, grad.flatten()
    
    try:
        result = minimize(folding_obj_and_grad, x0, method='L-BFGS-B',
                         jac=True,
                         options={'maxiter': max_iter, 'gtol': gtol, 'maxls': 100})
        folded = result.x.reshape(N, 3)
        
        # After folding, do a full LJ relaxation
        lj_positions = relax(folded, maxiter=1000, gtol=1e-10)
        
        if check_validity(lj_positions):
            energy = lj_energy_vectorized(lj_positions)
            return lj_positions, energy
    except Exception:
        pass
    
    return None, float('inf')

def _make_1d_chain():
    """
    Create a 1D chain of 38 atoms with fixed separations.
    This is the starting point for the folding approach.
    """
    positions = np.zeros((N, 3))
    separation = 1.2  # Slightly larger than LJ equilibrium to avoid overlap
    for i in range(N):
        positions[i] = [i * separation, 0.0, 0.0]
    return positions

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_energy = float('inf')
    best_positions = None
    
    # Phase 1: Symmetry-adapted FCC optimization with shell breathing
    try:
        fcc_pos, fcc_energy = _optimize_fcc_with_shell_breathing()
        if fcc_pos is not None:
            best_energy = fcc_energy
            best_positions = fcc_pos
    except Exception:
        pass
    
    # Phase 2: Two-stage topology-forcing optimization from multiple starting points
    if time.time() - START_TIME < DEADLINE - 60:
        # Try from FCC structure
        if best_positions is not None:
            try:
                topo_pos, topo_energy = _two_stage_topology_optimize(best_positions)
                if topo_pos is not None and topo_energy < best_energy:
                    best_energy = topo_energy
                    best_positions = topo_pos
            except Exception:
                pass
        
        # Try from icosahedral structure - this may escape the icosahedral funnel
        try:
            ico_positions = _make_icosahedral_structure()
            ico_relaxed = relax(ico_positions, maxiter=200, gtol=1e-6)
            
            if check_validity(ico_relaxed):
                topo_pos, topo_energy = _two_stage_topology_optimize(ico_relaxed)
                if topo_pos is not None:
                    if topo_energy < best_energy:
                        best_energy = topo_energy
                        best_positions = topo_pos
        except Exception:
            pass
        
        # Try from a perturbed FCC structure
        try:
            base_coords, shell_ids, unique_r = _get_fcc_shell_structure(N)
            scale = 2.0 ** (1.0/6.0)
            perturbed = base_coords * scale
            rng = np.random.default_rng(777)
            perturbed = perturbed + rng.normal(0, 0.1, perturbed.shape)
            
            topo_pos, topo_energy = _two_stage_topology_optimize(perturbed)
            if topo_pos is not None:
                if topo_energy < best_energy:
                    best_energy = topo_energy
                    best_positions = topo_pos
        except Exception:
            pass
    
    # Phase 3: Folding approach - start from 1D chain and fold into FCC structure
    if time.time() - START_TIME < DEADLINE - 50:
        try:
            chain = _make_1d_chain()
            target = _make_truncated_octahedron()
            
            folded_pos, folded_energy = _folding_optimize(chain, target, max_iter=500, gtol=1e-5)
            if folded_pos is not None and folded_energy < best_energy:
                best_energy = folded_energy
                best_positions = folded_pos
        except Exception:
            pass
        
        # Try folding from perturbed chain
        if time.time() - START_TIME < DEADLINE - 40:
            try:
                chain = _make_1d_chain()
                rng = np.random.default_rng(42)
                perturbed_chain = chain + rng.normal(0, 0.2, chain.shape)
                target = _make_truncated_octahedron()
                
                folded_pos, folded_energy = _folding_optimize(perturbed_chain, target, max_iter=500, gtol=1e-5)
                if folded_pos is not None and folded_energy < best_energy:
                    best_energy = folded_energy
                    best_positions = folded_pos
            except Exception:
                pass
    
    # Phase 4: High-precision polish of the current best
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 30:
        try:
            polished, pol_energy = _high_precision_polish(best_positions)
            if pol_energy < best_energy:
                best_energy = pol_energy
                best_positions = polished
        except Exception:
            pass
    
    # Phase 5: Basin hopping from the current best
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 50:
        x0 = best_positions.flatten()
        current_energy = _lj_energy_only(x0)
        
        for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000]:
            if time.time() - START_TIME > DEADLINE - 25:
                break
            
            rng = np.random.default_rng(seed)
            step_size = 0.15 + 0.05 * (seed % 4)
            
            bh_energy, bh_x = _basin_hop(x0, current_energy, rng, step_size, 60, deadline_margin=20, 
                                         temp_start=0.20, temp_end=0.03)
            
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_x.reshape(N, 3)
                # Update x0 for next iteration
                x0 = bh_x
                current_energy = bh_energy
    
    # Phase 6: Basin hopping from icosahedral structure
    if time.time() - START_TIME < DEADLINE - 40:
        try:
            ico_positions = _make_icosahedral_structure()
            ico_positions = relax(ico_positions, maxiter=300, gtol=1e-7)
            
            if check_validity(ico_positions):
                ico_energy = lj_energy_vectorized(ico_positions)
                
                if ico_energy < best_energy:
                    best_energy = ico_energy
                    best_positions = ico_positions
                
                x0 = ico_positions.flatten()
                current_energy = _lj_energy_only(x0)
                
                for seed in [11, 22, 33, 44, 55]:
                    if time.time() - START_TIME > DEADLINE - 20:
                        break
                    
                    rng = np.random.default_rng(seed)
                    step_size = 0.18 + 0.04 * (seed % 3)
                    
                    bh_energy, bh_x = _basin_hop(x0, current_energy, rng, step_size, 50, deadline_margin=18,
                                                 temp_start=0.25, temp_end=0.04)
                    
                    if bh_energy < best_energy:
                        best_energy = bh_energy
                        best_positions = bh_x.reshape(N, 3)
                        x0 = bh_x
                        current_energy = bh_energy
        except Exception:
            pass
    
    # Phase 7: More basin hopping from the current best with larger perturbations
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 20:
        try:
            x0 = best_positions.flatten()
            current_energy = _lj_energy_only(x0)
            
            for seed in [999, 888, 777]:
                if time.time() - START_TIME > DEADLINE - 15:
                    break
                
                rng = np.random.default_rng(seed)
                # Larger step size to escape local minima
                bh_energy, bh_x = _basin_hop(x0, current_energy, rng, 0.30, 40, deadline_margin=12,
                                             temp_start=0.30, temp_end=0.05)
                
                if bh_energy < best_energy:
                    best_energy = bh_energy
                    best_positions = bh_x.reshape(N, 3)
                    x0 = bh_x
                    current_energy = bh_energy
        except Exception:
            pass
    
    # Phase 8: Final ultra-high-precision polish
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 8:
        try:
            polished, pol_energy = _high_precision_polish(best_positions)
            if pol_energy < best_energy:
                best_energy = pol_energy
                best_positions = polished
        except Exception:
            pass
    
    # Fallback
    if best_positions is None:
        try:
            base_coords, _, _ = _get_fcc_shell_structure(N)
            positions = base_coords * (2.0 ** (1.0/6.0))
            if check_validity(positions):
                best_positions = positions
            else:
                best_positions = np.zeros((N, 3))
                for i in range(N):
                    best_positions[i] = [i*0.6, (i%3)*0.6, (i//3)%3*0.6]
        except Exception:
            best_positions = np.zeros((N, 3))
            for i in range(N):
                best_positions[i] = [i*0.6, (i%3)*0.6, (i//3)%3*0.6]

    return best_positions

if __name__ == "__main__":
    t0 = time.time()
    pos = construct()
    t1 = time.time()
    e = lj_energy(pos)
    print(f"Energy: {e}, Time: {t1-t0:.2f}s")
    print(f"Valid: {check_validity(pos)}")

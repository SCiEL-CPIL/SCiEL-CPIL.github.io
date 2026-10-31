"""Two-stage topology-forcing with topology-swap basin hopping for LJ38 global minimum."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 38
DEADLINE = 148.0
SAFETY_MARGIN = 8.0

def _t0():
    return getattr(construct, '_t0', time.time())

def _elapsed():
    return time.time() - _t0()

def _time_left():
    return DEADLINE - _elapsed() - SAFETY_MARGIN

def lj_energy(pos):
    """Compute total LJ energy for N atoms."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=-1)
    np.fill_diagonal(r2, np.inf)
    r2 = np.maximum(r2, 1e-12)
    r6 = r2**(-3)
    r12 = r6**2
    energy = 4.0 * np.sum(r12 - r6)
    return energy

def lj_forces(pos):
    """Compute forces (negative gradient of energy)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=-1)
    np.fill_diagonal(r2, np.inf)
    r2 = np.maximum(r2, 1e-12)
    r6 = r2**(-3)
    r12 = r6**2
    coeff = 4.0 * (12.0 * r12 - 6.0 * r6) / r2
    np.fill_diagonal(coeff, 0.0)
    forces = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return forces

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

def relax(pos, maxiter=500, gtol=1e-10):
    """L-BFGS local relaxation."""
    x0 = pos.flatten()
    try:
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B', 
                          jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100})
        return result.x.reshape(N, 3), result.fun
    except Exception:
        return pos, lj_energy(pos)

def _fcc_truncated_octahedron():
    """
    Generate the FCC truncated octahedron geometry for LJ38.
    Uses FCC lattice points sorted by distance from center, scaled to LJ equilibrium.
    """
    scale = 1.0 / math.sqrt(2.0)
    max_coord = 5
    points = []
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = i * scale, j * scale, k * scale
                    r2 = x*x + y*y + z*z
                    points.append((x, y, z, r2))
    
    points.sort(key=lambda p: (p[3], abs(p[0]), abs(p[1]), abs(p[2])))
    
    selected = [p[:3] for p in points[:N]]
    coords = np.array(selected, dtype=float)
    
    # Scale to LJ equilibrium: nearest neighbor should be 2^(1/6)
    # In base coords, NN distance is 1.0
    lj_scale = 2.0 ** (1.0/6.0)
    return coords * lj_scale

def _topology_loss(pos, target_pos):
    """
    Smooth differentiable loss that penalizes deviations from target geometry.
    Uses pairwise distance matching.
    """
    def pairwise_dists(p):
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        return np.sqrt(np.sum(diff**2, axis=-1) + 1e-12)
    
    d_pos = pairwise_dists(pos)
    d_tgt = pairwise_dists(target_pos)
    
    i_idx, j_idx = np.triu_indices(len(pos), k=1)
    loss = np.sum((d_pos[i_idx, j_idx] - d_tgt[i_idx, j_idx])**2)
    return loss

def _topology_loss_grad(pos, target_pos):
    """Gradient of topology loss w.r.t. positions."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=-1) + 1e-12
    r = np.sqrt(r2)
    np.fill_diagonal(r, 1.0)
    
    def pairwise_dists(p):
        d = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        return np.sqrt(np.sum(d**2, axis=-1) + 1e-12)
    
    d_tgt = pairwise_dists(target_pos)
    
    grad = np.zeros_like(pos)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dij = r[i, j]
            dtij = d_tgt[i, j]
            coeff = 2.0 * (dij - dtij) / dij
            grad[i] += coeff * diff[i, j]
    
    return grad

def _detect_icosahedral_motifs(pos):
    """
    Detect atoms with icosahedral-like coordination (5-fold symmetry).
    Returns indices of atoms that appear to be in icosahedral motifs.
    """
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=-1)
    np.fill_diagonal(r2, np.inf)
    
    # Find nearest neighbors for each atom
    ico_candidates = []
    for i in range(n):
        dists = np.sqrt(r2[i, :])
        # Get 12 nearest neighbors
        nn_idx = np.argsort(dists)[:12]
        nn_dists = dists[nn_idx]
        
        # Check if coordination is icosahedral-like
        # Icosahedral: 12 neighbors at similar distances, with specific angular distribution
        if len(nn_dists) < 12:
            continue
        
        # Check distance uniformity (icosahedral has uniform shell)
        dist_mean = np.mean(nn_dists)
        dist_std = np.std(nn_dists)
        if dist_mean < 1e-6:
            continue
        uniformity = dist_std / dist_mean
        
        # Icosahedral coordination has relatively uniform neighbor distances
        # but also specific angular correlations
        if uniformity < 0.15:
            # Check angular distribution for 5-fold symmetry
            neighbors = pos[nn_idx] - pos[i]
            neighbors = neighbors / (np.linalg.norm(neighbors, axis=1, keepdims=True) + 1e-12)
            
            # Count pairs with angle ~72 degrees (5-fold symmetry)
            n_pairs_72 = 0
            n_pairs_total = 0
            for a in range(12):
                for b in range(a+1, 12):
                    dot = np.dot(neighbors[a], neighbors[b])
                    angle = np.arccos(np.clip(dot, -1, 1))
                    n_pairs_total += 1
                    # 72 degrees = 2*pi/5
                    if abs(angle - 2*np.pi/5) < 0.2 or abs(angle - 4*np.pi/5) < 0.2:
                        n_pairs_72 += 1
            
            # Icosahedron has specific fraction of 72-degree angles
            frac_72 = n_pairs_72 / max(n_pairs_total, 1)
            if frac_72 > 0.25:
                ico_candidates.append(i)
    
    return ico_candidates

def _topology_swap_displacement(pos, target_pos, ico_atoms):
    """
    Generate a targeted displacement that pushes icosahedral motifs toward FCC-like arrangement.
    Uses the topology gradient as a guide for the swap direction.
    """
    n = len(pos)
    
    # Compute topology gradient (points toward FCC-like geometry)
    topo_grad = _topology_loss_grad(pos, target_pos)
    
    # Create displacement that strongly affects icosahedral atoms
    displacement = np.zeros_like(pos)
    
    if len(ico_atoms) == 0:
        # No icosahedral motifs detected; use generic topology gradient
        displacement = topo_grad * 0.5
    else:
        # Strongly displace icosahedral atoms toward FCC topology
        for idx in ico_atoms:
            displacement[idx] = topo_grad[idx] * 2.0
        
        # Also slightly displace neighbors of icosahedral atoms
        for idx in ico_atoms:
            diff = pos - pos[idx]
            dists = np.sqrt(np.sum(diff**2, axis=1))
            np.fill_diagonal(dists, np.inf)
            # Get 6 nearest neighbors
            nn_idx = np.argsort(dists)[:6]
            for nn in nn_idx:
                if nn not in ico_atoms:
                    displacement[nn] += topo_grad[nn] * 0.5
    
    # Normalize displacement to reasonable magnitude
    disp_norm = np.linalg.norm(displacement)
    if disp_norm > 1e-6:
        displacement = displacement / disp_norm * 0.4
    
    return displacement

def _add_noise(pos, sigma=0.3):
    """Add random noise to break symmetry."""
    return pos + np.random.normal(0, sigma, size=pos.shape)

def construct():
    construct._t0 = time.time()
    np.random.seed(42)
    rng = np.random.RandomState(42)
    
    best_pos = None
    best_energy = np.inf
    
    # Phase 1: Generate FCC truncated octahedron target
    fcc_target = _fcc_truncated_octahedron()
    
    # Phase 2: Multi-start topology-forcing + LJ relaxation
    starts = []
    
    # Start 0: FCC target itself
    starts.append(fcc_target.copy())
    
    # Start 1-5: FCC target with small noise
    for s in [0.1, 0.2, 0.3, 0.5, 0.8]:
        starts.append(_add_noise(fcc_target, sigma=s))
    
    # Start 6-10: Random perturbations of FCC
    for s in range(5):
        angle = 2 * np.pi * s / 5
        rot = np.array([[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0],
                        [0, 0, 1]])
        starts.append(fcc_target @ rot.T + np.random.normal(0, 0.2, size=fcc_target.shape))
    
    # Start 11-13: Cubic lattice with FCC bias
    side = 4
    cubic_pts = []
    for i in range(side):
        for j in range(side):
            for k in range(side):
                cubic_pts.append([(i - 1.5) * 1.12, (j - 1.5) * 1.12, (k - 1.5) * 1.12])
    cubic_pts.sort(key=lambda p: sum(x**2 for x in p))
    cubic_arr = np.array(cubic_pts[:N], dtype=float)
    for s in [0.0, 0.2, 0.4]:
        starts.append(cubic_arr + np.random.normal(0, s, size=cubic_arr.shape))
    
    # Start 14-17: Pure random
    for s in range(4):
        starts.append(np.random.normal(0, 2.0, size=(N, 3)))
    
    # Process each start
    for idx, start in enumerate(starts):
        if _time_left() < 3.0:
            break
        
        start = np.array(start, dtype=float)
        
        # Phase 2a: Topology-forcing optimization (short, to get near FCC)
        if idx < 10:
            def topo_fun(x):
                p = x.reshape(-1, 3)
                return _topology_loss(p, fcc_target)
            
            def topo_grad(x):
                p = x.reshape(-1, 3)
                return _topology_loss_grad(p, fcc_target).flatten()
            
            try:
                res = minimize(topo_fun, start.flatten(), jac=topo_grad, 
                              method='L-BFGS-B', options={'maxiter': 80, 'gtol': 1e-6})
                start = res.x.reshape(-1, 3)
            except:
                pass
        
        # Phase 2b: LJ energy relaxation
        try:
            relaxed, energy = relax(start)
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed.copy()
        except:
            continue
        
        # Phase 2c: Basin hopping with topology-swap from the relaxed structure
        if _time_left() < 5.0:
            break
            
        current_pos = relaxed.copy()
        current_energy = energy
        
        # Adaptive basin hopping with topology-swap
        base_step = 0.3
        n_bh_iters = 25
        
        for bh_iter in range(n_bh_iters):
            if _time_left() < 3.0:
                break
            
            # Annealing temperature
            progress = bh_iter / max(n_bh_iters - 1, 1)
            temp = 0.15 + (0.02 - 0.15) * progress
            
            # Adaptive step size
            step_size = base_step
            if bh_iter % 6 == 3:
                if rng.uniform() < 0.35:
                    step_size = base_step * rng.uniform(1.5, 3.0)
                elif rng.uniform() < 0.3:
                    step_size = base_step * rng.uniform(0.3, 0.7)
            
            # Decide whether to use topology-swap or generic perturbation
            use_swap = rng.uniform() < 0.4
            
            if use_swap:
                # Detect icosahedral motifs and apply targeted swap
                ico_atoms = _detect_icosahedral_motifs(current_pos)
                swap_disp = _topology_swap_displacement(current_pos, fcc_target, ico_atoms)
                noise = rng.normal(0, step_size * 0.5, size=current_pos.shape)
                perturbed = current_pos + swap_disp + noise
            else:
                # Generic random perturbation
                noise = rng.normal(0, step_size, size=current_pos.shape)
                perturbed = current_pos + noise
            
            # Relax
            try:
                new_pos, new_energy = relax(perturbed, maxiter=200, gtol=1e-7)
            except:
                continue
            
            # Metropolis accept
            if new_energy < current_energy or rng.uniform() < math.exp(-(new_energy - current_energy) / max(temp, 1e-10)):
                current_pos = new_pos
                current_energy = new_energy
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pos = new_pos.copy()
        
        if _time_left() < 5.0:
            break
    
    # Final: One more relaxation of best with tighter tolerance
    if best_pos is not None and _time_left() > 3.0:
        try:
            final_pos, final_energy = relax(best_pos, maxiter=1000, gtol=1e-12)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pos = final_pos
        except:
            pass
    
    # Ensure minimum distance constraint
    if best_pos is not None:
        # Check and fix any violations
        diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
        r2 = np.sum(diff**2, axis=-1)
        np.fill_diagonal(r2, np.inf)
        min_r2 = np.min(r2)
        if min_r2 < 0.25:  # 0.5^2
            # Nudge atoms apart
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(best_pos[i] - best_pos[j])
                    if d < 0.5:
                        direction = (best_pos[i] - best_pos[j]) / (d + 1e-12)
                        shift = (0.5 - d + 0.01) / 2
                        best_pos[i] += direction * shift
                        best_pos[j] -= direction * shift
    
    if best_pos is None:
        # Fallback: cubic lattice
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_pos = np.array(points[:N], dtype=float) * 1.12
    
    return best_pos

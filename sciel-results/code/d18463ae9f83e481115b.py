"""Topological basin-hopping for LJ38 global minimum."""
import numpy as np
import time
from scipy.optimize import minimize

N = 38
DEADLINE_MARGIN = 10.0  # seconds before the 150s kill


def _lj_energy(pos):
    """Compute total LJ energy: 4*sum(r^-12 - r^-6) over all pairs."""
    # pos: (N, 3)
    # Use vectorized pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** 3
    r12 = r6 ** 2
    energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    # Sum over upper triangle only
    energy = 0.5 * np.sum(energy)
    return energy


def _lj_force(pos):
    """Compute LJ forces on each atom (negative gradient of energy)."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** 3
    r12 = r6 ** 2
    # Force on i from j: 4*(12/r^13 * r_vec - 6/r^7 * r_vec) where r_vec = pos_j - pos_i
    # In our convention diff[i,j] = pos_i - pos_j, so force on i = 4*(12/r^12 - 6/r^6) * (pos_j - pos_i)/r
    # = 4*(12/r^12 - 6/r^6) * (-diff[i,j])/r
    # Let's compute: f_ij = 4*(12/r^13 - 6/r^7) * r_vec_ij where r_vec_ij = pos_j - pos_i = -diff[i,j]
    coeff = 4.0 * (12.0 / r12 - 6.0 / r6) / np.sqrt(r2)  # 4*(12/r^13 - 6/r^7)
    coeff[np.diag_indices_from(coeff)] = 0.0
    forces = -np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return forces


def _fcc_truncated_octahedron():
    """
    Construct an FCC truncated octahedron for 38 atoms.
    The truncated octahedron has 24 vertices (FCC lattice points) plus interior points.
    
    Actually, the known global minimum for LJ38 is a truncated octahedron derived from FCC.
    Let's use the standard construction:
    - Start with FCC lattice, take a Wigner-Seitz cell (truncated octahedron)
    - The 38-atom cluster is typically: 1 center + 12 first-shell + 14 second-shell + 11 third-shell
    or similar FCC-based arrangement.
    
    A simpler approach: use known coordinates or build from FCC lattice.
    FCC lattice points: all (i,j,k) with i+j+k even, scaled by a/2 where a is lattice constant.
    
    For LJ, the nearest-neighbor distance is 1.12246 (2^(1/6)).
    In FCC, nearest neighbors are at distance a/sqrt(2) where a is the cube edge.
    So a = 1.12246 * sqrt(2) ≈ 1.5874.
    
    Let's generate FCC points within a radius and select 38 closest to origin.
    """
    a = 2.0 ** (1.0 / 6.0) * np.sqrt(2.0)  # FCC lattice constant for LJ
    points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                if (i + j + k) % 2 == 0:
                    points.append((i * a / 2.0, j * a / 2.0, k * a / 2.0))
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:N]
    return points[idx].copy()


def _relax(pos, max_iter=500):
    """Local relaxation using L-BFGS."""
    try:
        res = minimize(_lj_energy, pos, jac=_lj_force, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        return res.x
    except Exception:
        return pos


def _detect_icosahedral_motif(pos):
    """
    Check if the structure has icosahedral-like coordination.
    In icosahedral clusters, the central atom has 12 neighbors at nearly equal distances
    with icosahedral symmetry. We check for high local coordination number with
    similar bond lengths.
    """
    # Find the most coordinated atom
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    
    # Count neighbors within a shell (e.g., 1.3 * r_min)
    r_min = r.min()
    shell = 1.3 * r_min
    neighbor_counts = (r < shell).sum(axis=1)
    
    max_coord = neighbor_counts.max()
    # Icosahedral clusters have a 12-coordinate center
    if max_coord >= 11:
        return True
    return False


def _topology_swap_displacement(pos, rng):
    """
    Apply a targeted displacement to break icosahedral symmetry and move toward FCC.
    Identify the central high-coordination atom and its 12 neighbors,
    then apply a displacement that breaks the icosahedral arrangement.
    """
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    
    r_min = r.min()
    shell = 1.3 * r_min
    neighbor_counts = (r < shell).sum(axis=1)
    
    # Find the most coordinated atom
    center_idx = np.argmax(neighbor_counts)
    
    # Get its neighbors
    neighbor_mask = r[center_idx] < shell
    neighbor_indices = np.where(neighbor_mask)[0]
    
    if len(neighbor_indices) < 12:
        # Not clearly icosahedral; apply generic perturbation
        return rng.normal(0, 0.3, size=pos.shape)
    
    # Take the 12 nearest neighbors
    neighbor_dists = r[center_idx, neighbor_indices]
    order = np.argsort(neighbor_dists)[:12]
    neighbors = neighbor_indices[order]
    
    # The icosahedral neighbors form a roughly spherical shell.
    # To break icosahedral symmetry, we displace some neighbors toward FCC positions.
    # FCC has 12 nearest neighbors but with cubic symmetry (4+4+4 in octants).
    # We nudge neighbors in the +x,+y,+z octant outward and others inward.
    
    displacement = np.zeros_like(pos)
    center = pos[center_idx]
    
    # FCC-like displacement: move neighbors to break icosahedral 5-fold axes
    # icosahedral has 6 five-fold axes; FCC has 4 three-fold axes.
    # Simple heuristic: displace each neighbor along a direction that breaks icosahedral symmetry.
    
    # Compute the average neighbor direction
    neighbor_vectors = pos[neighbors] - center
    neighbor_vectors = neighbor_vectors / (np.linalg.norm(neighbor_vectors, axis=1, keepdims=True) + 1e-10)
    
    # Apply a perturbation that breaks icosahedral symmetry
    # Displace neighbors by a small amount in a direction correlated with their position
    # to favor FCC over icosahedral
    
    for i, idx in enumerate(neighbors):
        vec = neighbor_vectors[i]
        # Add a component that favors cubic symmetry over icosahedral
        # The difference between FCC and icosahedral is in the angular distribution
        # We nudge toward the cubic axes
        cubic_component = np.sign(vec) * 0.1
        displacement[idx] += cubic_component + rng.normal(0, 0.1, size=3)
    
    # Also slightly displace the center
    displacement[center_idx] += rng.normal(0, 0.05, size=3)
    
    return displacement


def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    rng = np.random.default_rng(42)
    
    # Start from FCC truncated octahedron
    best_pos = _fcc_truncated_octahedron()
    best_energy = _lj_energy(best_pos)
    
    # Relax the initial FCC structure
    best_pos = _relax(best_pos, max_iter=200)
    best_energy = _lj_energy(best_pos)
    
    # Basin hopping with topology-swap perturbations
    step_sizes = [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]
    current_pos = best_pos.copy()
    current_energy = best_energy
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Choose perturbation strategy
        if _detect_icosahedral_motif(current_pos) and rng.random() < 0.6:
            # Use topology-swap displacement
            disp = _topology_swap_displacement(current_pos, rng)
            trial_pos = current_pos + disp
        else:
            # Generic random perturbation with adaptive step size
            step = step_sizes[min(iteration // 20, len(step_sizes) - 1)]
            disp = rng.normal(0, step, size=current_pos.shape)
            trial_pos = current_pos + disp
        
        # Relax the trial position
        trial_pos = _relax(trial_pos, max_iter=300)
        trial_energy = _lj_energy(trial_pos)
        
        # Metropolis acceptance
        if trial_energy < current_energy:
            current_pos = trial_pos
            current_energy = trial_energy
        else:
            # Accept with Metropolis probability
            delta_e = trial_energy - current_energy
            # Use a temperature that decreases over time
            temp = max(0.1, 2.0 / (1.0 + iteration * 0.05))
            if rng.random() < np.exp(-delta_e / temp):
                current_pos = trial_pos
                current_energy = trial_energy
        
        # Update best
        if current_energy < best_energy:
            best_pos = current_pos.copy()
            best_energy = current_energy
        
        # Check deadline
        if time.time() > deadline - 5.0:
            break
    
    # Final high-precision polish on the best structure
    if time.time() < deadline - 3.0:
        polished = _relax(best_pos, max_iter=1000)
        polished_energy = _lj_energy(polished)
        if polished_energy < best_energy:
            best_pos = polished
            best_energy = polished_energy
    
    # Ensure minimum distance constraint (atoms at least 0.5 apart)
    # Check and fix if needed
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    if r.min() < 0.5:
        # Scale up slightly
        scale = 0.5 / r.min() * 1.01
        best_pos = best_pos * scale
    
    return best_pos

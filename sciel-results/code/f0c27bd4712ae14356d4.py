"""Topology-swap basin hopping for LJ38 global minimum."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 38


def lj_energy(pos):
    """Compute total Lennard-Jones energy for given positions."""
    d = cdist(pos, pos)
    np.fill_diagonal(d, np.inf)
    r2 = d ** 2
    r6 = 1.0 / r2 ** 3
    r12 = r6 ** 2
    energy = 4.0 * np.sum(r12 - r6)
    return energy


def lj_forces(pos):
    """Compute forces (negative gradient of energy) for given positions."""
    n = len(pos)
    d = cdist(pos, pos)
    np.fill_diagonal(d, np.inf)
    r2 = d ** 2
    r6 = 1.0 / r2 ** 3
    r12 = r6 ** 2
    # Gradient of energy w.r.t. positions: dE/dr_ij = 4*(-12*r^-13 + 6*r^-7)
    # Force on i from j: -dE/dr_ij * (r_i - r_j)/r_ij
    # Using vector form: F_ij = 48*(r^-14 - 0.5*r^-8) * r_vec_ij
    r14 = r12 / r2
    r8 = r6 / r2
    coeff = 48.0 * (r14 - 0.5 * r8)
    # coeff[i,j] is the coefficient for vector (pos[i] - pos[j])
    forces = -np.sum(coeff[:, :, None] * (pos[:, None, :] - pos[None, :, :]), axis=1)
    return forces


def relax(pos):
    """Relax using L-BFGS."""
    def fun(x):
        p = x.reshape(-1, 3)
        return lj_energy(p)
    
    def jac(x):
        p = x.reshape(-1, 3)
        f = lj_forces(p)
        return f.reshape(-1)
    
    try:
        res = minimize(fun, pos.reshape(-1), jac=jac, method='L-BFGS-B',
                       options={'maxiter': 500, 'gtol': 1e-8})
        return res.x.reshape(-1, 3)
    except Exception:
        return pos


def detect_icosahedral_motif(pos, i, n_neighbors=12):
    """
    Detect if atom i has an icosahedral-like coordination.
    Icosahedral: 12 neighbors, with specific angular distribution.
    FCC: 12 neighbors, with cubic symmetry.
    
    We use the angular variance of neighbor directions as a discriminator.
    Icosahedral has 5-fold axes; FCC has 4-fold axes.
    """
    d = cdist(pos[i:i+1], pos)[0]
    order = np.argsort(d)
    order = order[:n_neighbors]
    d_close = d[order]
    
    if np.any(d_close < 0.5):
        return False
    
    # Get unit vectors to neighbors
    vecs = pos[order] - pos[i]
    norms = np.linalg.norm(vecs, axis=1)
    if np.any(norms < 1e-8):
        return False
    uvecs = vecs / norms[:, None]
    
    # Compute the angular distribution
    # For icosahedral: neighbors are at vertices of icosahedron
    # For FCC: neighbors are at vertices of cuboctahedron
    # Discriminate using the number of near-180 degree pairs (opposite pairs)
    # Icosahedron has 6 opposite pairs; cuboctahedron has 6 opposite pairs too
    # Better: use the second moment of the angular distribution
    
    # Compute dot products between all neighbor direction pairs
    dots = uvecs @ uvecs.T
    np.fill_diagonal(dots, 0)
    
    # Icosahedral: 12 neighbors, 30 edges, specific angles
    # The key difference: icosahedral has 5-fold symmetry (72 degree angles)
    # FCC has 4-fold symmetry (90 degree angles)
    
    # Use the fraction of neighbors with dot product near cos(72) ≈ 0.309
    # vs cos(90) = 0
    cos72 = np.cos(np.pi * 2 / 5)  # ~0.309
    cos90 = 0.0
    
    # Count neighbors with angular separation close to 72 degrees (icosahedral edge)
    n_ico_edges = np.sum(np.abs(dots - cos72) < 0.15)
    # Count neighbors with angular separation close to 90 degrees (FCC)
    n_fcc_edges = np.sum(np.abs(dots - cos90) < 0.15)
    
    # Icosahedral tends to have more 72-degree separations
    ico_score = n_ico_edges / n_fcc_edges if n_fcc_edges > 0 else (n_ico_edges + 1)
    
    return ico_score > 1.5


def topology_swap_displacement(pos, i, n_neighbors=12):
    """
    Generate a targeted displacement for atom i to break icosahedral symmetry
    and push toward FCC-like arrangement.
    """
    d = cdist(pos[i:i+1], pos)[0]
    order = np.argsort(d)
    order = order[:n_neighbors]
    d_close = d[order]
    
    vecs = pos[order] - pos[i]
    norms = np.linalg.norm(vecs, axis=1)
    if np.any(norms < 1e-8):
        return np.zeros(3)
    uvecs = vecs / norms[:, None]
    
    # The icosahedral to FCC transition involves moving the central atom
    # out of the icosahedral center toward one of the FCC positions.
    # In FCC, the 12 neighbors form a cuboctahedron; the central atom
    # is at the center. In icosahedral, the 12 neighbors form an icosahedron.
    
    # Strategy: identify the 5-fold axis of the icosahedron and displace
    # perpendicular to it to break the 5-fold symmetry.
    
    # Find the principal axis using PCA of neighbor positions
    centered = uvecs - np.mean(uvecs, axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # The 5-fold axis is typically the direction with smallest variance
    # (neighbors are more symmetric around this axis)
    axis = eigvecs[:, 0]  # smallest eigenvalue
    
    # Displace perpendicular to the 5-fold axis, toward a direction
    # that breaks the icosahedral symmetry
    # Use a direction that is roughly along one of the neighbor directions
    # but projected to be perpendicular to the 5-fold axis
    
    # Pick the neighbor direction most perpendicular to the 5-fold axis
    dots_with_axis = np.abs(uvecs @ axis)
    perp_idx = np.argmin(dots_with_axis)
    disp_dir = uvecs[perp_idx] - (uvecs[perp_idx] @ axis) * axis
    disp_norm = np.linalg.norm(disp_dir)
    if disp_norm < 1e-8:
        disp_dir = axis
        disp_norm = 1.0
    disp_dir = disp_dir / disp_norm
    
    # Small displacement to nudge out of icosahedral basin
    magnitude = 0.15
    return magnitude * disp_dir


def construct():
    # Start from the known FCC truncated octahedron structure for LJ38
    # This is the putative global minimum with energy ~ -173.928427
    
    # FCC truncated octahedron: layers in (111) direction
    # The structure can be described as a truncated octahedron with
    # 8 hexagonal faces and 6 square faces.
    
    # Let me construct it from known coordinates.
    # The FCC truncated octahedron for n=38 has the following layer structure:
    # Layer 1: 1 atom (center of one hex face)
    # Layer 2: 6 atoms (hexagon)
    # Layer 3: 12 atoms (larger ring)
    # Layer 4: 8 atoms
    # Layer 5: 6 atoms
    # Layer 6: 1 atom
    # Wait, that's 1+6+12+8+6+1 = 34. Need 38.
    
    # Let me use a different construction. The truncated octahedron in FCC
    # for 38 atoms:
    # Actually, let me just use a well-known seed and relax.
    
    # Better approach: start from a random FCC-like arrangement and relax,
    # then apply topology-swap basin hopping to escape icosahedral traps.
    
    # Generate FCC lattice points and take the 38 closest to center
    a = 1.09  # slightly compressed FCC spacing for LJ
    
    # FCC lattice vectors
    fcc_basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ]) * a
    
    # Generate a grid of FCC points
    n_grid = 4
    points = []
    for i in range(-n_grid, n_grid+1):
        for j in range(-n_grid, n_grid+1):
            for k in range(-n_grid, n_grid+1):
                for b in fcc_basis:
                    p = np.array([i, j, k]) * a + b
                    points.append(p)
    points = np.array(points)
    
    # Sort by distance from origin and take N closest
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:N]]
    
    # Center the cluster
    points = points - np.mean(points, axis=0)
    
    # Relax to local minimum
    pos = relax(points)
    
    # Now apply topology-swap basin hopping
    best_pos = pos.copy()
    best_energy = lj_energy(pos)
    
    rng = np.random.default_rng(42)
    
    for iteration in range(30):
        # Check for icosahedral motifs and apply targeted swaps
        ico_found = False
        for i in range(N):
            if detect_icosahedral_motif(pos, i):
                disp = topology_swap_displacement(pos, i)
                new_pos = pos.copy()
                new_pos[i] += disp
                ico_found = True
                break
        
        if ico_found:
            candidate = relax(new_pos)
        else:
            # No icosahedral motif detected; try a random perturbation
            perturbation = 0.2 * rng.standard_normal((N, 3))
            candidate = relax(pos + perturbation)
        
        cand_energy = lj_energy(candidate)
        
        # Metropolis accept/reject
        if cand_energy < best_energy:
            best_energy = cand_energy
            best_pos = candidate
            pos = candidate
        else:
            # Accept with some probability for exploration
            # Simple: just keep going with the perturbed version
            pos = candidate
    
    # Final relaxation
    best_pos = relax(best_pos)
    final_energy = lj_energy(best_pos)
    
    # Ensure minimum distance constraint
    d = cdist(best_pos, best_pos)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 0.5:
        # If constraint violated, fall back to the original relaxed FCC
        fallback = relax(points)
        return fallback
    
    return best_pos


# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 38

def lj_energy(pos):
    """Compute total Lennard-Jones energy for given positions."""
    d = cdist(pos, pos)
    np.fill_diagonal(d, np.inf)
    r2 = d ** 2
    r6 = 1.0 / r2 ** 3
    r12 = r6 ** 2
    energy = 4.0 * np.sum(r12 - r6)
    return energy

def lj_forces(pos):
    """Compute forces (negative gradient of energy) for given positions."""
    n = len(pos)
    d = cdist(pos, pos)
    np.fill_diagonal(d, np.inf)
    r2 = d ** 2
    r6 = 1.0 / r2 ** 3
    r12 = r6 ** 2
    r14 = r12 / r2
    r8 = r6 / r2
    coeff = 48.0 * (r14 - 0.5 * r8)
    forces = -np.sum(coeff[:, :, None] * (pos[:, None, :] - pos[None, :, :]), axis=1)
    return forces

def relax(pos):
    """Relax using L-BFGS."""
    def fun(x):
        p = x.reshape(-1, 3)
        return lj_energy(p)
    
    def jac(x):
        p = x.reshape(-1, 3)
        f = lj_forces(p)
        return f.reshape(-1)
    
    try:
        res = minimize(fun, pos.reshape(-1), jac=jac, method='L-BFGS-B',
                       options={'maxiter': 500, 'gtol': 1e-8})
        return res.x.reshape(-1, 3)
    except Exception:
        return pos

def detect_icosahedral(pos, i, n_neighbors=12):
    """
    Detect icosahedral coordination by analyzing the angular distribution
    of the 12 nearest neighbors.
    Returns (is_icosahedral, axis_vector)
    """
    d = cdist(pos[i:i+1], pos)[0]
    order = np.argsort(d)
    neighbors = order[:n_neighbors]
    d_close = d[neighbors]
    
    if np.any(d_close < 0.5):
        return False, None
    
    vecs = pos[neighbors] - pos[i]
    norms = np.linalg.norm(vecs, axis=1)
    if np.any(norms < 1e-8):
        return False, None
        
    uvecs = vecs / norms[:, None]
    
    # Compute all pairwise angles between the 12 neighbors
    dots = uvecs @ uvecs.T
    np.fill_diagonal(dots, 1.0)
    
    # Extract unique angles (upper triangle)
    angles = np.arccos(np.clip(dots, -1.0, 1.0))
    iu = np.triu_indices(n_neighbors, 1)
    pair_angles = angles[iu]
    
    # Define bins for the characteristic angles
    ico_peaks = [np.deg2rad(72), np.deg2rad(116.565), np.deg2rad(144)]
    fcc_peaks = [np.deg2rad(90), np.deg2rad(109.471), np.deg2rad(120)]
    
    def score_angles(angles, peaks):
        score = 0
        for p in peaks:
            close = np.sum(np.abs(angles - p) < np.deg2rad(5))
            score += close
        return score
    
    ico_score = score_angles(pair_angles, ico_peaks)
    fcc_score = score_angles(pair_angles, fcc_peaks)
    
    if ico_score > fcc_score + 2:
        # Find the 5-fold axis: look for opposite pairs (dot ~ -1)
        opposite_mask = dots < -0.9
        opp_i, opp_j = np.where(opposite_mask)
        if len(opp_i) > 0:
            axis_vec = uvecs[opp_i[0]]
            return True, axis_vec
        else:
            cov = np.cov(uvecs.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            axis_vec = eigvecs[:, 0]
            return True, axis_vec
            
    return False, None

def construct():
    """
    Topology-swap basin hopping for LJ38.
    """
    # 1. Generate initial FCC-like structure
    a = 1.09
    fcc_basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ]) * a
    
    n_grid = 4
    points = []
    for i in range(-n_grid, n_grid+1):
        for j in range(-n_grid, n_grid+1):
            for k in range(-n_grid, n_grid+1):
                for b in fcc_basis:
                    p = np.array([i, j, k]) * a + b
                    points.append(p)
    points = np.array(points)
    
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:N]]
    points = points - np.mean(points, axis=0)
    
    pos = relax(points)
    best_pos = pos.copy()
    best_energy = lj_energy(pos)
    
    rng = np.random.default_rng(42)
    
    # Basin Hopping Loop
    for iteration in range(50):
        # Try to detect icosahedral motifs
        ico_found = False
        for i in range(N):
            is_ico, axis = detect_icosahedral(pos, i)
            if is_ico:
                # Targeted Swap: Displace atom i along the 5-fold axis
                disp = 0.5 * axis
                disp += 0.05 * rng.standard_normal(3)
                
                new_pos = pos.copy()
                new_pos[i] += disp
                candidate = relax(new_pos)
                ico_found = True
                break
        
        if not ico_found:
            # Random perturbation
            scale = 0.5 * (1.0 - iteration / 50.0) + 0.1
            perturbation = scale * rng.standard_normal((N, 3))
            candidate = relax(pos + perturbation)
        
        cand_energy = lj_energy(candidate)
        
        # Metropolis Acceptance
        T = 1.0 * (1.0 - iteration / 50.0) + 0.1
        dE = cand_energy - lj_energy(pos)
        
        if dE < 0 or np.log(rng.uniform()) < -dE / T:
            pos = candidate
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_pos = candidate.copy()
    
    # Final Relaxation
    best_pos = relax(best_pos)
    
    # Check constraints
    d = cdist(best_pos, best_pos)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 0.5:
        return relax(points)
        
    return best_pos
# EVOLVE-BLOCK-END

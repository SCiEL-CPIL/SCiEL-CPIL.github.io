"""Morse cluster global minima: basin hopping with crystalline starts and deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(pos, rho):
    """Compute Morse energy E = sum_{i<j} exp(rho(1-r_ij))(exp(rho(1-r_ij)) - 2)."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    iu = np.triu_indices(n, k=1)
    r_ij = r[iu]
    x = rho * (1.0 - r_ij)
    e = np.exp(x) * (np.exp(x) - 2.0)
    return float(np.sum(e))


def _morse_energy_and_grad(pos, rho):
    """Compute Morse energy and gradient."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    iu = np.triu_indices(n, k=1)
    r_ij = r[iu]
    x = rho * (1.0 - r_ij)
    ex = np.exp(x)
    e = ex * (ex - 2.0)
    energy = float(np.sum(e))
    
    # Gradient: dE/dr_ij = 2*rho*ex*(ex-1)
    # dE/dpos_i = sum_j 2*rho*ex*(ex-1) * (pos_i - pos_j)/r_ij
    grad = np.zeros_like(pos)
    de_dr = 2.0 * rho * ex * (ex - 1.0)
    # For each pair (i,j), contribution to grad[i] is de_dr * (pos[i]-pos[j])/r_ij
    # and to grad[j] is de_dr * (pos[j]-pos[i])/r_ij = -de_dr * (pos[i]-pos[j])/r_ij
    for k in range(len(r_ij)):
        i, j = iu[0][k], iu[1][k]
        rij = r_ij[k]
        if rij < 1e-10:
            continue
        factor = de_dr[k] / rij
        grad[i] += factor * (pos[i] - pos[j])
        grad[j] -= factor * (pos[i] - pos[j])
    return energy, grad


def _is_valid(pos, min_dist=0.5):
    """Check if all pairwise distances are at least min_dist."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    return np.min(r) >= min_dist - 1e-9


def _generate_fcc(n, spacing=1.0):
    """Generate an FCC-like cluster of n atoms."""
    # FCC lattice vectors
    a = spacing
    # Generate points in FCC lattice
    points = []
    # Use a cubic region and keep points close to center
    max_coord = int(np.ceil(n ** (1/3))) + 2
    for i in range(-max_coord, max_coord+1):
        for j in range(-max_coord, max_coord+1):
            for k in range(-max_coord, max_coord+1):
                # FCC has 4 atoms per unit cell: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    if np.linalg.norm(p) <= max_coord * a * 0.7:
                        points.append(p)
    
    points = np.array(points)
    # Sort by distance from center and take first n
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    # Center the cluster
    points -= np.mean(points, axis=0)
    return points


def _generate_hcp(n, spacing=1.0):
    """Generate an HCP-like cluster of n atoms."""
    a = spacing
    c = a * np.sqrt(8.0/3.0)
    points = []
    max_coord = int(np.ceil(n ** (1/3))) + 2
    for i in range(-max_coord, max_coord+1):
        for j in range(-max_coord, max_coord+1):
            for k in range(-max_coord, max_coord+1):
                # HCP structure
                if k % 2 == 0:
                    # AB layer
                    p1 = np.array([i, j, k]) * a
                    p2 = np.array([i + 0.5, j + 0.5, k]) * a
                else:
                    # C layer (shifted)
                    p1 = np.array([i + 1/3.0, j + 2/3.0, k]) * a
                    p2 = np.array([i + 2/3.0, j + 1/3.0, k]) * a
                for p in [p1, p2]:
                    if np.linalg.norm(p) <= max_coord * a * 0.7:
                        points.append(p)
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points -= np.mean(points, axis=0)
    return points


def _generate_bcc(n, spacing=1.0):
    """Generate a BCC-like cluster of n atoms."""
    a = spacing
    points = []
    max_coord = int(np.ceil(n ** (1/3))) + 2
    for i in range(-max_coord, max_coord+1):
        for j in range(-max_coord, max_coord+1):
            for k in range(-max_coord, max_coord+1):
                # BCC: corners + center
                p1 = np.array([i, j, k]) * a
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5]) * a
                for p in [p1, p2]:
                    if np.linalg.norm(p) <= max_coord * a * 0.7:
                        points.append(p)
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points -= np.mean(points, axis=0)
    return points


def _generate_random_compact(n, spacing=1.0, rng=None):
    """Generate a random compact cluster."""
    if rng is None:
        rng = np.random.default_rng()
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 1000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
                points.append(p)
        attempts += 1
    
    # If we couldn't place all atoms, add more with looser constraints
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    return points


def _basin_hopping(pos, rho, max_steps=20, step_size=0.3, rng=None, deadline=None):
    """Perform basin hopping optimization."""
    if rng is None:
        rng = np.random.default_rng()
    
    energy, pos = _morse_energy_and_grad(pos, rho)
    
    for step in range(max_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random perturbation
        delta = rng.normal(0, step_size, size=pos.shape)
        new_pos = pos + delta
        
        # Check validity
        if not _is_valid(new_pos):
            continue
        
        new_energy, _ = _morse_energy_and_grad(new_pos, rho)
        
        # Accept if lower energy or with some probability
        if new_energy < energy:
            energy, pos = new_energy, new_pos
        else:
            # Metropolis criterion
            delta_e = new_energy - energy
            if delta_e < 10.0:
                prob = np.exp(-delta_e / 1.0)
                if rng.random() < prob:
                    energy, pos = new_energy, new_pos
    
    # Final local relaxation
    energy, pos = relax(pos, 'morse', rho)
    
    return energy, pos


def construct(instance):
    """Find low-energy Morse cluster configuration using basin hopping with multiple starts."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 290  # 290s deadline with 40s safety margin
    
    rng = np.random.default_rng(size)
    
    # Generate multiple starting configurations
    starts = []
    
    # FCC start
    try:
        fcc = _generate_fcc(size, spacing=1.0)
        starts.append(('fcc', fcc))
    except:
        pass
    
    # HCP start
    try:
        hcp = _generate_hcp(size, spacing=1.0)
        starts.append(('hcp', hcp))
    except:
        pass
    
    # BCC start
    try:
        bcc = _generate_bcc(size, spacing=1.0)
        starts.append(('bcc', bcc))
    except:
        pass
    
    # Random compact starts
    for i in range(3):
        try:
            rand = _generate_random_compact(size, spacing=1.0, rng=rng)
            starts.append((f'rand_{i}', rand))
        except:
            pass
    
    best_energy = np.inf
    best_pos = None
    
    for name, start_pos in starts:
        if time.time() > deadline:
            break
        
        # First, do local relaxation on the start
        try:
            energy, relaxed_pos = relax(start_pos, 'morse', parameter)
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos.copy()
        except:
            continue
        
        # Then do basin hopping
        try:
            bh_energy, bh_pos = _basin_hopping(relaxed_pos, parameter, max_steps=15, 
                                                step_size=0.25, rng=rng, deadline=deadline)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos.copy()
        except:
            continue
    
    # If no valid result, fall back to a simple random start
    if best_pos is None:
        try:
            rand = _generate_random_compact(size, spacing=1.0, rng=rng)
            best_energy, best_pos = relax(rand, potential, parameter)
        except:
            # Last resort: linear arrangement
            best_pos = np.zeros((size, 3))
            best_pos[:, 0] = np.arange(size) * 1.0
    
    return best_pos
# EVOLVE-BLOCK-END

import numpy as np
import time
import math
from scipy.optimize import minimize

N = 38
DEADLINE = 135.0
START_TIME = time.time()

def lj_energy(positions):
    """Compute total Lennard-Jones energy: 4 * sum(r^-12 - r^-6) over all pairs."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r2 = np.dot(rij, rij)
            if r2 < 1e-12:
                return 1e20
            inv_r2 = 1.0 / r2
            inv_r6 = inv_r2 * inv_r2 * inv_r2
            inv_r12 = inv_r6 * inv_r6
            energy += 4.0 * (inv_r12 - inv_r6)
    return energy

def lj_energy_and_grad(positions):
    """Compute LJ energy and gradient analytically (vectorized, optimized)."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 * inv_r2 * inv_r2
    inv_r12 = inv_r6 * inv_r6
    energy = 2.0 * np.sum(inv_r12 - inv_r6)
    coeff = 24.0 * inv_r2 * (inv_r6 - 2.0 * inv_r12)
    grad = np.einsum('ij,ijk->ik', coeff, diff)
    return energy, grad

def lj_energy_vectorized(positions):
    """Vectorized LJ energy computation (optimized)."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 * inv_r2 * inv_r2
    inv_r12 = inv_r6 * inv_r6
    energy = 2.0 * np.sum(inv_r12 - inv_r6)
    return energy

def check_validity(positions, min_dist=0.5):
    """Check that all atoms are at least min_dist apart (vectorized)."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    return min_r2 >= min_dist * min_dist

def make_fcc_truncated_octahedron(n_atoms=38):
    """
    Generate the FCC truncated octahedron cluster for n_atoms atoms.
    This is the known global minimum structure for LJ38.
    
    Build the cluster shell-by-shell from the FCC lattice:
    - Shell 0: origin (1 atom)
    - Shell 1: 12 nearest neighbors (total 13)
    - Shell 2: 24 next-nearest neighbors (total 37)
    - Shell 3: add 1 more from the next shell (total 38)
    
    FCC lattice: points (i, j, k) where i+j+k is even
    Scale by 1/sqrt(2) to get nearest neighbor distance = 1
    """
    scale = 1.0 / math.sqrt(2.0)
    max_coord = 8
    points = []
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = i * scale, j * scale, k * scale
                    r2 = x*x + y*y + z*z
                    points.append((x, y, z, r2))
    
    # Sort by distance from origin
    points.sort(key=lambda p: p[3])
    
    # Select the first n_atoms points
    selected = [p[:3] for p in points[:n_atoms]]
    return np.array(selected, dtype=float)

def _generate_fcc_points(max_coord=8):
    """Generate FCC lattice points sorted by distance from origin."""
    scale = 1.0 / math.sqrt(2.0)
    points = []
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = i * scale, j * scale, k * scale
                    r2 = x*x + y*y + z*z
                    points.append((x, y, z, r2))
    
    points.sort(key=lambda p: p[3])
    return points

def _get_fcc_shell_structure(n_atoms=38):
    """
    Get FCC shell structure with different outer-shell subsets.
    Returns list of candidate structures.
    """
    fcc_points = _generate_fcc_points(8)
    candidates = []
    
    # Base: first 38 points
    base = [p[:3] for p in fcc_points[:38]]
    candidates.append(np.array(base, dtype=float))
    
    # Try different outer-shell selections
    # Shell 0: 1 atom (origin)
    # Shell 1: 12 atoms (nearest neighbors)
    # Shell 2: 24 atoms (next nearest)
    # Shell 3+: remaining
    
    # Identify shells by distance
    shell_distances = [p[3] for p in fcc_points]
    
    # Find unique distances (shells)
    shells = []
    current_shell = [0]
    for i in range(1, len(shell_distances)):
        if abs(shell_distances[i] - shell_distances[i-1]) > 1e-6:
            shells.append(current_shell)
            current_shell = []
        current_shell.append(i)
    shells.append(current_shell)
    
    # Build core (shells 0, 1, 2) = 37 atoms
    core_indices = []
    for shell in shells[:3]:
        core_indices.extend(shell)
    
    # Try different single atoms from shell 3 onwards
    outer_indices = []
    for shell in shells[3:]:
        outer_indices.extend(shell)
    
    # Try each outer atom as the 38th
    for idx in outer_indices[:10]:
        candidate_indices = core_indices + [idx]
        candidate = [fcc_points[i][:3] for i in candidate_indices]
        candidates.append(np.array(candidate, dtype=float))
    
    return candidates

def make_fcc_with_noise(n_atoms=38, noise_scale=0.05, seed=42):
    """Generate FCC cluster with small noise to break symmetry."""
    base = make_fcc_truncated_octahedron(n_atoms)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_scale, base.shape)
    return base + noise

def make_icosahedral(n_atoms=38):
    """Generate icosahedral cluster (13 + 12 + 13 shells)."""
    # Icosahedral vertices: (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1)
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    # Scale so nearest neighbor distance ~1
    # Distance from center to vertex = sqrt(1 + phi^2)
    dist_to_vertex = math.sqrt(1.0 + phi * phi)
    scale = 1.0 / (2.0 * dist_to_vertex)  # NN distance = 1
    
    verts = []
    for x in [0, 1, phi]:
        for y in [0, 1, phi]:
            for z in [0, 1, phi]:
                if x == 0 and y == 0:
                    continue
                if x == 0 and z == 0:
                    continue
                if y == 0 and z == 0:
                    continue
                # Only take permutations where exactly one is 0
                if (x == 0) != (y == 0) and (x == 0) != (z == 0) and (y == 0) != (z == 0):
                    for sx in [1, -1]:
                        for sy in [1, -1]:
                            for sz in [1, -1]:
                                verts.append((sx * x * scale, sy * y * scale, sz * z * scale))
    
    # Remove duplicates
    unique_verts = []
    seen = set()
    for v in verts:
        key = tuple(round(c, 10) for c in v)
        if key not in seen:
            seen.add(key)
            unique_verts.append(v)
    
    # Shell 0: origin
    points = [(0.0, 0.0, 0.0)]
    
    # Shell 1: 12 icosahedral vertices
    for v in unique_verts[:12]:
        points.append(v)
    
    # Shell 2: 12 more (midpoints of edges, scaled)
    # For icosahedron, second shell has 12 atoms at distance ~1.7 from center
    # Use edge midpoints of the icosahedron
    edge_mids = []
    ico_verts = np.array(unique_verts[:12])
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(ico_verts[i] - ico_verts[j])
            if 0.9 < d < 1.2:  # Edge length in scaled coords
                mid = (ico_verts[i] + ico_verts[j]) / 2.0
                # Scale to correct distance
                mid_norm = np.linalg.norm(mid)
                if mid_norm > 0.1:
                    target_r = 1.7 * scale  # approximate second shell radius
                    mid_scaled = mid / mid_norm * target_r
                    edge_mids.append(tuple(mid_scaled))
    
    # Deduplicate edge midpoints
    unique_mids = []
    seen_mids = set()
    for m in edge_mids:
        key = tuple(round(c, 8) for c in m)
        if key not in seen_mids:
            seen_mids.add(key)
            unique_mids.append(m)
    
    for m in unique_mids[:13]:
        points.append(m)
    
    # Pad if needed
    while len(points) < n_atoms:
        points.append((0.0, 0.0, 0.0))
    
    return np.array(points[:n_atoms], dtype=float)

def relax(positions, maxiter=300, gtol=1e-8):
    """Relax using L-BFGS-B with analytical gradient."""
    def energy_and_grad_func(pos_flat):
        pos = pos_flat.reshape(N, 3)
        e, g = lj_energy_and_grad(pos)
        return e, g
    
    x0 = positions.flatten()
    
    try:
        result = minimize(energy_and_grad_func, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 50, 'ftol': 1e-12})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def polish(positions, maxiter=500, gtol=1e-10):
    """High-precision final polish pass with analytical gradient."""
    def energy_and_grad_func(pos_flat):
        pos = pos_flat.reshape(N, 3)
        e, g = lj_energy_and_grad(pos)
        return e, g
    
    x0 = positions.flatten()
    
    try:
        result = minimize(energy_and_grad_func, x0, method='L-BFGS-B', jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100, 'ftol': 1e-15})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def basin_hopping(positions, n_iter=10, step_size=0.5, maxiter=100, temperature=0.05, seed=123):
    """Perform basin hopping optimization with analytical gradient."""
    def energy_and_grad_func(pos_flat):
        pos = pos_flat.reshape(N, 3)
        return lj_energy_and_grad(pos)
    
    x0 = positions.flatten()
    current_energy, _ = lj_energy_and_grad(x0.reshape(N, 3))
    best_energy = current_energy
    best_x = x0.copy()
    
    rng = np.random.default_rng(seed)
    
    for i in range(n_iter):
        if time.time() - START_TIME > DEADLINE - 15:
            break
        
        # Perturb with adaptive step size
        actual_step = step_size * (1.0 + 0.2 * math.sin(i * 0.5))
        noise = rng.normal(0, actual_step, x0.shape)
        perturbed = x0 + noise
        
        # Relax
        try:
            result = minimize(energy_and_grad_func, perturbed, method='L-BFGS-B', jac=True,
                            options={'maxiter': maxiter, 'gtol': 1e-7, 'maxls': 50, 'ftol': 1e-12})
            new_x = result.x
            new_energy, _ = lj_energy_and_grad(new_x.reshape(N, 3))
        except Exception:
            continue
        
        # Metropolis acceptance
        if new_energy < current_energy:
            x0 = new_x
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_x = new_x.copy()
        else:
            delta = new_energy - current_energy
            if delta < 0:
                accept = 1.0
            else:
                accept = math.exp(-delta / temperature)
            if rng.uniform() < accept:
                x0 = new_x
                current_energy = new_energy
    
    return best_x.reshape(N, 3), best_energy

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_energy = float('inf')
    best_positions = None
    
    candidates = []
    
    # 1. FCC Truncated Octahedron variants with different outer-shell selections
    try:
        fcc_candidates = _get_fcc_shell_structure(N)
        for i, fcc in enumerate(fcc_candidates):
            if check_validity(fcc):
                candidates.append((f'fcc_variant_{i}', fcc))
    except Exception:
        pass
    
    # 2. Original FCC as fallback
    try:
        fcc = make_fcc_truncated_octahedron(N)
        if check_validity(fcc):
            candidates.append(('fcc_original', fcc))
    except Exception:
        pass
        
    # 3. FCC with small noise
    for seed in [42, 123, 456, 789, 101]:
        try:
            fcc_noisy = make_fcc_with_noise(N, noise_scale=0.05, seed=seed)
            if check_validity(fcc_noisy):
                candidates.append((f'fcc_noise_{seed}', fcc_noisy))
        except Exception:
            pass
        
    # 4. FCC with larger noise
    for seed in [202, 303]:
        try:
            fcc_noisy = make_fcc_with_noise(N, noise_scale=0.12, seed=seed)
            if check_validity(fcc_noisy):
                candidates.append((f'fcc_noise_large_{seed}', fcc_noisy))
        except Exception:
            pass
    
    # 5. Icosahedral structure (diverse candidate)
    try:
        ico = make_icosahedral(N)
        if check_validity(ico):
            candidates.append(('icosahedral', ico))
    except Exception:
        pass
    
    # Relax all candidates
    for name, cand in candidates:
        if time.time() - START_TIME > DEADLINE - 30:
            break
            
        try:
            relaxed = relax(cand, maxiter=200, gtol=1e-7)
            if check_validity(relaxed):
                energy = lj_energy_vectorized(relaxed)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed
        except Exception:
            pass
    
    # Multiple basin hopping passes with varied parameters
    bh_configs = [
        {'n_iter': 30, 'step_size': 0.25, 'maxiter': 80, 'temperature': 0.02, 'seed': 123},
        {'n_iter': 25, 'step_size': 0.4, 'maxiter': 70, 'temperature': 0.05, 'seed': 456},
        {'n_iter': 20, 'step_size': 0.6, 'maxiter': 60, 'temperature': 0.1, 'seed': 789},
    ]
    
    for cfg in bh_configs:
        if best_positions is None or time.time() - START_TIME > DEADLINE - 20:
            break
        try:
            bh_result, bh_energy = basin_hopping(best_positions, **cfg)
            if check_validity(bh_result) and bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_result
        except Exception:
            pass
    
    # Final high-precision polish
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 10:
        try:
            polished = polish(best_positions, maxiter=800, gtol=1e-12)
            if check_validity(polished):
                energy = lj_energy_vectorized(polished)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = polished
        except Exception:
            pass
    
    # Final check and return
    if best_positions is None:
        try:
            best_positions = make_fcc_truncated_octahedron(N)
            if not check_validity(best_positions):
                side = int(np.ceil(N ** (1/3)))
                pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
                pts.sort(key=lambda p: (p[0]-(side-1)/2)**2 + (p[1]-(side-1)/2)**2 + (p[2]-(side-1)/2)**2)
                best_positions = np.array(pts[:N], dtype=float) * 1.2
        except Exception:
            best_positions = np.zeros((N, 3))
            for i in range(N):
                best_positions[i] = [i*0.6, (i%3)*0.6, (i//3)%3*0.6]

    return best_positions

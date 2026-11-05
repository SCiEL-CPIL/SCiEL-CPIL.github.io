# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_GLOBAL_START_TIME = None
_DEADLINE = 315.0  # safety margin before 330s kill

def _get_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return _DEADLINE - (time.time() - _GLOBAL_START_TIME)

def _morse_energy(x, rho):
    """Compute Morse energy for a set of positions (vectorized)."""
    n = x.shape[0]
    if n < 2:
        return 0.0
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    er = rho * (1.0 - dist)
    er = np.clip(er, -500, 500)
    exp_er = np.exp(er)
    energy = np.sum(exp_er * (exp_er - 2.0)) / 2.0
    return float(energy)

def _min_distance(x):
    """Compute minimum distance between any pair of atoms (vectorized)."""
    n = x.shape[0]
    if n < 2:
        return np.inf
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    return float(np.min(dist))

def _is_valid(x):
    """Check if the configuration is valid (atoms at least 0.5 apart)."""
    return _min_distance(x) >= 0.5

def _icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron, normalized to unit length."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    return verts

def _make_icosahedron_shell(n_atoms, radius, center=np.zeros(3)):
    """Place n_atoms on an icosahedral shell of given radius centered at center."""
    verts = _icosahedron_vertices()
    if n_atoms <= 12:
        # Use first n_atoms vertices
        selected = verts[:n_atoms]
    elif n_atoms <= 42:
        # Use all 12 vertices plus distribute remaining on edges/faces
        selected = [verts]
        remaining = n_atoms - 12
        # Add points on edges (30 edges) and faces (20 faces)
        edges = []
        # Find edges: pairs of vertices at distance ~0.756 (for unit icosahedron)
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if d < 0.8:
                    edges.append((i, j))
        # Distribute remaining on edges
        extra = []
        for k in range(remaining):
            ei = k % len(edges)
            i, j = edges[ei]
            t = (k // len(edges) + 0.5) / (remaining // len(edges) + 1)
            extra.append(verts[i] * (1-t) + verts[j] * t)
        selected = np.vstack([verts] + [np.array(e) for e in extra])
    else:
        # For larger shells, use golden spiral
        selected = [verts]
        remaining = n_atoms - 12
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        extra = []
        for k in range(remaining):
            z = 1.0 - 2.0 * (k + 0.5) / remaining
            r = np.sqrt(1.0 - z*z)
            theta = golden_angle * k
            extra.append(np.array([r*np.cos(theta), r*np.sin(theta), z]))
        selected = np.vstack([verts] + [np.array(e) for e in extra])
    
    return selected * radius + center

def _generate_icosahedral_cluster(n, seed=42):
    """
    Generate an icosahedral cluster with concentric shells.
    Icosahedral shells: 1, 12, 42, 120, ...
    """
    rng = np.random.default_rng(seed)
    positions = [np.array([0.0, 0.0, 0.0])]
    remaining = n - 1
    
    # Shell 1: 12 atoms on icosahedron at radius 1.0
    if remaining >= 12:
        positions.extend(_make_icosahedron_shell(12, 1.0))
        remaining -= 12
    
    # Shell 2: up to 42 atoms at radius ~1.75
    if remaining >= 1:
        n_shell2 = min(remaining, 42)
        positions.extend(_make_icosahedron_shell(n_shell2, 1.75))
        remaining -= n_shell2
    
    # Additional shells if needed
    shell_idx = 3
    while remaining > 0:
        shell_radius = 1.0 + (shell_idx - 1) * 0.75
        max_in_shell = min(remaining, 4 * shell_idx * (shell_idx + 1) + 2)
        positions.extend(_make_icosahedron_shell(max_in_shell, shell_radius))
        remaining -= max_in_shell
        shell_idx += 1
    
    return np.array(positions[:n])

def _generate_fcc_cluster(n, seed=42):
    """
    Generate an FCC-like cluster.
    Optimal for long-range Morse potentials (low rho).
    """
    rng = np.random.default_rng(seed)
    
    radius = 1.2 * (n ** (1.0/3.0)) + 1.0
    points = []
    
    # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    a = 1.0 / np.sqrt(2.0)
    basis = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
    
    max_coord = int(radius / a) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in basis:
                    p = a * np.array([i + dx, j + dy, k + dz])
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
    
    points.sort(key=lambda p: np.linalg.norm(p))
    
    if len(points) >= n:
        return np.array(points[:n])
    else:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                points.append(p)
            else:
                points.append(p * 0.5)
        return np.array(points)

def _generate_bcc_cluster(n, seed=42):
    """Generate a BCC-like cluster."""
    rng = np.random.default_rng(seed)
    
    radius = 1.3 * (n ** (1.0/3.0)) + 1.0
    points = []
    
    a = 1.0 / np.sqrt(3.0)
    basis = [(0, 0, 0), (0.5, 0.5, 0.5)]
    
    max_coord = int(radius / a) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in basis:
                    p = a * np.array([i + dx, j + dy, k + dz])
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
    
    points.sort(key=lambda p: np.linalg.norm(p))
    
    if len(points) >= n:
        return np.array(points[:n])
    else:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
        return np.array(points)

def _generate_hcp_cluster(n, seed=42):
    """Generate an HCP-like cluster."""
    rng = np.random.default_rng(seed)
    
    radius = 1.3 * (n ** (1.0/3.0)) + 1.0
    points = []
    
    a = 1.0
    c = 1.633 * a
    
    max_coord = int(radius / a) + 2
    for layer in range(-int(radius/c), int(radius/c) + 1):
        z = layer * c
        offset = 0.5 if layer % 2 != 0 else 0.0
        for i in range(-max_coord, max_coord + 1):
            for j in range(-max_coord, max_coord + 1):
                x = a * (i + offset + 0.5 * (j % 2))
                y = a * j * np.sqrt(3.0)/2
                p = np.array([x, y, z])
                if np.linalg.norm(p) <= radius:
                    points.append(p)
    
    points.sort(key=lambda p: np.linalg.norm(p))
    
    if len(points) >= n:
        return np.array(points[:n])
    else:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
        return np.array(points)

def _generate_random_cluster(n, seed=42, min_dist=0.6):
    """Generate a random cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    
    radius = 1.8 * (n ** (1.0/3.0)) + 0.5
    points = []
    attempts = 0
    max_attempts = 50000
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)

def _relax_and_validate(positions, potential, rho):
    """Relax positions and check validity."""
    n = positions.shape[0]
    try:
        energy, relaxed = relax(positions, potential, rho)
        if _min_distance(relaxed) >= 0.5:
            return float(energy), relaxed
        else:
            return float('inf'), None
    except Exception:
        return float('inf'), None

def _polish(positions, rho, max_iter=200):
    """
    Perform a high-precision local minimization using L-BFGS-B.
    """
    n = positions.shape[0]
    x_flat = positions.flatten()
    
    def objective(x):
        pos = x.reshape(-1, 3)
        if _min_distance(pos) < 0.5:
            return 1e10
        return _morse_energy(pos, rho)
    
    def gradient(x):
        pos = x.reshape(-1, 3)
        if _min_distance(pos) < 0.5:
            return np.zeros_like(x)
        grad_3d = morse_energy_gradient(pos, rho)
        return np.asarray(grad_3d).flatten()
    
    try:
        res = minimize(objective, x_flat, jac=gradient, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
        if res.success or res.fun < 1e10:
            pos_res = res.x.reshape(-1, 3)
            if _min_distance(pos_res) >= 0.5:
                return float(res.fun), pos_res
    except Exception:
        pass
    
    return _morse_energy(positions, rho), positions

def _basin_hop(start_positions, potential, rho, max_iterations=100, initial_step=0.5, initial_temp=3.0):
    """
    Perform basin hopping optimization.
    Returns (best_energy, best_positions).
    """
    n = start_positions.shape[0]
    current_energy, current_positions = _relax_and_validate(start_positions, potential, rho)
    
    if current_positions is None:
        perturbed = start_positions + np.random.normal(0, 0.1, start_positions.shape)
        current_energy, current_positions = _relax_and_validate(perturbed, potential, rho)
        if current_positions is None:
            return float('inf'), None
    
    best_energy = current_energy
    best_positions = current_positions.copy()
    
    step_size = initial_step
    temp = initial_temp
    
    for it in range(max_iterations):
        if _get_remaining_time() < 3.0:
            break
        
        noise = np.random.normal(0, step_size, current_positions.shape)
        new_positions = current_positions + noise
        
        new_energy, new_relaxed = _relax_and_validate(new_positions, potential, rho)
        
        if new_relaxed is None:
            noise = np.random.normal(0, step_size * 0.5, current_positions.shape)
            new_positions = current_positions + noise
            new_energy, new_relaxed = _relax_and_validate(new_positions, potential, rho)
        
        if new_relaxed is None:
            continue
        
        delta_e = new_energy - current_energy
        if delta_e < 0 or (temp > 1e-8 and np.random.rand() < math.exp(-delta_e / temp)):
            current_energy = new_energy
            current_positions = new_relaxed
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = current_positions.copy()
        
        temp *= 0.97
        step_size *= 0.99
        
        if it > 0 and it % 30 == 0:
            step_size *= 1.3
    
    return best_energy, best_positions

def _generate_superatom_clusters(n, rho, seed=42):
    """
    Generate clusters using super-atom (rigid sub-cluster) approach.
    For high rho: icosahedral sub-clusters
    For low rho: FCC sub-clusters
    """
    rng = np.random.default_rng(seed)
    
    if rho >= 8:
        sub_cluster_size = 13
        sub_cluster_gen = _generate_icosahedral_cluster
    elif rho >= 5:
        sub_cluster_size = 7
        sub_cluster_gen = _generate_bcc_cluster
    else:
        sub_cluster_size = 4
        sub_cluster_gen = _generate_fcc_cluster
    
    n_sub = n // sub_cluster_size
    remainder = n % sub_cluster_size
    
    all_positions = []
    
    if sub_cluster_size > 0:
        sc = sub_cluster_gen(sub_cluster_size, seed=seed)
        all_positions.append(sc)
    
    for i in range(1, n_sub):
        sc = sub_cluster_gen(sub_cluster_size, seed=seed + i)
        r = 2.0 + i * 0.5
        u = rng.uniform(-1, 1)
        theta = rng.uniform(0, 2 * np.pi)
        center = np.array([r * np.sqrt(1 - u**2) * np.cos(theta),
                          r * np.sqrt(1 - u**2) * np.sin(theta),
                          r * u])
        all_positions.append(sc + center)
    
    if remainder > 0:
        sc_rem = sub_cluster_gen(remainder, seed=seed + n_sub)
        u = rng.uniform(-1, 1)
        theta = rng.uniform(0, 2 * np.pi)
        r = 2.5
        center = np.array([r * np.sqrt(1 - u**2) * np.cos(theta),
                          r * np.sqrt(1 - u**2) * np.sin(theta),
                          r * u])
        all_positions.append(sc_rem + center)
    
    positions = np.vstack(all_positions)
    
    return positions

def construct(instance):
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 6))
    potential = instance["potential"]
    
    if rho is None:
        rho = 6
    
    best_energy = float('inf')
    best_positions = None
    
    initial_guesses = []
    
    # Choose initial guesses based on rho
    if rho <= 4:
        for seed in [42, 1, 2, 3]:
            try:
                initial_guesses.append(_generate_fcc_cluster(n, seed=seed))
            except Exception:
                pass
        for seed in [4, 5]:
            try:
                initial_guesses.append(_generate_hcp_cluster(n, seed=seed))
            except Exception:
                pass
        for seed in [10, 11, 12]:
            try:
                initial_guesses.append(_generate_random_cluster(n, seed=seed))
            except Exception:
                pass
    elif rho <= 7:
        for seed in [42, 1, 2]:
            try:
                initial_guesses.append(_generate_fcc_cluster(n, seed=seed))
            except Exception:
                pass
        for seed in [42, 1, 2]:
            try:
                initial_guesses.append(_generate_icosahedral_cluster(n, seed=seed))
            except Exception:
                pass
        for seed in [3, 4, 5]:
            try:
                initial_guesses.append(_generate_bcc_cluster(n, seed=seed))
            except Exception:
                pass
        for seed in [10, 11, 12, 13]:
            try:
                initial_guesses.append(_generate_random_cluster(n, seed=seed))
            except Exception:
                pass
    else:
        for seed in [42, 1, 2, 3]:
            try:
                initial_guesses.append(_generate_icosahedral_cluster(n, seed=seed))
            except Exception:
                pass
        for seed in [4, 5]:
            try:
                initial_guesses.append(_generate_bcc_cluster(n, seed=seed))
            except Exception:
                pass
        for seed in [10, 11, 12, 13, 14]:
            try:
                initial_guesses.append(_generate_random_cluster(n, seed=seed))
            except Exception:
                pass
    
    for seed in [100, 101, 102]:
        try:
            initial_guesses.append(_generate_superatom_clusters(n, rho, seed=seed))
        except Exception:
            pass
    
    # Relax all initial guesses
    relaxed_results = []
    for guess in initial_guesses:
        if _get_remaining_time() < 8.0:
            break
        
        energy, positions = _relax_and_validate(guess, potential, rho)
        if positions is not None and energy < float('inf'):
            relaxed_results.append((energy, positions))
    
    relaxed_results.sort(key=lambda x: x[0])
    
    if relaxed_results:
        best_energy = relaxed_results[0][0]
        best_positions = relaxed_results[0][1].copy()
    
    # Basin hopping from multiple starting points
    num_bh_starts = min(len(relaxed_results), 5)
    
    if rho <= 4:
        initial_step = 0.6
        initial_temp = 5.0
    elif rho <= 7:
        initial_step = 0.4
        initial_temp = 3.0
    else:
        initial_step = 0.3
        initial_temp = 2.0
    
    if n <= 40:
        max_iter = 120
    elif n <= 60:
        max_iter = 80
    else:
        max_iter = 50
    
    for idx in range(num_bh_starts):
        if _get_remaining_time() < 10.0:
            break
        
        start_energy, start_pos = relaxed_results[idx]
        
        bh_energy, bh_positions = _basin_hop(
            start_pos, potential, rho,
            max_iterations=max_iter,
            initial_step=initial_step,
            initial_temp=initial_temp
        )
        
        if bh_positions is not None and bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_positions.copy()
        
        if _get_remaining_time() > 15.0 and idx == 0:
            bh_energy2, bh_positions2 = _basin_hop(
                start_pos, potential, rho,
                max_iterations=max_iter // 2,
                initial_step=initial_step * 1.5,
                initial_temp=initial_temp * 1.5
            )
            if bh_positions2 is not None and bh_energy2 < best_energy:
                best_energy = bh_energy2
                best_positions = bh_positions2.copy()
    
    # Final polish with high precision
    if best_positions is not None and _get_remaining_time() > 5.0:
        polished_energy, polished_pos = _polish(best_positions, rho, max_iter=300)
        if polished_pos is not None and polished_energy < best_energy:
            best_energy = polished_energy
            best_positions = polished_pos
    
    # Safety: if no valid positions found, generate a fallback
    if best_positions is None or not _is_valid(best_positions):
        best_positions = _generate_icosahedral_cluster(n, seed=0)
        if not _is_valid(best_positions):
            best_positions = _generate_fcc_cluster(n, seed=0)
        if not _is_valid(best_positions):
            best_positions = _generate_random_cluster(n, seed=99)
    
    return best_positions
# EVOLVE-BLOCK-END

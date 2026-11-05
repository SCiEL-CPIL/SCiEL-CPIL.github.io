# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_GLOBAL_START_TIME = None
_DEADLINE = 315.0  # 315s wall clock limit (evaluator kills at 330s)

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

def _generate_icosahedral_cluster(n, seed=42):
    """
    Generate an icosahedral cluster with concentric shells.
    Icosahedral shells: 1, 12, 42, 120, ...
    This is optimal for short-range Morse potentials (high rho).
    """
    rng = np.random.default_rng(seed)
    
    # Icosahedron vertices (12 vertices)
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    norm = np.linalg.norm(vertices[0])
    vertices = vertices / norm
    
    positions = [np.array([0.0, 0.0, 0.0])]
    remaining = n - 1
    
    # Shell 1: 12 atoms on icosahedron
    shell1_radius = 1.0
    if remaining >= 12:
        for v in vertices:
            positions.append(v * shell1_radius)
        remaining -= 12
    
    # Shell 2: 42 atoms (truncated icosahedron / fullerenes)
    if remaining >= 1:
        # Generate points on a sphere for second shell
        shell2_radius = 2.0
        # Use golden spiral for uniform distribution
        n_in_shell = min(remaining, 42)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n_in_shell):
            z = 1.0 - 2.0 * i / (n_in_shell - 1) if n_in_shell > 1 else 0.0
            r = np.sqrt(1.0 - z*z)
            theta = golden_angle * i
            p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * shell2_radius
            positions.append(p)
        remaining -= n_in_shell
    
    # Add more shells if needed
    shell_idx = 3
    while remaining > 0:
        shell_radius = shell_idx * 1.0
        max_in_shell = min(remaining, 4 * shell_idx * (shell_idx + 1) + 2)
        for _ in range(max_in_shell):
            u = rng.uniform(-1, 1)
            theta = rng.uniform(0, 2 * np.pi)
            x = shell_radius * np.sqrt(1 - u**2) * cos(theta)
            y = shell_radius * np.sqrt(1 - u**2) * sin(theta)
            z = shell_radius * u
            positions.append(np.array([x, y, z]))
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
    
    # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) in units of lattice constant
    # For pair distance 1, lattice constant a = 1/sqrt(2)
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
    
    # Sort by distance from center
    points.sort(key=lambda p: np.linalg.norm(p))
    
    if len(points) >= n:
        return np.array(points[:n])
    else:
        # Fill in with random points
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
    
    # BCC basis: (0,0,0), (0.5,0.5,0.5)
    a = 1.0 / np.sqrt(3.0)  # for pair distance 1
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
    
    # HCP: ABAB stacking
    a = 1.0  # in-plane spacing
    c = 1.633 * a  # c/a ratio for HCP
    
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
    max_attempts = 100000
    
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
            # Try smaller perturbation
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
        
        # Annealing
        temp *= 0.97
        step_size *= 0.99
        
        # Occasionally increase step size to escape local minima
        if it > 0 and it % 30 == 0:
            step_size *= 1.3
    
    return best_energy, best_positions

def _rigid_motif_search(n, rho, time_budget=15.0):
    """
    Rigid motif decomposition search:
    1. Build small rigid motifs (5-15 atoms) from known optimal substructures
    2. Treat each motif as a super-atom with position + orientation DOFs
    3. Optimize relative positions/orientations via L-BFGS-B
    4. This reduces search dimensionality from 3n to ~7k (k=2-3 motifs)
    """
    start_time = time.time()
    
    # Determine number of motifs based on n
    if n <= 40:
        num_motifs = 3
        motif_size = n // num_motifs
    elif n <= 60:
        num_motifs = 4
        motif_size = n // num_motifs
    else:
        num_motifs = 5
        motif_size = n // num_motifs
    
    # Ensure motif_size is reasonable (5-15)
    motif_size = max(5, min(15, motif_size))
    
    # Generate rigid motifs
    motifs = []
    if rho <= 4:
        # Long range: FCC-like motifs
        for i in range(num_motifs):
            m = _generate_fcc_cluster(motif_size, seed=100 + i)
            motifs.append(m)
    elif rho <= 7:
        # Medium range: mix
        for i in range(num_motifs):
            if i % 2 == 0:
                m = _generate_fcc_cluster(motif_size, seed=100 + i)
            else:
                m = _generate_icosahedral_cluster(motif_size, seed=100 + i)
            motifs.append(m)
    else:
        # Short range: icosahedral motifs
        for i in range(num_motifs):
            m = _generate_icosahedral_cluster(motif_size, seed=100 + i)
            motifs.append(m)
    
    # Trim/pad to exact n
    total_atoms = sum(len(m) for m in motifs)
    if total_atoms > n:
        # Remove excess atoms from last motif
        excess = total_atoms - n
        motifs[-1] = motifs[-1][:-excess]
    elif total_atoms < n:
        # Add random atoms to last motif
        deficit = n - total_atoms
        extra = _generate_random_cluster(deficit, seed=200)
        motifs[-1] = np.vstack([motifs[-1], extra])
    
    # Each motif has 3 (position) + 3 (orientation) = 6 DOFs
    # Total DOFs = 6 * num_motifs
    total_dofs = 6 * num_motifs
    
    # Initial positions: place motifs in a rough cluster
    initial_params = np.zeros(total_dofs)
    for i, m in enumerate(motifs):
        # Position: spread out on a sphere
        angle = 2 * np.pi * i / num_motifs
        radius = 2.0
        initial_params[i*6:i*6+3] = [radius * cos(angle), radius * sin(angle), 0]
        # Orientation: identity (zero angles)
    
    def _apply_motif_transform(motif, params):
        """Apply position and orientation to a motif."""
        pos = params[:3]
        angles = params[3:6]
        
        # Rotation from Euler angles (XYZ)
        cx, cy, cz = cos(angles[0]), cos(angles[1]), cos(angles[2])
        sx, sy, sz = sin(angles[0]), sin(angles[1]), sin(angles[2])
        
        # Rotation matrix
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cx, -sx, 0], [sx, cx, 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        
        return motif @ R.T + pos
    
    def _assemble_from_params(params):
        """Assemble full configuration from motif parameters."""
        positions = []
        for i in range(num_motifs):
            motif = motifs[i]
            p = params[i*6:(i+1)*6]
            transformed = _apply_motif_transform(motif, p)
            positions.append(transformed)
        return np.vstack(positions)
    
    def _objective(params):
        """Energy objective for motif parameters."""
        if time.time() - start_time > time_budget:
            return 1e10
        config = _assemble_from_params(params)
        if _min_distance(config) < 0.5:
            return 1e10
        return _morse_energy(config, rho)
    
    def _gradient(params):
        """Gradient via finite differences (expensive but robust)."""
        if time.time() - start_time > time_budget:
            return np.zeros_like(params)
        
        grad = np.zeros_like(params)
        h = 1e-5
        f0 = _objective(params)
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += h
            f_plus = _objective(params_plus)
            grad[i] = (f_plus - f0) / h
            params_plus[i] -= 2 * h
            f_minus = _objective(params_plus)
            grad[i] = (f_plus - f_minus) / (2 * h)
        
        return grad
    
    # Optimize using L-BFGS-B
    try:
        res = minimize(_objective, initial_params, method='L-BFGS-B',
                       options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.fun < 1e10:
            config = _assemble_from_params(res.x)
            if _min_distance(config) >= 0.5:
                energy = _morse_energy(config, rho)
                return energy, config
    except Exception:
        pass
    
    return float('inf'), None

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
    
    # Phase 1: Generate diverse initial guesses
    initial_guesses = []
    
    if rho <= 4:
        # Long range: FCC-like structures
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
        # Medium range: mix of FCC and icosahedral
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
        # Short range: icosahedral structures
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
    
    # Phase 2: Relax all initial guesses
    relaxed_results = []
    for guess in initial_guesses:
        if _get_remaining_time() < 10.0:
            break
        
        energy, positions = _relax_and_validate(guess, potential, rho)
        if positions is not None and energy < float('inf'):
            relaxed_results.append((energy, positions))
    
    relaxed_results.sort(key=lambda x: x[0])
    
    if relaxed_results:
        best_energy = relaxed_results[0][0]
        best_positions = relaxed_results[0][1].copy()
    
    # Phase 3: Rigid motif search (structural change)
    if _get_remaining_time() > 20.0:
        motif_energy, motif_positions = _rigid_motif_search(n, rho, time_budget=15.0)
        if motif_positions is not None and motif_energy < best_energy:
            best_energy = motif_energy
            best_positions = motif_positions.copy()
    
    # Phase 4: Basin hopping from multiple starting points
    num_bh_starts = min(len(relaxed_results), 5)
    
    # Parameters based on rho and n
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
        
        # If we have time, do a second basin hop with different parameters
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
    
    # Phase 5: Final polish with high precision
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

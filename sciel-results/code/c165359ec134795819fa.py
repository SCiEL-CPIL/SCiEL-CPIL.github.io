# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_GLOBAL_START_TIME = None
_DEADLINE = 325.0  # 325s wall clock limit (evaluator kills at 330s)

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
    return energy

def _min_distance(x):
    """Compute minimum distance between any pair of atoms (vectorized)."""
    n = x.shape[0]
    if n < 2:
        return np.inf
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    return np.min(dist)

def _is_valid(x):
    """Check if the configuration is valid (atoms at least 0.5 apart)."""
    return _min_distance(x) >= 0.5

def _rotation_matrix(angle, axis):
    """Create a rotation matrix."""
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    x, y, z = axis
    R = np.array([
        [c + x*x*(1-c),   x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s, c + y*y*(1-c),   y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)]
    ])
    return R

def _generate_icosahedron(center, radius, seed=42):
    """Generate 13-atom icosahedral cluster (1 center + 12 vertices)."""
    rng = np.random.default_rng(seed)
    phi = (1 + np.sqrt(5)) / 2
    scale = radius / np.sqrt(1 + phi*phi)
    
    # Icosahedron vertices
    verts = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                if sx*sy*sz == 1 or sx*sy*sz == -1:  # All 8 combinations
                    pass
    # Standard icosahedron vertices
    verts = [
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    verts = np.array(verts) * scale
    
    positions = [center.copy()]
    for v in verts:
        positions.append(center + v)
    
    return np.array(positions)

def _generate_decahedron(center, radius, seed=42):
    """Generate 13-atom decahedral cluster (1 center + 12 in 3-fold symmetric shells)."""
    rng = np.random.default_rng(seed)
    positions = [center.copy()]
    
    # First shell: 6 atoms in octahedral arrangement
    r1 = radius * 0.6
    octahedral = [
        [r1, 0, 0], [-r1, 0, 0],
        [0, r1, 0], [0, -r1, 0],
        [0, 0, r1], [0, 0, -r1]
    ]
    for p in octahedral:
        positions.append(center + np.array(p))
    
    # Second shell: 6 atoms in tetrahedral-ish arrangement
    r2 = radius * 0.85
    tetra = [
        [r2, r2, r2], [r2, -r2, -r2],
        [-r2, r2, -r2], [-r2, -r2, r2],
        [r2, r2, -r2], [-r2, -r2, -r2]
    ]
    for p in tetra:
        positions.append(center + np.array(p))
    
    return np.array(positions)

def _generate_fcc_chunk(center, radius, n_atoms, seed=42):
    """Generate FCC-like chunk with specified number of atoms."""
    rng = np.random.default_rng(seed)
    basis = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    max_coord = int(radius) + 2
    points = []
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in basis:
                    p = np.array([i + dx, j + dy, k + dz])
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
    
    points.sort(key=lambda p: np.linalg.norm(p))
    
    if len(points) >= n_atoms:
        selected = np.array(points[:n_atoms]) + center
    else:
        selected = np.array(points) + center
        while len(selected) < n_atoms:
            p = rng.uniform(-radius, radius, 3)
            selected = np.vstack([selected, [center + p]])
    
    return selected[:n_atoms]

def _build_supercell_cluster(n, seed=42):
    """
    Build cluster using super-atom approach:
    Place icosahedral/decahedral sub-clusters and optimize their positions.
    """
    rng = np.random.default_rng(seed)
    
    # Determine number of sub-clusters
    if n <= 13:
        # Single icosahedron or decahedron
        if n == 13:
            return _generate_icosahedron(np.zeros(3), 1.0, seed=seed)
        else:
            # Use smaller cluster
            ico = _generate_icosahedron(np.zeros(3), 1.0, seed=seed)
            return ico[:n]
    
    # For larger n, use multiple sub-clusters
    num_sub = max(1, n // 13)
    remaining = n % 13
    if remaining == 0:
        num_sub -= 1
        remaining = 13
    
    sub_clusters = []
    used_atoms = 0
    
    for i in range(num_sub):
        # Alternate between icosahedron and decahedron
        if i % 2 == 0:
            sub = _generate_icosahedron(np.zeros(3), 1.0, seed=seed + i)
        else:
            sub = _generate_decahedron(np.zeros(3), 1.0, seed=seed + i)
        sub_clusters.append(sub)
        used_atoms += 13
    
    # Handle remaining atoms
    if remaining > 0:
        if remaining <= 13:
            ico = _generate_icosahedron(np.zeros(3), 1.0, seed=seed + 100)
            sub_clusters.append(ico[:remaining])
        used_atoms += remaining
    
    # Place sub-clusters in a rough arrangement
    # Use a grid-like pattern
    positions = []
    center = np.zeros(3)
    
    # Place first sub-cluster at center
    if sub_clusters:
        positions.append(sub_clusters[0])
    
    # Place remaining sub-clusters around
    for i in range(1, len(sub_clusters)):
        # Distribute on a sphere
        angle_theta = i * 2 * np.pi / max(1, len(sub_clusters) - 1)
        angle_phi = np.pi / 3
        dist = 3.5
        offset = np.array([
            dist * np.sin(angle_phi) * np.cos(angle_theta),
            dist * np.sin(angle_phi) * np.sin(angle_theta),
            dist * np.cos(angle_phi)
        ])
        positions.append(sub_clusters[i] + offset)
    
    all_positions = np.vstack(positions)
    
    # Trim to exact n if needed
    if len(all_positions) > n:
        all_positions = all_positions[:n]
    elif len(all_positions) < n:
        # Add random atoms
        radius = 3.0
        while len(all_positions) < n:
            p = rng.uniform(-radius, radius, 3)
            all_positions = np.vstack([all_positions, [p]])
    
    return all_positions

def _relax_and_validate(positions, potential, rho):
    """Relax positions and check validity."""
    n = positions.shape[0]
    try:
        energy, relaxed = relax(positions, potential, rho)
        if _min_distance(relaxed) >= 0.5:
            return energy, relaxed
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
        return grad_3d.flatten()
    
    try:
        res = minimize(objective, x_flat, jac=gradient, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.success or res.fun < 1e10:
            pos_res = res.x.reshape(-1, 3)
            if _min_distance(pos_res) >= 0.5:
                return res.fun, pos_res
    except Exception:
        pass
    
    return _morse_energy(positions, rho), positions

def _basin_hop(start_positions, potential, rho, max_iterations=100, initial_step=0.5, initial_temp=2.0):
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
        if _get_remaining_time() < 5.0:
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
        if delta_e < 0 or (temp > 1e-6 and np.random.rand() < math.exp(-delta_e / temp)):
            current_energy = new_energy
            current_positions = new_relaxed
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = current_positions.copy()
        
        temp *= 0.95
        step_size *= 0.98
        
        if it > 0 and it % 20 == 0:
            step_size *= 1.2
    
    return best_energy, best_positions

def _supercell_optimize(sub_clusters, n_total, rho, max_iter=50):
    """
    Optimize relative positions and orientations of sub-clusters.
    This is a lower-dimensional optimization problem.
    """
    num_sub = len(sub_clusters)
    if num_sub == 1:
        # Just relax the single cluster
        return _relax_and_validate(sub_clusters[0], 'morse', rho)
    
    # Parameters: for each sub-cluster, 3 position + 3 rotation (Euler angles)
    # Total parameters: num_sub * 6
    x0 = np.zeros(num_sub * 6)
    for i in range(num_sub):
        # Initial position: distribute on a ring
        angle = 2 * np.pi * i / num_sub
        x0[i*6:i*6+3] = [2.0 * np.cos(angle), 2.0 * np.sin(angle), 0.0]
        # Initial rotation: identity
        x0[i*6+3:i*6+6] = [0.0, 0.0, 0.0]
    
    def build_positions(params):
        """Build full atom positions from sub-cluster parameters."""
        all_pos = []
        for i in range(num_sub):
            pos = params[i*6:i*6+3]
            rot_euler = params[i*6+3:i*6+6]
            # Convert Euler angles to rotation matrix
            R = _rotation_matrix(rot_euler[0], [1,0,0]) @ \
                _rotation_matrix(rot_euler[1], [0,1,0]) @ \
                _rotation_matrix(rot_euler[2], [0,0,1])
            sub = sub_clusters[i]
            # Apply rotation and translation
            rotated = (R @ sub.T).T + pos
            all_pos.append(rotated)
        return np.vstack(all_pos)
    
    def objective(params):
        pos = build_positions(params)
        if len(pos) != n_total:
            return 1e10
        if _min_distance(pos) < 0.5:
            return 1e10
        return _morse_energy(pos, rho)
    
    def gradient(params):
        # Numerical gradient (approximate)
        eps = 1e-6
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            f_plus = objective(params_plus)
            params_minus = params.copy()
            params_minus[i] -= eps
            f_minus = objective(params_minus)
            grad[i] = (f_plus - f_minus) / (2 * eps)
        return grad
    
    try:
        res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-10})
        if res.fun < 1e10:
            final_pos = build_positions(res.x)
            if _min_distance(final_pos) >= 0.5:
                return res.fun, final_pos
    except Exception:
        pass
    
    # Fallback: just relax the initial arrangement
    initial_pos = build_positions(x0)
    return _relax_and_validate(initial_pos, 'morse', rho)

def construct(instance):
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    potential = instance["potential"]
    
    if rho is None:
        rho = 6
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate diverse initial guesses using super-atom approach
    initial_guesses = []
    
    # Super-atom based clusters
    for seed in [42, 100, 200, 300]:
        try:
            initial_guesses.append(_build_supercell_cluster(n, seed=seed))
        except Exception:
            pass
    
    # Traditional structured clusters
    try:
        ico = _generate_icosahedron(np.zeros(3), 1.0, seed=42)
        if len(ico) == n:
            initial_guesses.append(ico)
        elif len(ico) > n:
            initial_guesses.append(ico[:n])
    except Exception:
        pass
    
    try:
        deca = _generate_decahedron(np.zeros(3), 1.0, seed=42)
        if len(deca) == n:
            initial_guesses.append(deca)
        elif len(deca) > n:
            initial_guesses.append(deca[:n])
    except Exception:
        pass
    
    # FCC chunk
    try:
        fcc = _generate_fcc_chunk(np.zeros(3), 2.5, n, seed=42)
        initial_guesses.append(fcc)
    except Exception:
        pass
    
    # Relax all initial guesses
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
    
    # Perform super-cell optimization from best relaxed result
    if best_positions is not None and _get_remaining_time() > 30.0:
        # Try to decompose into sub-clusters and optimize
        # For simplicity, just do basin hopping with larger steps
        num_bh_starts = min(len(relaxed_results), 4)
        
        for idx in range(num_bh_starts):
            if _get_remaining_time() < 20.0:
                break
            
            start_energy, start_pos = relaxed_results[idx]
            
            if rho <= 3:
                initial_step = 0.8
                initial_temp = 5.0
            elif rho <= 6:
                initial_step = 0.5
                initial_temp = 3.0
            else:
                initial_step = 0.3
                initial_temp = 2.0
            
            if n <= 40:
                max_iter = 100
            elif n <= 60:
                max_iter = 70
            else:
                max_iter = 40
            
            bh_energy, bh_positions = _basin_hop(
                start_pos, potential, rho,
                max_iterations=max_iter,
                initial_step=initial_step,
                initial_temp=initial_temp
            )
            
            if bh_positions is not None and bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_positions.copy()
    
    # Final polish
    if best_positions is not None and _get_remaining_time() > 5.0:
        polished_energy, polished_pos = _polish(best_positions, rho)
        if polished_pos is not None and polished_energy < best_energy:
            best_energy = polished_energy
            best_positions = polished_pos
    
    if best_positions is None:
        # Fallback: simple shell
        rng = np.random.default_rng(0)
        positions = [np.array([0.0, 0.0, 0.0])]
        remaining = n - 1
        shell_index = 1
        while remaining > 0:
            shell_radius = shell_index * 1.1
            max_in_shell = min(remaining, int(4 * shell_index * (shell_index + 1)))
            if max_in_shell <= 0:
                max_in_shell = 1
            for _ in range(max_in_shell):
                u = rng.uniform(-1, 1)
                theta = rng.uniform(0, 2 * np.pi)
                r = shell_radius
                x = r * np.sqrt(1 - u**2) * np.cos(theta)
                y = r * np.sqrt(1 - u**2) * np.sin(theta)
                z = r * u
                positions.append(np.array([x, y, z]))
            remaining -= max_in_shell
            shell_index += 1
        best_positions = np.array(positions[:n])
        if not _is_valid(best_positions):
            best_positions = best_positions * 1.5
    
    return best_positions
# EVOLVE-BLOCK-END

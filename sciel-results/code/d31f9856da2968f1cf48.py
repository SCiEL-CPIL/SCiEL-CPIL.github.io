# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_GLOBAL_START_TIME = None
_DEADLINE = 320.0  # safety margin before 330s kill

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

def _generate_icosahedron(n, seed=42):
    """Generate icosahedral-like cluster for small n."""
    rng = np.random.default_rng(seed)
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts[0])
    verts *= 1.0
    
    positions = [np.array([0.0, 0.0, 0.0])]
    if n <= 13:
        positions.extend(verts[:n-1].tolist())
        return np.array(positions[:n])
    
    # Second shell: octahedron vertices at larger radius
    octa = np.array([
        [1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]
    ]) * 1.8
    remaining = n - len(positions)
    for v in octa:
        if remaining <= 0:
            break
        positions.append(v)
        remaining -= 1
    
    while len(positions) < n:
        u = rng.uniform(-1, 1)
        theta = rng.uniform(0, 2*np.pi)
        r = 2.5
        x = r * np.sqrt(1-u**2) * np.cos(theta)
        y = r * np.sqrt(1-u**2) * np.sin(theta)
        z = r * u
        positions.append(np.array([x,y,z]))
    
    return np.array(positions[:n])

def _generate_shell(n, seed=42):
    """Generate a compact cluster by placing atoms in concentric shells."""
    rng = np.random.default_rng(seed)
    positions = [np.array([0.0, 0.0, 0.0])]
    remaining = n - 1
    shell_index = 1
    while remaining > 0:
        shell_radius = shell_index * 1.05
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
    return np.array(positions[:n])

def _generate_fcc_cluster(n, seed=42):
    """Generate an FCC-like cluster."""
    rng = np.random.default_rng(seed)
    radius = 1.4 * (n ** (1.0/3.0))
    points = []
    basis = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    max_coord = int(radius) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in basis:
                    p = np.array([i + dx, j + dy, k + dz])
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
    points.sort(key=lambda p: np.linalg.norm(p))
    if len(points) >= n:
        jitter = rng.uniform(-0.03, 0.03, (n, 3))
        return np.array(points[:n]) + jitter
    else:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
        return np.array(points)

def _generate_bcc_cluster(n, seed=42):
    """Generate a BCC-like cluster."""
    rng = np.random.default_rng(seed)
    radius = 1.4 * (n ** (1.0/3.0))
    points = []
    basis = [(0, 0, 0), (0.5, 0.5, 0.5)]
    max_coord = int(radius) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in basis:
                    p = np.array([i + dx, j + dy, k + dz])
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
    points.sort(key=lambda p: np.linalg.norm(p))
    if len(points) >= n:
        jitter = rng.uniform(-0.03, 0.03, (n, 3))
        return np.array(points[:n]) + jitter
    else:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
        return np.array(points)

def _generate_random_cluster(n, seed=42, min_dist=0.55):
    """Generate a random cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = 1.8 * (n ** (1.0/3.0))
    points = []
    attempts = 0
    max_attempts = 30000
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
    x_flat = positions.flatten().copy()
    
    def objective(x):
        pos = x.reshape(-1, 3)
        md = _min_distance(pos)
        if md < 0.45:
            return 1e10 + (0.45 - md) * 1e6
        return _morse_energy(pos, rho)
    
    def gradient(x):
        pos = x.reshape(-1, 3)
        md = _min_distance(pos)
        if md < 0.45:
            return np.zeros_like(x)
        grad_3d = morse_energy_gradient(pos, rho)
        return grad_3d.flatten()
    
    try:
        res = minimize(objective, x_flat, jac=gradient, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10,
                                'maxls': 50, 'maxfun': 30000})
        if res.fun < 1e10:
            pos_res = res.x.reshape(-1, 3)
            if _min_distance(pos_res) >= 0.5:
                return float(res.fun), pos_res
    except Exception:
        pass
    
    return _morse_energy(positions, rho), positions

def _simulated_annealing(start_positions, potential, rho, max_iterations=100, 
                          initial_temp=10.0, cooling=0.9, initial_step=0.5):
    """
    Simulated annealing with L-BFGS local minimization.
    """
    n = start_positions.shape[0]
    current_energy, current_positions = _relax_and_validate(start_positions, potential, rho)
    
    if current_positions is None:
        perturbed = start_positions + np.random.normal(0, 0.15, start_positions.shape)
        current_energy, current_positions = _relax_and_validate(perturbed, potential, rho)
        if current_positions is None:
            return float('inf'), None
    
    best_energy = current_energy
    best_positions = current_positions.copy()
    
    temp = initial_temp
    step_size = initial_step
    
    for it in range(max_iterations):
        if _get_remaining_time() < 5.0:
            break
        
        noise = np.random.normal(0, step_size, current_positions.shape)
        new_positions = current_positions + noise
        
        new_energy, new_relaxed = _relax_and_validate(new_positions, potential, rho)
        
        if new_relaxed is None:
            noise = np.random.normal(0, step_size * 0.4, current_positions.shape)
            new_positions = current_positions + noise
            new_energy, new_relaxed = _relax_and_validate(new_positions, potential, rho)
        
        if new_relaxed is None:
            continue
        
        delta_e = new_energy - current_energy
        accept = False
        if delta_e < 0:
            accept = True
        elif temp > 1e-8:
            exponent = -delta_e / temp
            if exponent > -700:
                accept = np.random.rand() < math.exp(exponent)
        
        if accept:
            current_energy = new_energy
            current_positions = new_relaxed
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = current_positions.copy()
        
        temp *= cooling
        step_size = max(step_size * 0.98, 0.05)
    
    return best_energy, best_positions

def _basin_hop(start_positions, potential, rho, max_iterations=60, initial_step=0.4, initial_temp=3.0):
    """
    Perform basin hopping optimization with adaptive step size.
    """
    n = start_positions.shape[0]
    current_energy, current_positions = _relax_and_validate(start_positions, potential, rho)
    
    if current_positions is None:
        perturbed = start_positions + np.random.normal(0, 0.15, start_positions.shape)
        current_energy, current_positions = _relax_and_validate(perturbed, potential, rho)
        if current_positions is None:
            return float('inf'), None
    
    best_energy = current_energy
    best_positions = current_positions.copy()
    
    step_size = initial_step
    temp = initial_temp
    no_improve_count = 0
    
    for it in range(max_iterations):
        if _get_remaining_time() < 5.0:
            break
        
        if no_improve_count > 5:
            step_size *= 1.3
            no_improve_count = 0
        elif it > 0 and it % 10 == 0:
            step_size *= 1.15
        
        noise = np.random.normal(0, step_size, current_positions.shape)
        new_positions = current_positions + noise
        
        new_energy, new_relaxed = _relax_and_validate(new_positions, potential, rho)
        
        if new_relaxed is None:
            noise = np.random.normal(0, step_size * 0.4, current_positions.shape)
            new_positions = current_positions + noise
            new_energy, new_relaxed = _relax_and_validate(new_positions, potential, rho)
        
        if new_relaxed is None:
            no_improve_count += 1
            continue
        
        delta_e = new_energy - current_energy
        accept = False
        if delta_e < 0:
            accept = True
        elif temp > 1e-8:
            exponent = -delta_e / temp
            if exponent > -700:
                accept = np.random.rand() < math.exp(exponent)
        
        if accept:
            current_energy = new_energy
            current_positions = new_relaxed
            no_improve_count = 0
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = current_positions.copy()
        else:
            no_improve_count += 1
        
        temp *= 0.93
        step_size = max(step_size * 0.97, 0.05)
    
    return best_energy, best_positions

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
    
    initial_guesses = []
    
    # Structured guesses
    if n <= 20:
        try:
            initial_guesses.append(_generate_icosahedron(n, seed=42))
        except Exception:
            pass
    
    try:
        initial_guesses.append(_generate_fcc_cluster(n, seed=42))
    except Exception:
        pass
    
    try:
        initial_guesses.append(_generate_shell(n, seed=42))
    except Exception:
        pass
    
    try:
        initial_guesses.append(_generate_bcc_cluster(n, seed=42))
    except Exception:
        pass
    
    # Random guesses with different seeds
    num_random = 5 if n <= 50 else 4
    for seed in range(1, num_random + 1):
        try:
            initial_guesses.append(_generate_random_cluster(n, seed=seed))
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
    
    # Use simulated annealing from top candidates
    num_sa_starts = min(len(relaxed_results), 5)
    
    for idx in range(num_sa_starts):
        if _get_remaining_time() < 15.0:
            break
        
        start_energy, start_pos = relaxed_results[idx]
        
        # Parameters tuned by rho and n
        if rho <= 3:
            initial_step = 0.8
            initial_temp = 10.0
        elif rho <= 6:
            initial_step = 0.5
            initial_temp = 5.0
        else:
            initial_step = 0.35
            initial_temp = 3.0
        
        if n <= 40:
            max_iter = 80
        elif n <= 60:
            max_iter = 60
        else:
            max_iter = 40
        
        sa_energy, sa_positions = _simulated_annealing(
            start_pos, potential, rho,
            max_iterations=max_iter,
            initial_temp=initial_temp,
            cooling=0.92,
            initial_step=initial_step
        )
        
        if sa_positions is not None and sa_energy < best_energy:
            best_energy = sa_energy
            best_positions = sa_positions.copy()
        
        # Also try basin hopping from best start
        if idx == 0 and _get_remaining_time() > 20.0:
            bh_energy, bh_positions = _basin_hop(
                start_pos, potential, rho,
                max_iterations=max_iter // 2,
                initial_step=initial_step * 1.5,
                initial_temp=initial_temp * 1.5
            )
            if bh_positions is not None and bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_positions.copy()
    
    # Final high-precision polish
    if best_positions is not None and _get_remaining_time() > 8.0:
        polished_energy, polished_pos = _polish(best_positions, rho, max_iter=200)
        if polished_pos is not None and polished_energy < best_energy:
            best_energy = polished_energy
            best_positions = polished_pos
    
    if best_positions is None:
        best_positions = _generate_shell(n, seed=0)
        if not _is_valid(best_positions):
            best_positions = best_positions * 1.5
            if not _is_valid(best_positions):
                best_positions = _generate_random_cluster(n, seed=99)
    
    return best_positions
# EVOLVE-BLOCK-END

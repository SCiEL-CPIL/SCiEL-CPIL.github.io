"""Morse cluster global minima search with Parallel Annealed Basin Hopping (PABH)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

# Global deadline shared across all six construct() calls
_START_TIME = time.time()
_TOTAL_BUDGET = 320.0  # 10s safety margin before 330s kill

def _time_left():
    return _TOTAL_BUDGET - (time.time() - _START_TIME)

def _morse_energy(pos, rho):
    """Compute total Morse energy."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    # Avoid log(0) or exp issues if d is very small, though constraints should prevent this
    d = np.maximum(d, 1e-6)
    e = np.exp(rho * (1.0 - d))
    return float(np.sum(e * (e - 2.0)))

def _morse_gradient(pos, rho):
    """Compute gradient of Morse energy."""
    n = len(pos)
    grad = np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    d = np.maximum(d, 1e-6)
    
    # Force magnitude per pair
    # E_pair = exp(rho(1-r))(exp(rho(1-r))-2)
    # dE/dr = -rho * exp(rho(1-r)) * (2*exp(rho(1-r)) - 1)
    e = np.exp(rho * (1.0 - d))
    force_mag = -rho * e * (2.0 * e - 1.0)
    
    # Distribute forces
    # diff[i,j] = pos[i] - pos[j]
    # grad[i] += force_mag * diff[i,j] / d
    # grad[j] -= force_mag * diff[i,j] / d
    
    # Create a full matrix of forces for vectorized addition
    # We only have upper triangle indices.
    # Let's use np.add.at for efficiency
    i_idx, j_idx = iu
    diff_vec = diff[i_idx, j_idx]
    d_vec = d
    
    # Unit vectors
    unit_vec = diff_vec / d_vec[:, np.newaxis]
    # Force vector on i is force_mag * unit_vec (repulsive if force_mag > 0? 
    # Let's check signs. 
    # If d < 1, rho(1-d) > 0, e > 1. 2e-1 > 1. force_mag = -rho * positive * positive < 0.
    # grad[i] += negative * (pos[i]-pos[j])/d. This pushes i towards j? 
    # Wait. If d < 1 (too close), we want repulsion. 
    # pos[i] - pos[j] points from j to i. 
    # If force_mag is negative, grad[i] points from i to j. This is attraction. 
    # Let's re-verify derivative.
    # E = e(e-2). dE/de = 2e-1. de/dr = -rho e. dE/dr = -rho e (2e-1).
    # If r < 1, e > 1, 2e-1 > 0, dE/dr < 0. 
    # So energy decreases as r increases. Force is -dE/dr > 0 (repulsive).
    # Gradient with respect to pos[i]: 
    # dE/dpos[i] = sum_{j} (dE/dr_ij) * (pos[i]-pos[j])/r_ij
    # Since dE/dr is negative, and (pos[i]-pos[j])/r is direction from j to i, 
    # the gradient points from i to j? No. 
    # Let vector u = (pos[i]-pos[j])/r. 
    # grad[i] = (dE/dr) * u. 
    # If dE/dr < 0, grad[i] is opposite to u. u points i->j? No, pos[i]-pos[j] points from j to i.
    # So u points from j to i. 
    # grad[i] points from i to j. 
    # This means the atom moves towards j? 
    # Wait. If I am at x=1, j is at x=0. u = 1. dE/dr < 0. grad = -k. 
    # Move x by -grad = +k. x increases. Moves away from j. Correct.
    
    f_vec = force_mag[:, np.newaxis] * unit_vec
    np.add.at(grad, i_idx, f_vec)
    np.add.at(grad, j_idx, -f_vec)
    
    return grad

def _local_relax(pos, rho, max_iter=50):
    """Relax a configuration using L-BFGS-B."""
    try:
        res = opt.minimize(
            _morse_energy, pos, args=(rho,),
            jac=_morse_gradient,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': 1e-5}
        )
        return res.x, res.fun
    except Exception:
        return pos, _morse_energy(pos, rho)

def _build_fcc_cluster(n, spacing=1.0):
    """Build an FCC-like compact cluster by taking the n closest FCC lattice points to origin."""
    points = []
    for i in range(-6, 7):
        for j in range(-6, 7):
            for k in range(-6, 7):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * spacing * 0.7071
                    points.append(p)
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx].copy()

def _build_bcc_cluster(n, spacing=1.0):
    """Build a BCC-like compact cluster."""
    points = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            for k in range(-5, 6):
                p = np.array([i, j, k]) * spacing * 0.577
                points.append(p)
                pc = np.array([i + 0.5, j + 0.5, k + 0.5]) * spacing * 0.577
                points.append(pc)
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx].copy()

def _build_icosahedral_shells(n, spacing=1.0):
    """Build cluster using icosahedral shell geometry."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico12 = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norm_ico = np.linalg.norm(ico12[0])
    ico12 /= norm_ico
    
    points = [np.zeros(3)]
    
    shells = [
        (0.0, 1),      
        (1.0, 12),     
        (1.633, 30),   
        (2.0, 56),     
        (2.449, 92),   
    ]
    
    for radius, count in shells:
        if radius == 0.0:
            continue
        for v in ico12:
            if len(points) >= n:
                break
            points.append(v * radius)
        if count > 12 and len(points) < n:
            rng = np.random.default_rng(radius * 1000)
            while len(points) < min(n, len(points) + count - 12):
                p = rng.normal(0, 1, 3)
                p = p / np.linalg.norm(p) * radius
                if all(np.linalg.norm(p - q) >= 0.4 * spacing for q in points):
                    points.append(p)
    
    shell_radius = 3.0
    while len(points) < n:
        rng = np.random.default_rng(int(shell_radius * 1000))
        added = 0
        for _ in range(100):
            if len(points) >= n:
                break
            p = rng.normal(0, 1, 3)
            p = p / np.linalg.norm(p) * shell_radius
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
                added += 1
        if added == 0:
            shell_radius += 0.5
        else:
            shell_radius += 0.3
    
    return np.array(points[:n])

def _build_random_compact(n, spacing=1.0, seed=42):
    """Build a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.85
    points = []
    max_attempts = 10000
    for _ in range(max_attempts):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.65 * spacing for q in points):
                points.append(p)
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])

def _ensure_min_distance(pos, min_d=0.5, max_iter=20):
    """Ensure all pairs are at least min_d apart by nudging."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_d:
                    diff = pos[i] - pos[j]
                    if np.linalg.norm(diff) < 1e-10:
                        diff = np.array([1.0, 0, 0])
                    diff /= np.linalg.norm(diff)
                    shift = (min_d - d) * 0.5 + 0.01
                    pos[i] += diff * shift
                    pos[j] -= diff * shift
                    moved = True
        if not moved:
            break
    return pos

def _basin_hopping_ensemble(pos_list, rho, n_chains=4, total_time=10.0):
    """
    Perform Parallel Annealed Basin Hopping.
    pos_list: list of initial positions
    rho: Morse parameter
    n_chains: number of parallel chains
    total_time: time budget for this ensemble
    """
    n_atoms = len(pos_list[0])
    # Initialize chains
    # We use the provided pos_list as seeds for different chains if possible, 
    # or replicate the best one.
    if len(pos_list) < n_chains:
        # Pad with perturbed versions of the best one
        best_idx = 0
        # Assume pos_list[0] is best or just use first
        for i in range(len(pos_list), n_chains):
            noise = np.random.normal(0, 0.1, pos_list[0].shape)
            pos_list.append(pos_list[best_idx] + noise)
    
    # Take first n_chains
    chains = [p.copy() for p in pos_list[:n_chains]]
    energies = [_morse_energy(p, rho) for p in chains]
    
    # Local relaxation of initial chains
    for i in range(n_chains):
        chains[i], energies[i] = _local_relax(chains[i], rho)
    
    # Find global best
    global_best_idx = np.argmin(energies)
    global_best_energy = energies[global_best_idx]
    global_best_pos = chains[global_best_idx].copy()
    
    # Simulated Annealing parameters
    # Estimate energy scale
    # E ~ -n*rho? No, E ~ -n. 
    # Let's use a temperature based on the variance of energies or just a fixed scale.
    # Typical Morse well depth is 1. Total energy ~ -n. 
    # Let's start T high enough to allow moves.
    T_start = 0.5 * n_atoms # Rough estimate
    T_end = 0.01
    n_steps = 100 # Fixed steps per chain per "round"? No, time-based.
    
    start_time = time.time()
    step_size = 0.15 # Initial perturbation size
    
    # We will run until time runs out
    while _time_left() > 5.0 and (time.time() - start_time) < total_time:
        # Generate perturbations
        new_chains = []
        new_energies = []
        
        for i in range(n_chains):
            # Perturb
            noise = np.random.normal(0, step_size, chains[i].shape)
            new_pos = chains[i] + noise
            # Relax
            new_pos, new_e = _local_relax(new_pos, rho)
            
            # Check minimum distance constraint during search? 
            # If invalid, penalize energy heavily or reject.
            # For simplicity, if min dist < 0.5, we reject the move (or penalize).
            # Let's check min dist.
            d_min = np.min(np.linalg.norm(new_pos[:, np.newaxis, :] - new_pos[np.newaxis, :, :], axis=2)[np.triu_indices(n_atoms, k=1)])
            if d_min < 0.5:
                new_e += 1000.0 # Heavy penalty
            
            new_chains.append(new_pos)
            new_energies.append(new_e)
        
        # Metropolis Step for each chain
        # Parallel Annealing: We use a single temperature T for all chains? 
        # Or individual temperatures? 
        # Standard PABH: All chains at same T, but different configurations.
        # Let's use a single T that cools down.
        # Cooling schedule: Linear or Geometric.
        # Let's use Geometric: T = T_start * (T_end/T_start)^step/total_steps
        # Since we don't know total steps, use time fraction.
        elapsed = time.time() - start_time
        frac = elapsed / total_time
        if frac > 1.0: frac = 1.0
        T = T_start * (T_end / T_start) ** frac
        
        for i in range(n_chains):
            delta_e = new_energies[i] - energies[i]
            if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
                chains[i] = new_chains[i]
                energies[i] = new_energies[i]
                # Update global best
                if energies[i] < global_best_energy:
                    global_best_energy = energies[i]
                    global_best_pos = chains[i].copy()
        
        # Occasionally restart the worst chain with the best chain (reheat/swap)
        # This helps escape local minima.
        if np.random.rand() < 0.1:
            worst_idx = np.argmax(energies)
            chains[worst_idx] = global_best_pos + np.random.normal(0, 0.1, global_best_pos.shape)
            chains[worst_idx], energies[worst_idx] = _local_relax(chains[worst_idx], rho)
        
        # Adapt step size? 
        # If acceptance rate is too low, decrease step size. If too high, increase.
        # For simplicity, keep constant or slowly decrease.
        step_size *= 0.999
        
    return global_best_pos, global_best_energy

def construct(instance):
    """Find low-energy Morse cluster configuration."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Check time budget
    tl = _time_left()
    if tl < 10.0:
        # Very little time left, just return a quick result
        pos = _build_random_compact(size, spacing=1.0, seed=size)
        pos, _ = _local_relax(pos, parameter)
        return _ensure_min_distance(pos)
    
    # Generate diverse starting configurations
    start_configs = []
    builders = [
        lambda: _build_fcc_cluster(size, 1.0),
        lambda: _build_bcc_cluster(size, 1.0),
        lambda: _build_icosahedral_shells(size, 1.0),
        lambda: _build_random_compact(size, 1.0, seed=size),
        lambda: _build_random_compact(size, 1.0, seed=size + 100),
        lambda: _build_fcc_cluster(size, 1.0) + np.random.default_rng(size).normal(0, 0.1, (size, 3)),
    ]
    
    for builder in builders:
        try:
            cfg = builder()
            if len(cfg) == size:
                start_configs.append(cfg)
        except Exception:
            pass
    
    if not start_configs:
        start_configs = [_build_random_compact(size, 1.0, seed=size)]
    
    # Relax all initial configurations to find local minima
    relaxed_configs = []
    relaxed_energies = []
    for cfg in start_configs:
        pos, e = _local_relax(cfg, parameter)
        relaxed_configs.append(pos)
        relaxed_energies.append(e)
        
    # Sort by energy
    sorted_indices = np.argsort(relaxed_energies)
    sorted_configs = [relaxed_configs[i] for i in sorted_indices]
    sorted_energies = [relaxed_energies[i] for i in sorted_indices]
    
    # Use the best few as seeds for PABH
    # Allocate time for PABH
    # Reserve 5s for final processing
    pabh_time = max(5.0, _time_left() - 5.0)
    
    # Run PABH
    best_pos, best_energy = _basin_hopping_ensemble(sorted_configs, parameter, n_chains=4, total_time=pabh_time)
    
    # Final verification: ensure minimum distance constraint
    pos = _ensure_min_distance(best_pos)
    
    return pos
# EVOLVE-BLOCK-END

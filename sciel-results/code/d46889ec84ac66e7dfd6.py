"""Morse cluster global minima: Langevin dynamics with annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_HARD_DEADLINE = _START_TIME + 285.0  # 15s safety margin before 300s kill
_instance_call_count = [0]


def _time_left():
    return _HARD_DEADLINE - time.time()


def _morse_energy(positions, rho):
    """Compute Morse energy E = sum_{i<j} exp(rho(1-r_ij))(exp(rho(1-r_ij)) - 2)."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    u = rho * (1.0 - dists)
    # Clip u to avoid overflow
    u = np.clip(u, -50, 50)
    exp_u = np.exp(u)
    v = exp_u * (exp_u - 2.0)
    v[np.isinf(v)] = 0.0
    return 0.5 * np.sum(v)


def _morse_energy_grad(positions, rho):
    """Compute Morse energy and gradient."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Avoid division by zero
    dists_safe = np.where(dists < 1e-10, 1e-10, dists)
    
    u = rho * (1.0 - dists)
    u = np.clip(u, -50, 50)
    exp_u = np.exp(u)
    v = exp_u * (exp_u - 2.0)
    v[np.isinf(v)] = 0.0
    energy = 0.5 * np.sum(v)
    
    # Gradient: dV/dr = -rho * exp(u) * (exp(u) - 1)
    # dV/dx_i = sum_j dV/dr_ij * (r_i - r_j)/r_ij
    dv_dr = -rho * exp_u * (exp_u - 1.0)
    dv_dr[np.isinf(dv_dr)] = 0.0
    
    # grad[i] = sum_j dv_dr[i,j] * (pos[i] - pos[j]) / dist[i,j]
    # diff[i,j] = pos[i] - pos[j]
    # factor[i,j] = dv_dr[i,j] / dist[i,j]
    factor = dv_dr / dists_safe
    factor[np.isinf(factor)] = 0.0
    factor[~np.isfinite(factor)] = 0.0
    
    grad = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    return energy, grad


def _random_compact(n, radius=3.0, seed=0):
    """Generate n random points in a sphere."""
    rng = np.random.default_rng(seed)
    points = []
    min_dist = 0.55
    attempts = 0
    max_attempts = n * 500
    while len(points) < n and attempts < max_attempts:
        vec = rng.normal(size=3)
        norm = np.linalg.norm(vec)
        if norm > 0:
            p = vec / norm * radius * (rng.uniform(0, 1) ** (1/3))
            if len(points) == 0:
                points.append(p)
            else:
                pts = np.array(points)
                dists = np.linalg.norm(pts - p, axis=1)
                if np.min(dists) >= min_dist:
                    points.append(p)
        attempts += 1
    while len(points) < n:
        vec = rng.normal(size=3)
        norm = np.linalg.norm(vec)
        if norm > 0:
            p = vec / norm * radius * (rng.uniform(0, 1) ** (1/3))
        else:
            p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _fcc_points(n, spacing=1.0):
    """Generate n points from an FCC lattice."""
    rng = np.random.default_rng(42)
    a = spacing * 0.7071
    points = []
    side = int(np.ceil(n ** (1/3))) + 2
    for i in range(-side, side+1):
        for j in range(-side, side+1):
            for k in range(-side, side+1):
                points.append([i*a, j*a, k*a])
                points.append([i*a + a, j*a + a, k*a])
                points.append([i*a + a, j*a, k*a + a])
                points.append([i*a, j*a + a, k*a + a])
    points = np.array(points)
    if len(points) >= n:
        idx = rng.choice(len(points), n, replace=False)
        return points[idx]
    else:
        extra = _random_compact(n - len(points), radius=5.0, seed=100)
        return np.vstack([points, extra])


def _bcc_points(n, spacing=1.0):
    """Generate n points from a BCC lattice."""
    rng = np.random.default_rng(43)
    a = spacing * 0.7071
    points = []
    side = int(np.ceil(n ** (1/3))) + 2
    for i in range(-side, side+1):
        for j in range(-side, side+1):
            for k in range(-side, side+1):
                points.append([i*a, j*a, k*a])
                points.append([i*a + a/2, j*a + a/2, k*a + a/2])
    points = np.array(points)
    if len(points) >= n:
        idx = rng.choice(len(points), n, replace=False)
        return points[idx]
    else:
        extra = _random_compact(n - len(points), radius=5.0, seed=101)
        return np.vstack([points, extra])


def _sc_points(n, spacing=1.0):
    """Generate n points from a simple cubic lattice."""
    rng = np.random.default_rng(44)
    a = spacing
    points = []
    side = int(np.ceil(n ** (1/3))) + 2
    for i in range(-side, side+1):
        for j in range(-side, side+1):
            for k in range(-side, side+1):
                points.append([i*a, j*a, k*a])
    points = np.array(points)
    if len(points) >= n:
        idx = rng.choice(len(points), n, replace=False)
        return points[idx]
    else:
        extra = _random_compact(n - len(points), radius=5.0, seed=102)
        return np.vstack([points, extra])


def _perturb(points, magnitude, seed=0):
    """Perturb points by adding random noise."""
    rng = np.random.default_rng(seed)
    return points + rng.uniform(-magnitude, magnitude, points.shape)


def _langevin_anneal(initial_pos, rho, time_limit, seed=0):
    """
    Run Langevin dynamics with annealing.
    Initialize from high-temperature ideal gas (random positions),
    use gradient-driven dynamics with adaptive step sizes,
    and linear cooling schedule.
    """
    rng = np.random.default_rng(seed)
    n = len(initial_pos)
    
    # Start from a more diffuse configuration (high-temperature ideal gas)
    # Spread out the initial positions
    current_pos = initial_pos.copy()
    # Add some random spread to simulate high-T
    current_pos += rng.normal(0, 0.5, (n, 3))
    
    # Initial energy and gradient
    current_energy, current_grad = _morse_energy_grad(current_pos, rho)
    
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    # Temperature parameters
    T_start = 100.0
    T_end = 0.5
    start_time = time.time()
    
    # Adaptive step size
    dt = 0.01
    friction = 1.0
    
    # Track acceptance rate for adaptive dt
    accept_count = 0
    total_count = 0
    window_size = 100
    
    step_count = 0
    relax_interval = 200
    
    while time.time() - start_time < time_limit:
        if time.time() > _HARD_DEADLINE - 5.0:
            break
        
        # Linear cooling in log space
        elapsed = time.time() - start_time
        frac = min(1.0, elapsed / time_limit)
        T = T_start * (T_end / T_start) ** frac
        
        # Compute gradient
        current_energy, current_grad = _morse_energy_grad(current_pos, rho)
        
        # Langevin step: m*x'' = -grad - gamma*x' + sqrt(2*kT*gamma)*noise
        # For overdamped Langevin: x' = -mu*grad + sqrt(2*kT*mu)*noise
        # We use a simple Euler-Maruyama scheme
        noise = rng.normal(0, 1, (n, 3))
        displacement = -dt * current_grad + np.sqrt(2 * T * dt) * noise
        
        # Adaptive dt based on gradient magnitude
        grad_norm = np.linalg.norm(current_grad)
        if grad_norm > 1e-6:
            # Limit step size to avoid instability
            max_step = 0.1
            step_norm = np.linalg.norm(displacement)
            if step_norm > max_step:
                displacement *= max_step / step_norm
        
        old_pos = current_pos.copy()
        current_pos = current_pos + displacement
        
        # Compute new energy
        new_energy, new_grad = _morse_energy_grad(current_pos, rho)
        
        # Metropolis acceptance criterion
        delta_e = new_energy - current_energy
        if delta_e < 0 or (T > 0 and rng.random() < np.exp(-delta_e / max(T, 1e-6))):
            current_energy = new_energy
            current_grad = new_grad
            accept_count += 1
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
        else:
            current_pos = old_pos
            # Keep old gradient
        
        total_count += 1
        step_count += 1
        
        # Adaptive dt adjustment every window_size steps
        if total_count % window_size == 0:
            accept_rate = accept_count / total_count
            if accept_rate < 0.2:
                dt *= 0.8  # Reduce step size
            elif accept_rate > 0.8:
                dt *= 1.2  # Increase step size
            dt = np.clip(dt, 0.001, 0.05)
            accept_count = 0
            total_count = 0
        
        # Periodic relaxation
        if step_count % relax_interval == 0:
            try:
                relaxed_energy, relaxed_pos = relax(current_pos, 'morse', rho)
                if relaxed_energy < best_energy:
                    best_energy = relaxed_energy
                    best_pos = relaxed_pos.copy()
                # Update current state
                current_energy = relaxed_energy
                current_pos = relaxed_pos.copy()
                current_grad, _ = _morse_energy_grad(current_pos, rho)
            except:
                pass
    
    return best_energy, best_pos


def construct(instance):
    """
    Find low-energy Morse cluster configurations.
    """
    _instance_call_count[0] += 1
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    
    # Time budget
    per_instance_budget = 40.0
    if _instance_call_count[0] == 1:
        per_instance_budget = 45.0
    elif _instance_call_count[0] == 6:
        per_instance_budget = 35.0
        
    instance_deadline = time.time() + per_instance_budget
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Random compact (multiple seeds)
    for seed in range(4):
        initial_configs.append(_random_compact(n, radius=3.0, seed=seed))
        
    # 2. Lattice-based
    initial_configs.append(_fcc_points(n, spacing=1.0))
    initial_configs.append(_bcc_points(n, spacing=1.0))
    initial_configs.append(_sc_points(n, spacing=1.0))
    
    # 3. Perturbed lattice
    base_fcc = _fcc_points(n, spacing=1.0)
    for mag in [0.1, 0.3]:
        initial_configs.append(_perturb(base_fcc, mag, seed=200 + int(mag*10)))
        
    # 4. Perturbed random
    base_rand = _random_compact(n, radius=3.0, seed=999)
    for mag in [0.1, 0.3]:
        initial_configs.append(_perturb(base_rand, mag, seed=300 + int(mag*10)))
    
    # Run Langevin annealing on each, keeping the best
    best_energy = np.inf
    best_positions = None
    
    num_runs = len(initial_configs)
    time_per_run = (per_instance_budget - 5.0) / num_runs
    
    for i, init_pos in enumerate(initial_configs):
        if time.time() > instance_deadline - 2.0:
            break
            
        try:
            e, pos = _langevin_anneal(init_pos, rho, time_per_run, seed=i+42)
            if e < best_energy:
                best_energy = e
                best_positions = pos
        except Exception as ex:
            pass
    
    # If no run succeeded, do a quick relax
    if best_positions is None:
        init_pos = _random_compact(n, radius=3.0, seed=0)
        try:
            best_energy, best_positions = relax(init_pos, 'morse', rho)
        except:
            best_positions = init_pos
            best_energy = _morse_energy(init_pos, rho)
    
    # Final refinement
    if time.time() < instance_deadline - 1.0:
        try:
            final_energy, final_pos = relax(best_positions, 'morse', rho)
            if final_energy < best_energy:
                best_energy = final_energy
                best_positions = final_pos
        except:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END

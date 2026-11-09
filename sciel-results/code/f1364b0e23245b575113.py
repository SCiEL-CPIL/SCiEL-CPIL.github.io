"""Physics-driven self-assembly for Dzugutov cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient


def _dzugutov_energy_and_gradient(positions, A=5.82, B=1.28, a=1.87, b=1.94, c=1.10, d=0.27, m=16):
    """Compute total energy and gradient for Dzugutov potential."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    # Pairwise distance matrix
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    energy = 0.0
    grad = np.zeros_like(positions)
    
    for i in range(n):
        for j in range(i + 1, n):
            r = dist[i, j]
            if r < 1e-10:
                continue
            
            if r < a:
                r_neg_m = r ** (-m)
                exp_term = np.exp(c / (r - a))
                v = A * (r_neg_m - B) * exp_term
                dv_dr = A * (-m * r ** (-m - 1)) * exp_term + A * (r_neg_m - B) * exp_term * (-c / (r - a) ** 2)
            elif r < b:
                exp_term = np.exp(d / (r - b))
                v = B * exp_term
                dv_dr = B * exp_term * (-d / (r - b) ** 2)
            else:
                v = 0.0
                dv_dr = 0.0
            
            energy += v
            
            if r > 1e-10 and r < b:
                r_vec = positions[i] - positions[j]
                f = -dv_dr / r * r_vec
                grad[i] += f
                grad[j] -= f
    
    return energy, grad


def _langevin_step(pos, vel, dt, temperature, damping, rng):
    """Single Langevin dynamics step."""
    energy, grad = _dzugutov_energy_and_gradient(pos)
    forces = -grad
    
    sigma = np.sqrt(2 * temperature * damping)
    noise = rng.normal(0, 1, size=pos.shape)
    
    new_vel = vel + (forces - damping * vel) * dt + sigma * np.sqrt(dt) * noise
    new_pos = pos + new_vel * dt
    
    # Enforce minimum distance
    n = len(new_pos)
    for i in range(n):
        for j in range(i + 1, n):
            d = new_pos[i] - new_pos[j]
            dist = np.linalg.norm(d)
            if dist < 0.5 and dist > 1e-10:
                push = (0.5 - dist) / dist * d
                new_pos[i] += 0.5 * push
                new_pos[j] -= 0.5 * push
    
    return new_pos, new_vel


def _sequential_assembly(n, seed, time_budget=15):
    """Sequentially add atoms with Langevin relaxation."""
    rng = np.random.default_rng(seed)
    start_time = time.time()
    
    positions = np.array([[0.0, 0.0, 0.0], [1.05, 0.0, 0.0]])
    velocities = np.zeros((2, 3))
    
    for target_n in range(3, n + 1):
        elapsed = time.time() - start_time
        if elapsed > time_budget * 0.85:
            break
        
        center = np.mean(positions, axis=0)
        radius = 1.2 * target_n ** (1.0 / 3.0)
        
        new_atom = None
        for _ in range(100):
            candidate = center + rng.uniform(-radius, radius, 3)
            if all(np.linalg.norm(candidate - p) >= 0.55 for p in positions):
                new_atom = candidate
                break
        
        if new_atom is None:
            new_atom = center + rng.normal(0, 1.0, 3)
        
        positions = np.vstack([positions, new_atom])
        velocities = np.vstack([velocities, np.zeros(3)])
        
        remaining = time_budget - (time.time() - start_time)
        if remaining > 0.5:
            n_steps = min(200, max(20, int(remaining * 30 / target_n)))
            temp = max(0.02, 0.4 * (1.0 - target_n / (n + 1)))
            
            for _ in range(n_steps):
                positions, velocities = _langevin_step(positions, velocities, 0.005, temp, 1.0, rng)
        
        energy, pos = relax(positions, 'dzugutov', None)
        positions = pos
    
    return positions


def _random_langevin(n, seed, time_budget=20):
    """Random start with Langevin annealing."""
    rng = np.random.default_rng(seed)
    start_time = time.time()
    
    best_energy = np.inf
    best_positions = None
    
    n_starts = max(2, min(6, int(time_budget / 8)))
    
    for start in range(n_starts):
        elapsed = time.time() - start_time
        if elapsed > time_budget - 3:
            break
        
        radius = 1.1 * n ** (1.0 / 3.0) * 0.8
        positions = rng.uniform(-radius, radius, size=(n, 3))
        
        # Enforce min distance
        for i in range(n):
            for j in range(i + 1, n):
                for _ in range(50):
                    d = positions[i] - positions[j]
                    dist = np.linalg.norm(d)
                    if dist >= 0.5:
                        break
                    if dist > 1e-10:
                        positions[i] += (0.5 - dist) / dist * d * 0.5
                        positions[j] -= (0.5 - dist) / dist * d * 0.5
        
        velocities = rng.normal(0, 0.1, size=(n, 3))
        
        remaining = time_budget - (time.time() - start_time)
        n_steps = min(2000, max(100, int(remaining * 80)))
        
        for step in range(n_steps):
            progress = step / max(1, n_steps)
            temp = 0.6 * (1 - progress) ** 2 + 0.01
            
            positions, velocities = _langevin_step(positions, velocities, 0.005, temp, 0.8, rng)
            
            if step % 150 == 149:
                energy, pos = relax(positions, 'dzugutov', None)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = pos.copy()
        
        energy, pos = relax(positions, 'dzugutov', None)
        if energy < best_energy:
            best_energy = energy
            best_positions = pos
    
    return best_positions


def _polish(positions, seed, time_budget=5):
    """Polish with small perturbations."""
    rng = np.random.default_rng(seed)
    start_time = time.time()
    
    energy, _ = _dzugutov_energy_and_gradient(positions)
    best_energy = energy
    best_pos = positions.copy()
    
    n_attempts = max(2, min(8, int(time_budget)))
    
    for _ in range(n_attempts):
        if time.time() - start_time > time_budget - 1:
            break
        
        pos = positions + rng.normal(0, 0.06, size=positions.shape)
        
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                if dist < 0.5 and dist > 1e-10:
                    push = (0.5 - dist) / dist * d
                    pos[i] += 0.5 * push
                    pos[j] -= 0.5 * push
        
        energy, pos = relax(pos, 'dzugutov', None)
        if energy < best_energy:
            best_energy = energy
            best_pos = pos
    
    return best_pos


def construct(instance):
    """Physics-driven self-assembly for Dzugutov cluster global minima."""
    size = instance["n"]
    potential = instance["potential"]
    
    if potential != 'dzugutov':
        spacing = {"morse": 1.0, "sutton_chen": 0.72}.get(potential, 1.0)
        rng = np.random.default_rng(size)
        radius = spacing * size ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
                points.append(p)
        energy, positions = relax(np.array(points), potential, instance.get("rho", instance.get("kind")))
        return positions
    
    # 300s total / 6 instances = 50s per instance, use 45s
    total_budget = 45
    
    # Strategy 1: Sequential assembly
    t0 = time.time()
    seed1 = size * 7919 + 1
    pos1 = _sequential_assembly(size, seed1, time_budget=total_budget * 0.35)
    energy1, pos1 = relax(pos1, 'dzugutov', None)
    
    # Strategy 2: Random Langevin
    seed2 = size * 7919 + 2
    pos2 = _random_langevin(size, seed2, time_budget=total_budget * 0.55)
    if pos2 is not None:
        energy2, pos2 = relax(pos2, 'dzugutov', None)
    else:
        energy2 = np.inf
        pos2 = None
    
    # Choose best
    if energy2 < energy1:
        best_pos = pos2
    else:
        best_pos = pos1
    
    # Final polish
    remaining = total_budget - (time.time() - t0)
    if best_pos is not None and remaining > 3:
        seed3 = size * 7919 + 3
        best_pos = _polish(best_pos, seed3, time_budget=min(8, remaining - 2))
    
    if best_pos is None:
        rng = np.random.default_rng(size)
        radius = 1.1 * size ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        energy, best_pos = relax(np.array(points), 'dzugutov', None)
    
    return best_pos
# EVOLVE-BLOCK-END

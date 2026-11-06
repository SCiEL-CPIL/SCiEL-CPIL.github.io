"""Simulated annealing with Langevin dynamics for Morse cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy_positions(positions, rho):
    """Compute Morse energy for given positions."""
    n = len(positions)
    if n < 2:
        return 0.0
    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Mask out self-interactions
    np.fill_diagonal(r, np.inf)
    # Morse potential: exp(rho*(1-r))*(exp(rho*(1-r))-2)
    x = rho * (1.0 - r)
    # Avoid overflow for very small r
    x = np.minimum(x, 50.0)
    e = np.exp(x) * (np.exp(x) - 2.0)
    return 0.5 * np.sum(e)  # Each pair counted twice


def _morse_grad_positions(positions, rho):
    """Compute Morse energy gradient for given positions."""
    n = len(positions)
    if n < 2:
        return np.zeros_like(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-12)
    np.fill_diagonal(r, 1.0)
    # Gradient of Morse potential wrt position
    # dE/dr = rho * exp(rho*(1-r)) * (2*exp(rho*(1-r)) - 3)
    x = rho * (1.0 - r)
    x = np.minimum(x, 50.0)
    e = np.exp(x)
    de_dr = rho * e * (2.0 * e - 3.0)
    # Gradient wrt position: de_dr * (r_ij / |r_ij|)
    # diff[i,j] = pos[i] - pos[j], so dr/dpos[i] = diff[i,j]/r[i,j]
    r_safe = r.copy()
    np.fill_diagonal(r_safe, 1.0)
    grad = np.einsum('ij,ijk->ik', de_dr, diff / r_safe[:, :, np.newaxis])
    return grad


def _min_distance(positions):
    """Compute minimum pairwise distance efficiently."""
    n = len(positions)
    if n < 2:
        return np.inf
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    return np.min(r)


def _langevin_step(positions, velocities, rho, T, dt, gamma, rng):
    """One Langevin dynamics step."""
    n = len(positions)
    # Compute forces (negative gradient)
    grad = _morse_grad_positions(positions, rho)
    forces = -grad
    
    # Velocity Verlet-like Langevin update
    # v_new = v + dt * (F - gamma*v) + sqrt(2*gamma*T*dt) * noise
    noise = rng.standard_normal((n, 3)) * np.sqrt(2.0 * gamma * T * dt)
    velocities = velocities + dt * (forces - gamma * velocities) + noise
    positions = positions + dt * velocities
    return positions, velocities


def _geometric_cooling(T0, T_end, steps, step_idx):
    """Geometric cooling schedule - more effective than linear."""
    if steps <= 1:
        return T_end
    ratio = (T_end / T0) ** (1.0 / steps)
    return T0 * (ratio ** step_idx)


def _anneal_cluster(n, rho, T0=5.0, T_end=0.005, n_steps=4000, gamma=3.0, dt=0.01, rng=None):
    """Run simulated annealing with Langevin dynamics."""
    if rng is None:
        rng = np.random.default_rng()
    
    # Initialize from high-temperature ideal gas: random positions in a sphere
    radius = 2.0 * (n / (4.0/3.0 * np.pi)) ** (1.0/3.0)
    positions = rng.uniform(-radius, radius, (n, 3))
    velocities = np.zeros((n, 3))
    
    best_energy = _morse_energy_positions(positions, rho)
    best_positions = positions.copy()
    
    current_energy = best_energy
    
    # Adaptive step size
    dt_current = dt
    T_current = T0
    
    for step in range(n_steps):
        T_current = _geometric_cooling(T0, T_end, n_steps, step)
        
        # Adaptive dt: reduce if energy increases too much
        new_positions, new_velocities = _langevin_step(
            positions, velocities, rho, T_current, dt_current, gamma, rng
        )
        
        # Check minimum distance constraint using vectorized computation
        min_dist = _min_distance(new_positions)
        
        new_energy = _morse_energy_positions(new_positions, rho)
        
        # Handle constraint violation
        if min_dist < 0.5:
            # Push apart atoms that are too close
            diff = new_positions[:, np.newaxis, :] - new_positions[np.newaxis, :, :]
            r = np.sqrt(np.sum(diff ** 2, axis=-1))
            np.fill_diagonal(r, np.inf)
            
            # Find pairs too close
            mask = r < 0.5
            for i in range(n):
                for j in range(i+1, n):
                    if mask[i, j]:
                        d = r[i, j]
                        if d > 1e-12:
                            direction = diff[i, j] / d
                            shift = (0.5 - d) / 2.0
                            new_positions[i] += shift * direction
                            new_positions[j] -= shift * direction
                        else:
                            # Random separation
                            new_positions[i] += rng.normal(0, 0.1, 3)
                            new_positions[j] -= rng.normal(0, 0.1, 3)
            
            new_energy = _morse_energy_positions(new_positions, rho)
        
        # Accept if energy decreased or with Metropolis criterion
        if new_energy < current_energy:
            positions = new_positions
            velocities = new_velocities
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = positions.copy()
        else:
            # Metropolis acceptance
            delta_E = new_energy - current_energy
            if T_current > 1e-10:
                accept_prob = np.exp(-delta_E / T_current)
                if rng.random() < accept_prob:
                    positions = new_positions
                    velocities = new_velocities
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_positions = positions.copy()
        
        # Adaptive step size
        if new_energy > current_energy * 1.5 + 1.0:
            dt_current = max(dt_current * 0.9, 0.001)
        elif step % 100 == 0 and new_energy < current_energy:
            dt_current = min(dt_current * 1.05, 0.05)
    
    return best_energy, best_positions


def _basin_hopping(n, rho, T0=3.0, T_end=0.005, n_restarts=5, n_steps_per_restart=2000, gamma=3.0, dt=0.01, rng=None):
    """Basin hopping: multiple annealing restarts from perturbed best solutions."""
    if rng is None:
        rng = np.random.default_rng()
    
    best_energy = np.inf
    best_positions = None
    
    for restart in range(n_restarts):
        if best_positions is not None and restart > 0:
            # Perturb best solution
            perturbation_scale = 0.5 / (restart + 1)
            positions = best_positions + rng.normal(0, perturbation_scale, (n, 3))
        else:
            # Initial random start
            radius = 2.0 * (n / (4.0/3.0 * np.pi)) ** (1.0/3.0)
            positions = rng.uniform(-radius, radius, (n, 3))
        
        energy, positions = _anneal_cluster(
            n, rho,
            T0=T0 * (1.0 + 0.2 * restart),
            T_end=T_end,
            n_steps=n_steps_per_restart,
            gamma=gamma,
            dt=dt,
            rng=rng
        )
        
        if energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()
    
    return best_energy, best_positions


def construct(instance):
    """Find global minimum of Morse cluster using simulated annealing with Langevin dynamics."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    if potential != "morse":
        # Fallback for non-Morse
        spacing = {"sutton_chen": 0.72, "dzugutov": 1.1}.get(potential, 1.0)
        rng = np.random.default_rng(size)
        radius = spacing * size ** (1.0 / 3.0) * 0.75
        best = None
        for _ in range(4):
            points = []
            while len(points) < size:
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
                    points.append(p)
            energy, positions = relax(np.array(points), potential, parameter)
            if best is None or energy < best[0]:
                best = (energy, positions)
        return best[1]
    
    rho = parameter
    rng = np.random.default_rng(size)
    
    # Use basin hopping with multiple restarts
    best_energy = np.inf
    best_positions = None
    
    # Adapt parameters based on cluster size
    if size <= 40:
        n_restarts = 8
        n_steps_per_restart = 3000
        T0 = 4.0
    elif size <= 60:
        n_restarts = 6
        n_steps_per_restart = 2500
        T0 = 5.0
    else:
        n_restarts = 5
        n_steps_per_restart = 2000
        T0 = 6.0
    
    energy, positions = _basin_hopping(
        size, rho,
        T0=T0,
        T_end=0.005,
        n_restarts=n_restarts,
        n_steps_per_restart=n_steps_per_restart,
        gamma=3.0,
        dt=0.01,
        rng=rng
    )
    
    if positions is not None:
        # Local relaxation to polish
        relaxed_energy, relaxed_positions = relax(positions, "morse", rho)
        # Check minimum distance
        valid = _min_distance(relaxed_positions) >= 0.5
        if valid:
            best_energy = relaxed_energy
            best_positions = relaxed_positions
        else:
            best_energy = energy
            best_positions = positions
    
    if best_positions is None:
        # Fallback: random compact cluster
        radius = size ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 for q in points):
                points.append(p)
        _, best_positions = relax(np.array(points), "morse", rho)
    
    return best_positions
# EVOLVE-BLOCK-END

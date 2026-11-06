"""Morse cluster global minima: lattice seeds + basin hopping with adaptive cooling."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_HARD_DEADLINE = _START_TIME + 280.0  # 20s safety margin before 300s kill


def _time_left():
    return _HARD_DEADLINE - time.time()


def _morse_energy_positions(pos, rho):
    """Compute Morse energy for positions."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, 1e10)
    # Avoid division by zero / invalid distances
    mask = dist > 0.5
    dist_masked = np.where(mask, dist, 1e10)
    x = rho * (1.0 - dist_masked)
    e = np.exp(x)
    pair_e = e * (e - 2.0)
    # Zero out diagonal and invalid pairs
    pair_e[~mask] = 0.0
    np.fill_diagonal(pair_e, 0.0)
    return float(np.sum(pair_e) * 0.5)


def _morse_grad_positions(pos, rho):
    """Compute Morse gradient for positions."""
    n = len(pos)
    if n < 2:
        return np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, 1e10)
    mask = dist > 0.5
    dist_masked = np.where(mask, dist, 1e10)
    x = rho * (1.0 - dist_masked)
    e = np.exp(x)
    # dE/dr = -rho * e * (2*e - 2) = -2*rho*e*(e-1)
    de_dr = -2.0 * rho * e * (e - 1.0)
    de_dr[~mask] = 0.0
    np.fill_diagonal(de_dr, 0.0)
    # Gradient: grad_i = sum_{j!=i} de_dr_ij * (r_i - r_j) / r_ij
    # diff[i,j] = r_i - r_j
    unit_vec = diff / dist_masked[:, :, np.newaxis]
    unit_vec[~mask] = 0.0
    grad = np.sum(de_dr[:, :, np.newaxis] * unit_vec, axis=1)
    return grad


def _fcc_points(n, spacing=1.0):
    """Generate FCC lattice points, take first n."""
    points = []
    s = spacing
    # FCC basis vectors
    basis = [
        (0, 0, 0),
        (0, 0.5*s, 0.5*s),
        (0.5*s, 0, 0.5*s),
        (0.5*s, 0.5*s, 0),
    ]
    # Generate grid
    max_coord = int(np.ceil(n ** (1/3))) + 2
    for i in range(-max_coord, max_coord+1):
        for j in range(-max_coord, max_coord+1):
            for k in range(-max_coord, max_coord+1):
                for b in basis:
                    p = (i*s + b[0], j*s + b[1], k*s + b[2])
                    points.append(p)
    points = np.array(points)
    # Center
    points -= points.mean(axis=0)
    # Sort by distance from center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    return points


def _bcc_points(n, spacing=1.0):
    """Generate BCC lattice points, take first n."""
    points = []
    s = spacing
    basis = [
        (0, 0, 0),
        (0.5*s, 0.5*s, 0.5*s),
    ]
    max_coord = int(np.ceil(n ** (1/3))) + 2
    for i in range(-max_coord, max_coord+1):
        for j in range(-max_coord, max_coord+1):
            for k in range(-max_coord, max_coord+1):
                for b in basis:
                    p = (i*s + b[0], j*s + b[1], k*s + b[2])
                    points.append(p)
    points = np.array(points)
    points -= points.mean(axis=0)
    dists = np.linalg.norm(points, axis=0) if points.ndim == 2 else np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    return points


def _icosahedral_shell(n, spacing=1.0):
    """Generate icosahedral-like shell structure."""
    # Start with a central atom and add shells
    points = [np.zeros(3)]
    # Icosahedral vertices (12)
    phi = (1 + np.sqrt(5)) / 2
    ico_vertices = [
        (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
        (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
        (0, 1/phi, phi), (0, -1/phi, phi), (0, 1/phi, -phi), (0, -1/phi, -phi),
        (1/phi, phi, 0), (-1/phi, phi, 0), (1/phi, -phi, 0), (-1/phi, -phi, 0),
        (phi, 0, 1/phi), (phi, 0, -1/phi), (-phi, 0, 1/phi), (-phi, 0, -1/phi),
    ]
    ico_vertices = np.array(ico_vertices, dtype=float)
    ico_vertices /= np.linalg.norm(ico_vertices, axis=1, keepdims=True)
    
    shell_radius = spacing
    for v in ico_vertices:
        if len(points) >= n:
            break
        points.append(v * shell_radius)
    
    # Add second shell if needed
    if len(points) < n:
        # Use octahedral directions for second shell
        oct_dirs = [
            (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1),
            (1,1,0)/np.sqrt(2), (-1,1,0)/np.sqrt(2), (1,-1,0)/np.sqrt(2), (-1,-1,0)/np.sqrt(2),
            (1,0,1)/np.sqrt(2), (-1,0,1)/np.sqrt(2), (1,0,-1)/np.sqrt(2), (-1,0,-1)/np.sqrt(2),
            (0,1,1)/np.sqrt(2), (0,-1,1)/np.sqrt(2), (0,1,-1)/np.sqrt(2), (0,-1,-1)/np.sqrt(2),
        ]
        shell2_radius = spacing * 1.6
        for d in oct_dirs:
            if len(points) >= n:
                break
            points.append(np.array(d) * shell2_radius)
    
    # Add more if still needed
    rng = np.random.default_rng(42)
    while len(points) < n:
        p = rng.normal(size=3) * spacing * 2.0
        if all(np.linalg.norm(p - np.array(q)) >= 0.5 * spacing for q in points):
            points.append(p)
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points


def _random_compact(n, spacing=1.0, seed=None):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.75
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - np.array(q)) >= 0.6 * spacing for q in points):
            points.append(p)
        attempts += 1
    # Fill remaining with random points
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _relax_with_gradient(pos, potential, rho, max_iter=500, tol=1e-8):
    """Relax using gradient descent with adaptive step size."""
    pos = pos.copy()
    n = len(pos)
    if n < 2:
        return 0.0, pos
    
    step = 0.05
    prev_energy = _morse_energy_positions(pos, rho)
    
    for it in range(max_iter):
        grad = _morse_grad_positions(pos, rho)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        # Adaptive step size
        new_pos = pos - step * grad
        # Ensure minimum distance constraint
        new_energy = _morse_energy_positions(new_pos, rho)
        
        if new_energy < prev_energy:
            pos = new_pos
            prev_energy = new_energy
            step = min(step * 1.1, 0.5)
        else:
            step *= 0.5
            if step < 1e-6:
                break
    
    return prev_energy, pos


def construct(instance):
    """Find low-energy Morse cluster configurations."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    spacing = 1.0
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # FCC lattice
    fcc = _fcc_points(size, spacing)
    initial_configs.append(fcc)
    
    # BCC lattice
    bcc = _bcc_points(size, spacing)
    initial_configs.append(bcc)
    
    # Icosahedral
    ico = _icosahedral_shell(size, spacing)
    initial_configs.append(ico)
    
    # Random compact (multiple seeds)
    for seed in [0, 1, 2, 3]:
        rc = _random_compact(size, spacing, seed=seed)
        initial_configs.append(rc)
    
    # Relax each configuration and keep the best
    best_energy = np.inf
    best_positions = initial_configs[0]
    
    for config in initial_configs:
        if _time_left() < 5.0:
            break
        
        # Use the provided relax function first
        try:
            energy, positions = relax(config.copy(), potential, parameter)
        except Exception:
            energy, positions = _relax_with_gradient(config, potential, parameter)
        
        # Verify minimum distance constraint
        if np.min([np.linalg.norm(positions[i] - positions[j]) for i in range(len(positions)) for j in range(i+1, len(positions))]) < 0.5:
            # Try to fix by slight perturbation and re-relax
            positions = positions + np.random.default_rng(0).normal(0, 0.01, positions.shape)
            try:
                energy, positions = relax(positions.copy(), potential, parameter)
            except Exception:
                energy, positions = _relax_with_gradient(positions, potential, parameter)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()
    
    # Basin hopping on the best configuration
    if _time_left() > 10.0:
        current_energy = best_energy
        current_pos = best_positions.copy()
        T = 2.0  # Initial temperature
        cooling = 0.95
        
        while _time_left() > 5.0 and T > 0.01:
            # Perturb
            noise_scale = np.sqrt(2 * T / size)
            perturbed = current_pos + np.random.default_rng().normal(0, noise_scale, current_pos.shape)
            
            # Relax perturbed
            try:
                new_energy, new_pos = relax(perturbed.copy(), potential, parameter)
            except Exception:
                new_energy, new_pos = _relax_with_gradient(perturbed, potential, parameter)
            
            # Accept if lower energy or with Boltzmann probability
            if new_energy < current_energy or np.random.random() < np.exp(-(new_energy - current_energy) / T):
                current_energy = new_energy
                current_pos = new_pos.copy()
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_positions = current_pos.copy()
            
            T *= cooling
    
    return best_positions
# EVOLVE-BLOCK-END

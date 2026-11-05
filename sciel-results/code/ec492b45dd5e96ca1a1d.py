"""
Morse cluster global minima search: combined model-and-optimize with diverse
crystalline/random starts and aggressive local relaxation.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Deadline guard
TOTAL_DEADLINE = 280.0
_START_TIME = time.time()

def _time_left():
    return max(0.0, TOTAL_DEADLINE - (time.time() - _START_TIME))

def _morse_energy_fast(pos, rho):
    """Vectorized Morse energy."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    x = rho * (1.0 - dists)
    ex = np.exp(x)
    e = ex * (ex - 2.0)
    iu = np.triu_indices(n, k=1)
    return float(np.sum(e[iu]))

def _morse_gradient_fast(pos, rho):
    """Compute gradient of Morse energy."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    x = rho * (1.0 - dists)
    ex = np.exp(x)
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = -rho * ex * (2.0 * ex - 1.0) / dists
    np.fill_diagonal(factor, 0.0)
    row_sums = np.sum(factor, axis=1)
    grad = pos * row_sums[:, np.newaxis] - factor.dot(pos)
    return grad

def _enforce_min_distance(pos, min_dist=0.5, max_iter=100):
    """Nudge atoms to satisfy minimum distance constraint."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        violated = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist and d > 1e-10:
                    direction = (pos[i] - pos[j]) / d
                    push = (min_dist - d) / 2.0 + 0.01
                    pos[i] += direction * push
                    pos[j] -= direction * push
                    violated = True
        if not violated:
            break
    return pos

def _check_min_distance(pos, min_dist=0.5):
    """Check if all pairs are at least min_dist apart."""
    n = len(pos)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(pos[i] - pos[j]) < min_dist:
                return False
    return True

def _generate_fcc(n, spacing=1.0):
    """Generate FCC-like configuration."""
    points = []
    # Generate a larger FCC lattice and select n closest to center
    a = spacing * np.sqrt(2)  # lattice constant for NN = spacing
    basis = np.array([[0,0,0], [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5]])
    for ix in range(-4, 5):
        for iy in range(-4, 5):
            for iz in range(-4, 5):
                for b in basis:
                    pt = np.array([ix, iy, iz]) * a + b * a
                    points.append(pt)
    points = np.array(points)
    center = points.mean(axis=0)
    points -= center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _generate_hcp(n, spacing=1.0):
    """Generate HCP-like configuration."""
    a = spacing
    c = a * np.sqrt(8.0/3.0)
    points = []
    radius = int(np.ceil(np.sqrt(n / 3.0))) + 2
    for layer in range(2 * radius + 1):
        z = (layer - radius) * c
        for q in range(-radius, radius + 1):
            for r in range(max(-radius, -q - radius), min(radius, -q + radius) + 1):
                x = a * (q + r/2.0)
                y = a * (np.sqrt(3.0)/2.0) * r
                if layer % 2 == 0:
                    points.append([x, y, z])
                else:
                    points.append([x + a/2.0, y + a*np.sqrt(3.0)/6.0, z])
    points = np.array(points)
    center = points.mean(axis=0)
    points -= center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _generate_bcc(n, spacing=1.0):
    """Generate BCC-like configuration."""
    a = spacing * (2.0 / np.sqrt(3.0))  # lattice constant for NN = spacing
    points = []
    for ix in range(-4, 5):
        for iy in range(-4, 5):
            for iz in range(-4, 5):
                points.append([ix*a, iy*a, iz*a])
                points.append([(ix+0.5)*a, (iy+0.5)*a, (iz+0.5)*a])
    points = np.array(points)
    center = points.mean(axis=0)
    points -= center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _generate_simple_cubic(n, spacing=1.0):
    """Generate simple cubic configuration."""
    points = []
    for ix in range(-4, 5):
        for iy in range(-4, 5):
            for iz in range(-4, 5):
                points.append([ix*spacing, iy*spacing, iz*spacing])
    points = np.array(points)
    center = points.mean(axis=0)
    points -= center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _generate_2d_hexagonal(n, spacing=1.0):
    """2D hexagonal (triangular lattice) cluster."""
    points = []
    a = spacing
    radius = int(np.ceil(np.sqrt(n / 2.5))) + 2
    for q in range(-radius, radius + 1):
        for r in range(max(-radius, -q - radius), min(radius, -q + radius) + 1):
            x = a * (q + r/2.0)
            y = a * (np.sqrt(3.0)/2.0) * r
            points.append([x, y, 0.0])
    points = np.array(points)
    center = points.mean(axis=0)
    points -= center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _generate_spherical_shell(n, spacing=1.0):
    """Points on/near a spherical shell with Fibonacci distribution."""
    if n == 1:
        return np.array([[0,0,0]])
    golden = (1 + np.sqrt(5)) / 2
    points = []
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = np.sqrt(1 - z*z)
        phi = 2 * np.pi * i / golden
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        points.append([x, y, z])
    points = np.array(points)
    # Scale so typical NN ~ spacing
    scale = spacing * 0.8 * np.sqrt(4.0 * np.pi / n)
    points *= scale
    return points

def _generate_linear_chain(n, spacing=1.0):
    """Linear chain."""
    pts = []
    total_len = (n-1) * spacing
    start = -total_len / 2.0
    for i in range(n):
        pts.append([start + i * spacing, 0, 0])
    return np.array(pts)

def _generate_random_compact(n, rng, spacing=1.0):
    """Random compact cluster with minimum distance."""
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 20000:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            min_dist = min([np.linalg.norm(p - q) for q in points], default=np.inf)
            if min_dist >= 0.65 * spacing:
                points.append(p)
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)

def _generate_jittered(base, rng, jitter=0.15):
    """Add random jitter to a base configuration."""
    return base + rng.normal(0, jitter, size=base.shape)

def _relax_l_bfgsb(pos, rho, max_iter=3000):
    """Relax with L-BFGS-B using our own energy and gradient."""
    def objective_and_grad(x):
        pos2d = x.reshape(-1, 3)
        e = _morse_energy_fast(pos2d, rho)
        g = _morse_gradient_fast(pos2d, rho).ravel()
        return e, g
    
    x0 = pos.ravel()
    result = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-10})
    return result.x.reshape(-1, 3), float(result.fun)

def _local_relax(pos, rho):
    """Use the provided relax function."""
    try:
        energy, relaxed_pos = relax(pos.copy(), 'morse', rho)
        return float(energy), relaxed_pos
    except Exception:
        return np.inf, pos

def construct(instance):
    """
    Find low-energy Morse cluster configurations using diverse initializations
    and aggressive local relaxation.
    """
    global _START_TIME
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    best_energy = np.inf
    best_positions = None
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # Crystalline templates
    templates = [
        ('fcc', _generate_fcc),
        ('hcp', _generate_hcp),
        ('bcc', _generate_bcc),
        ('sc', _generate_simple_cubic),
        ('hex2d', _generate_2d_hexagonal),
        ('shell', _generate_spherical_shell),
        ('chain', _generate_linear_chain),
    ]
    
    for name, gen_func in templates:
        try:
            pos = gen_func(n, spacing=1.0)
            if len(pos) == n:
                initial_configs.append((name, pos))
        except Exception:
            pass
    
    # Jittered versions of best templates
    for name, base in initial_configs[:5]:
        for jitter in [0.1, 0.2, 0.3]:
            try:
                perturbed = _generate_jittered(base, rng, jitter=jitter)
                initial_configs.append((f'{name}_j{jitter}', perturbed))
            except Exception:
                pass
    
    # Random compact clusters with different seeds
    for i in range(8):
        try:
            rand = _generate_random_compact(n, rng, spacing=1.0)
            initial_configs.append((f'rand_{i}', rand))
        except Exception:
            pass
    
    # Quick energy evaluation and sort
    config_energies = []
    for name, pos in initial_configs:
        try:
            e = _morse_energy_fast(pos, rho)
            config_energies.append((e, name, pos))
        except Exception:
            pass
    
    config_energies.sort(key=lambda x: x[0])
    
    # Process top configurations with local relaxation
    max_configs = min(len(config_energies), 15)
    
    for e_init, name, pos in config_energies[:max_configs]:
        if _time_left() < 15.0:
            break
        
        # Try provided relax function
        try:
            energy, relaxed_pos = _local_relax(pos, rho)
            if energy < best_energy and _check_min_distance(relaxed_pos, 0.5):
                best_energy = energy
                best_positions = relaxed_pos.copy()
            elif energy < best_energy:
                relaxed_pos = _enforce_min_distance(relaxed_pos, 0.5)
                e_check = _morse_energy_fast(relaxed_pos, rho)
                if e_check < best_energy:
                    best_energy = e_check
                    best_positions = relaxed_pos.copy()
        except Exception:
            pass
        
        # If we have time, do L-BFGS-B refinement on the best found so far
        if _time_left() > 30.0 and best_positions is not None:
            try:
                refined, e_refined = _relax_l_bfgsb(best_positions.copy(), rho, max_iter=2000)
                if _check_min_distance(refined, 0.5) and e_refined < best_energy:
                    best_energy = e_refined
                    best_positions = refined.copy()
                elif e_refined < best_energy:
                    refined = _enforce_min_distance(refined, 0.5)
                    e_check = _morse_energy_fast(refined, rho)
                    if e_check < best_energy:
                        best_energy = e_check
                        best_positions = refined.copy()
            except Exception:
                pass
    
    # Random restarts from best found
    if _time_left() > 20.0 and best_positions is not None:
        n_restarts = max(1, min(int(_time_left() / 12.0), 12))
        for i in range(n_restarts):
            if _time_left() < 10.0:
                break
            try:
                scale = 0.2 + 0.1 * i
                perturbed = best_positions + rng.normal(0, scale, size=best_positions.shape)
                energy, relaxed_pos = _local_relax(perturbed, rho)
                if energy < best_energy and _check_min_distance(relaxed_pos, 0.5):
                    best_energy = energy
                    best_positions = relaxed_pos.copy()
                elif energy < best_energy:
                    relaxed_pos = _enforce_min_distance(relaxed_pos, 0.5)
                    e_check = _morse_energy_fast(relaxed_pos, rho)
                    if e_check < best_energy:
                        best_energy = e_check
                        best_positions = relaxed_pos.copy()
            except Exception:
                pass
    
    # Final refinement if time permits
    if _time_left() > 15.0 and best_positions is not None:
        try:
            refined, e_refined = _relax_l_bfgsb(best_positions.copy(), rho, max_iter=3000)
            if _check_min_distance(refined, 0.5) and e_refined < best_energy:
                best_energy = e_refined
                best_positions = refined.copy()
            elif e_refined < best_energy:
                refined = _enforce_min_distance(refined, 0.5)
                e_check = _morse_energy_fast(refined, rho)
                if e_check < best_energy:
                    best_energy = e_check
                    best_positions = refined.copy()
        except Exception:
            pass
    
    # Fallback if no valid result
    if best_positions is None:
        best_positions = _generate_random_compact(n, rng, spacing=1.0)
        try:
            _, best_positions = _local_relax(best_positions, rho)
        except Exception:
            pass
        best_positions = _enforce_min_distance(best_positions, 0.5)
    
    # Final min-distance enforcement
    best_positions = _enforce_min_distance(best_positions, 0.5)
    
    return best_positions
# EVOLVE-BLOCK-END

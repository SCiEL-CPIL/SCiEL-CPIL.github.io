"""Rewritten program using symmetry-constrained variational optimization for Dzugutov clusters."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from scipy.optimize import minimize, differential_evolution
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
TIME_LIMIT = 290.0  # Safety margin before 300s limit


def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)


def _dzugutov_energy(pos):
    """Compute Dzugutov potential energy."""
    n = len(pos)
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(pos[i] - pos[j])
            if r < 1e-10:
                continue
            if r < a:
                total += A * (r ** (-m) - B) * np.exp(c / (r - a))
            elif r < b:
                total += B * np.exp(d / (r - b))
    return total


def _dzugutov_gradient(pos):
    """Compute gradient of Dzugutov potential energy."""
    n = len(pos)
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    grad = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            rij = pos[i] - pos[j]
            r = np.linalg.norm(rij)
            if r < 1e-10:
                continue
            if r < a:
                exp_term = np.exp(c / (r - a))
                dV_dr = A * (-m * r ** (-m - 1)) * exp_term + A * (r ** (-m) - B) * exp_term * (-c / (r - a) ** 2)
            elif r < b:
                exp_term = np.exp(d / (r - b))
                dV_dr = B * exp_term * (-d / (r - b) ** 2)
            else:
                continue
            factor = dV_dr / r
            grad[i] += factor * rij
            grad[j] -= factor * rij
    return grad


def _get_icosahedron_vertices():
    """Return the 12 vertices of a regular icosahedron on the unit sphere."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts


def _get_dodecahedron_vertices():
    """Return the 20 vertices of a regular dodecahedron on the unit sphere."""
    phi = (1 + np.sqrt(5)) / 2
    verts = []
    # 8 cube vertices
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                verts.append([x, y, z])
    # 12 rectangle vertices
    for x in [-1/phi, 1/phi]:
        for y in [-phi, phi]:
            for z in [0, 0]:
                pass
    # Proper dodecahedron vertices
    verts = []
    # 8 cube vertices
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                verts.append([x, y, z])
    # 12 vertices from rectangles
    for x in [0, 0]:
        for y in [-1/phi, 1/phi]:
            for z in [-phi, phi]:
                verts.append([x, y, z])
    for x in [-1/phi, 1/phi]:
        for y in [-phi, phi]:
            for z in [0, 0]:
                verts.append([x, y, z])
    for x in [-phi, phi]:
        for y in [0, 0]:
            for z in [-1/phi, 1/phi]:
                verts.append([x, y, z])
    verts = np.array(verts, dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts


def _icosahedral_shells(n, r1=1.0, r2=1.7, r3=2.3, twist1=0.0, twist2=0.0, twist3=0.0):
    """Generate icosahedral cluster with parameterized shell radii and twists."""
    verts = _get_icosahedron_vertices()
    points = [np.array([0.0, 0.0, 0.0])]  # center
    
    # Shell 1: 12 icosahedron vertices
    if n > 1:
        shell1 = verts * r1
        if abs(twist1) > 1e-10:
            axis = np.array([0.0, 0.0, 1.0])
            c, s = np.cos(twist1), np.sin(twist1)
            R = np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])
            shell1 = shell1 @ R.T
        points.extend(shell1)
    
    if len(points) >= n:
        return np.array(points[:n])
    
    # Shell 2: 42 atoms (30 edge midpoints + 12 face centers)
    if n > 13:
        edge_mids = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if 0.95 < d < 1.15:
                    mid = (verts[i] + verts[j]) / 2.0
                    mid = mid / np.linalg.norm(mid)
                    edge_mids.append(mid)
        
        face_centers = []
        for i in range(12):
            for j in range(i + 1, 12):
                for k in range(j + 1, 12):
                    d1 = np.linalg.norm(verts[i] - verts[j])
                    d2 = np.linalg.norm(verts[j] - verts[k])
                    d3 = np.linalg.norm(verts[k] - verts[i])
                    if 0.95 < d1 < 1.15 and 0.95 < d2 < 1.15 and 0.95 < d3 < 1.15:
                        center = (verts[i] + verts[j] + verts[k]) / 3.0
                        center = center / np.linalg.norm(center)
                        face_centers.append(center)
        
        shell2_dirs = np.array(edge_mids + face_centers)
        shell2 = shell2_dirs * r2
        if abs(twist2) > 1e-10:
            axis = np.array([0.0, 0.0, 1.0])
            c, s = np.cos(twist2), np.sin(twist2)
            R = np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])
            shell2 = shell2 @ R.T
        points.extend(shell2)
    
    if len(points) >= n:
        return np.array(points[:n])
    
    # Shell 3: 120 atoms - use Fibonacci sphere distribution
    if n > 55:
        shell3 = []
        for i in range(120):
            y = 1 - 2 * (i + 0.5) / 120
            r = np.sqrt(max(0, 1 - y ** 2))
            theta = 2 * np.pi * i * 0.381966
            p = np.array([r * np.cos(theta), y, r * np.sin(theta)])
            shell3.append(p)
        shell3 = np.array(shell3) * r3
        if abs(twist3) > 1e-10:
            axis = np.array([0.0, 0.0, 1.0])
            c, s = np.cos(twist3), np.sin(twist3)
            R = np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])
            shell3 = shell3 @ R.T
        points.extend(shell3)
    
    if len(points) >= n:
        return np.array(points[:n])
    
    rng = np.random.default_rng(42)
    while len(points) < n:
        idx = rng.integers(0, len(points))
        p = points[idx] + rng.normal(0, 0.05, 3)
        points.append(p)
    
    return np.array(points[:n])


def _dodecahedral_shells(n, r1=1.0, r2=1.8, r3=2.4):
    """Generate dodecahedral cluster."""
    verts = _get_dodecahedron_vertices()
    points = [np.array([0.0, 0.0, 0.0])]
    
    if n > 1:
        shell1 = verts * r1
        points.extend(shell1)
    
    if len(points) >= n:
        return np.array(points[:n])
    
    # Shell 2: Add points between dodecahedron vertices
    if n > 21:
        shell2 = []
        # Use icosahedron vertices scaled for shell 2
        ico_verts = _get_icosahedron_vertices()
        for v in ico_verts:
            shell2.append(v * r2)
        # Add more points
        for i in range(30):
            y = 1 - 2 * (i + 0.5) / 30
            r = np.sqrt(max(0, 1 - y ** 2))
            theta = 2 * np.pi * i * 0.381966
            p = np.array([r * np.cos(theta), y, r * np.sin(theta)])
            shell2.append(p * r2)
        points.extend(shell2)
    
    if len(points) >= n:
        return np.array(points[:n])
    
    # Shell 3
    if n > 55:
        shell3 = []
        for i in range(120):
            y = 1 - 2 * (i + 0.5) / 120
            r = np.sqrt(max(0, 1 - y ** 2))
            theta = 2 * np.pi * i * 0.381966
            p = np.array([r * np.cos(theta), y, r * np.sin(theta)])
            shell3.append(p * r3)
        points.extend(shell3)
    
    if len(points) >= n:
        return np.array(points[:n])
    
    rng = np.random.default_rng(42)
    while len(points) < n:
        idx = rng.integers(0, len(points))
        p = points[idx] + rng.normal(0, 0.05, 3)
        points.append(p)
    
    return np.array(points[:n])


def _cuboctahedral_shells(n, r1=1.0, r2=1.8, r3=2.5):
    """Generate cuboctahedral-like cluster."""
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Shell 1: 12 cuboctahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    shell1 = []
    # 6 face centers of cube
    for axis in range(3):
        for sign in [-1, 1]:
            p = np.zeros(3)
            p[axis] = sign
            shell1.append(p)
    # 8 edge midpoints of cube
    for i in range(3):
        for j in range(3):
            if i != j:
                for si in [-1, 1]:
                    for sj in [-1, 1]:
                        p = np.zeros(3)
                        p[i] = si * 0.5
                        p[j] = sj * 0.5
                        shell1.append(p)
    # 8 vertex positions
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                shell1.append(np.array([x, y, z]))
    
    shell1 = np.array(shell1, dtype=float)
    norms = np.linalg.norm(shell1, axis=1, keepdims=True)
    shell1 = shell1 / norms * r1
    points.extend(shell1)
    
    if len(points) >= n:
        return np.array(points[:n])
    
    # Shell 2
    if n > 13:
        shell2 = []
        for i in range(42):
            y = 1 - 2 * (i + 0.5) / 42
            r = np.sqrt(max(0, 1 - y ** 2))
            theta = 2 * np.pi * i * 0.381966
            p = np.array([r * np.cos(theta), y, r * np.sin(theta)])
            shell2.append(p * r2)
        points.extend(shell2)
    
    if len(points) >= n:
        return np.array(points[:n])
    
    # Shell 3
    if n > 55:
        shell3 = []
        for i in range(120):
            y = 1 - 2 * (i + 0.5) / 120
            r = np.sqrt(max(0, 1 - y ** 2))
            theta = 2 * np.pi * i * 0.381966
            p = np.array([r * np.cos(theta), y, r * np.sin(theta)])
            shell3.append(p * r3)
        points.extend(shell3)
    
    if len(points) >= n:
        return np.array(points[:n])
    
    rng = np.random.default_rng(42)
    while len(points) < n:
        idx = rng.integers(0, len(points))
        p = points[idx] + rng.normal(0, 0.05, 3)
        points.append(p)
    
    return np.array(points[:n])


def _random_compact(n, scale=1.0, seed=42):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = scale * 1.2
    points = []
    min_dist = 0.6
    attempts = 0
    max_attempts = n * 500
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    
    if len(points) < n:
        min_dist = 0.5
        attempts = 0
        while len(points) < n and attempts < max_attempts:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= min_dist for q in points):
                    points.append(p)
            attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-radius * 0.5, radius * 0.5, 3)
        points.append(p)
    
    return np.array(points[:n])


def _validate_positions(pos, n):
    """Check minimum distance constraint."""
    if len(pos) != n:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(pos[i] - pos[j]) < 0.5:
                return False
    return True


def _relax_and_evaluate(pos, n, potential, parameter):
    """Relax positions and return energy and relaxed positions."""
    try:
        energy, relaxed_pos = relax(pos, potential, parameter)
        if _validate_positions(relaxed_pos, n):
            return energy, relaxed_pos
        else:
            return _dzugutov_energy(pos), pos
    except Exception:
        return _dzugutov_energy(pos), pos


def _basin_hopping(pos, n, potential, parameter, n_steps=15, step_size=0.15, seed=None):
    """Perform basin hopping starting from given positions."""
    if _time_remaining() < 15:
        return pos, _dzugutov_energy(pos)
    
    rng = np.random.default_rng(seed if seed is not None else int(time.time() * 1000) % 10000)
    
    current_energy, current_pos = _relax_and_evaluate(pos, n, potential, parameter)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    for step in range(n_steps):
        if _time_remaining() < 10:
            break
        
        # Perturb
        perturbed = current_pos + rng.normal(0, step_size, current_pos.shape)
        
        # Relax
        energy, relaxed_pos = _relax_and_evaluate(perturbed, n, potential, parameter)
        
        # Metropolis acceptance
        if energy < current_energy:
            current_energy = energy
            current_pos = relaxed_pos
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos.copy()
        else:
            delta = energy - current_energy
            if delta < 1.0:
                accept_prob = np.exp(-delta / 0.15)
                if rng.random() < accept_prob:
                    current_energy = energy
                    current_pos = relaxed_pos
    
    return best_pos, best_energy


def _generate_seeds(n, base_scale):
    """Generate diverse initial seeds for optimization."""
    seeds = []
    
    # Icosahedral seeds
    seeds.append(_icosahedral_shells(n, r1=base_scale, r2=base_scale*1.7, r3=base_scale*2.3))
    seeds.append(_icosahedral_shells(n, r1=base_scale*0.9, r2=base_scale*1.6, r3=base_scale*2.2))
    seeds.append(_icosahedral_shells(n, r1=base_scale*1.1, r2=base_scale*1.8, r3=base_scale*2.4))
    seeds.append(_icosahedral_shells(n, r1=base_scale, r2=base_scale*1.7, r3=base_scale*2.3,
                                      twist1=0.1, twist2=0.05, twist3=0.03))
    
    # Dodecahedral seeds
    seeds.append(_dodecahedral_shells(n, r1=base_scale, r2=base_scale*1.8, r3=base_scale*2.4))
    seeds.append(_dodecahedral_shells(n, r1=base_scale*0.95, r2=base_scale*1.7, r3=base_scale*2.3))
    
    # Cuboctahedral seeds
    seeds.append(_cuboctahedral_shells(n, r1=base_scale, r2=base_scale*1.8, r3=base_scale*2.5))
    
    # Random compact seeds
    seeds.append(_random_compact(n, scale=base_scale, seed=n))
    seeds.append(_random_compact(n, scale=base_scale*1.05, seed=n+100))
    seeds.append(_random_compact(n, scale=base_scale*0.95, seed=n+200))
    
    return seeds


def construct(instance):
    """Construct low-energy Dzugutov cluster using multi-polyhedral basin hopping."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    def _update_best(energy, positions):
        nonlocal best_energy, best_positions
        if positions is None or len(positions) != size:
            return
        if not _validate_positions(positions, size):
            return
        if energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()
    
    base_scale = size ** (1.0 / 3.0) * 0.38
    
    # Generate diverse seeds
    seeds = _generate_seeds(size, base_scale)
    
    # Relax all seeds first to get good starting points
    relaxed_seeds = []
    for seed_idx, seed in enumerate(seeds):
        if _time_remaining() < 30:
            break
        energy, relaxed_pos = _relax_and_evaluate(seed, size, potential, parameter)
        relaxed_seeds.append((energy, relaxed_pos))
        _update_best(energy, relaxed_pos)
    
    # Sort by energy and do basin hopping from best seeds
    relaxed_seeds.sort(key=lambda x: x[0])
    
    # Basin hopping from top seeds
    for energy, pos in relaxed_seeds[:6]:
        if _time_remaining() < 40:
            break
        
        # Multiple basin hopping runs with different step sizes
        for step_size in [0.12, 0.18, 0.25]:
            if _time_remaining() < 20:
                break
            bh_pos, bh_energy = _basin_hopping(pos, size, potential, parameter,
                                                 n_steps=12, step_size=step_size,
                                                 seed=int(hash((size, step_size)) % 100000))
            _update_best(bh_energy, bh_pos)
    
    # If we have time, do more basin hopping from best found
    if best_positions is not None and _time_remaining() > 50:
        for trial in range(5):
            if _time_remaining() < 25:
                break
            # Perturb best and do basin hopping
            rng = np.random.default_rng(42 + trial)
            perturbed = best_positions + rng.normal(0, 0.1, best_positions.shape)
            bh_pos, bh_energy = _basin_hopping(perturbed, size, potential, parameter,
                                                 n_steps=15, step_size=0.15,
                                                 seed=42 + trial)
            _update_best(bh_energy, bh_pos)
    
    # Final relaxation of best
    if best_positions is not None and _time_remaining() > 10:
        energy, relaxed_pos = _relax_and_evaluate(best_positions, size, potential, parameter)
        _update_best(energy, relaxed_pos)
    
    if best_positions is None:
        # Fallback: return ideal icosahedron
        best_positions = _icosahedral_shells(size, r1=base_scale, r2=base_scale*1.7, r3=base_scale*2.3)
    
    return best_positions
# EVOLVE-BLOCK-END

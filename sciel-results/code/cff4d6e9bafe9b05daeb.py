"""Dzugutov cluster global minima search with improved basin hopping and L-BFGS refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
TIME_LIMIT = 295.0  # Safety margin before 330s limit


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


def _check_min_dist(pos, min_d=0.5):
    """Check if all pairwise distances are >= min_d."""
    n = len(pos)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(pos[i] - pos[j]) < min_d:
                return False
    return True


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
    # 12 vertices from three rectangles
    for x in [0]:
        for y in [-1/phi, 1/phi]:
            for z in [-phi, phi]:
                verts.append([x, y, z])
    for x in [-1/phi, 1/phi]:
        for y in [-phi, phi]:
            for z in [0]:
                verts.append([x, y, z])
    for x in [-phi, phi]:
        for y in [-1/phi, 1/phi]:
            for z in [0]:
                verts.append([x, y, z])
    verts = np.array(verts, dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts


def _generate_macky_icosahedron(n, scale=1.0):
    """Generate a Mackay icosahedral cluster with n atoms.
    
    Shell structure:
    - Center: 1 atom
    - Shell 1 (12 atoms): icosahedron vertices
    - Shell 2 (42 atoms): 30 edge midpoints + 12 face centers
    - Shell 3 (120 atoms): more complex structure
    """
    ico_verts = _get_icosahedron_vertices()
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Shell 1: 12 icosahedron vertices
    if n > 1:
        shell1 = ico_verts * scale
        points.extend(shell1.tolist())
    
    if len(points) >= n:
        return np.array(points[:n])
    
    # Shell 2: 42 atoms
    if n > 13:
        # Edge midpoints (30 edges)
        edge_mids = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(ico_verts[i] - ico_verts[j])
                if 0.95 < d < 1.15:  # Adjacent vertices
                    mid = (ico_verts[i] + ico_verts[j]) / 2.0
                    mid = mid / np.linalg.norm(mid)
                    edge_mids.append(mid)
        
        # Face centers (20 faces)
        face_centers = []
        for i in range(12):
            for j in range(i + 1, 12):
                for k in range(j + 1, 12):
                    d1 = np.linalg.norm(ico_verts[i] - ico_verts[j])
                    d2 = np.linalg.norm(ico_verts[j] - ico_verts[k])
                    d3 = np.linalg.norm(ico_verts[k] - ico_verts[i])
                    if 0.95 < d1 < 1.15 and 0.95 < d2 < 1.15 and 0.95 < d3 < 1.15:
                        center = (ico_verts[i] + ico_verts[j] + ico_verts[k]) / 3.0
                        center = center / np.linalg.norm(center)
                        face_centers.append(center)
        
        shell2_dirs = np.array(edge_mids + face_centers)
        shell2 = shell2_dirs * scale * 1.7013  # Golden ratio related
        points.extend(shell2.tolist())
    
    if len(points) >= n:
        return np.array(points[:n])
    
    # Shell 3: 120 atoms - use dodecahedron-based positions
    if n > 55:
        # Use a combination of dodecahedron vertices and other positions
        dodec_verts = _get_dodecahedron_vertices()
        
        # Create shell 3 positions
        shell3 = []
        # 20 dodecahedron vertices
        shell3.extend((dodec_verts * scale * 2.3979).tolist())
        
        # Add more points using Fibonacci sphere for remaining
        remaining = n - len(points)
        if remaining > 0:
            for i in range(remaining):
                y = 1 - 2 * (i + 0.5) / max(remaining, 1)
                r = np.sqrt(max(0, 1 - y ** 2))
                theta = 2 * np.pi * i * 0.381966  # Golden angle
                p = np.array([r * np.cos(theta), y, r * np.sin(theta)])
                shell3.append(p * scale * 2.3979)
        
        points.extend(shell3[:n - len(points)])
    
    # If still not enough, add perturbed points
    rng = np.random.default_rng(42)
    while len(points) < n:
        idx = rng.integers(0, len(points))
        p = points[idx] + rng.normal(0, 0.05, 3)
        points.append(p)
    
    return np.array(points[:n])


def _generate_fcc_cluster(n, scale=1.0):
    """Generate an FCC-like cluster."""
    points = []
    # FCC lattice points within a sphere
    a = scale * 1.5
    max_r = scale * 2.5
    for ix in range(-3, 4):
        for iy in range(-3, 4):
            for iz in range(-3, 4):
                # FCC positions
                p1 = np.array([ix * a, iy * a, iz * a])
                p2 = np.array([(ix + 0.5) * a, (iy + 0.5) * a, iz * a])
                p3 = np.array([(ix + 0.5) * a, iy * a, (iz + 0.5) * a])
                p4 = np.array([ix * a, (iy + 0.5) * a, (iz + 0.5) * a])
                for p in [p1, p2, p3, p4]:
                    if np.linalg.norm(p) <= max_r:
                        points.append(p)
    
    # Sort by distance from center and take n closest
    points = np.array(points)
    if len(points) > n:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:n]
        points = points[idx]
    
    # If not enough, add more
    rng = np.random.default_rng(42)
    while len(points) < n:
        p = rng.uniform(-max_r, max_r, 3)
        points = np.vstack([points, p])
    
    return points[:n]


def _generate_bcc_cluster(n, scale=1.0):
    """Generate a BCC-like cluster."""
    points = []
    a = scale * 1.5
    max_r = scale * 2.5
    for ix in range(-3, 4):
        for iy in range(-3, 4):
            for iz in range(-3, 4):
                p1 = np.array([ix * a, iy * a, iz * a])
                p2 = np.array([(ix + 0.5) * a, (iy + 0.5) * a, (iz + 0.5) * a])
                for p in [p1, p2]:
                    if np.linalg.norm(p) <= max_r:
                        points.append(p)
    
    points = np.array(points)
    if len(points) > n:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:n]
        points = points[idx]
    
    rng = np.random.default_rng(42)
    while len(points) < n:
        p = rng.uniform(-max_r, max_r, 3)
        points = np.vstack([points, p])
    
    return points[:n]


def _random_compact(n, scale=1.0, seed=42):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = scale * 1.5
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


def _relax_with_fallback(pos, potential, parameter, max_iter=2000):
    """Relax positions with fallback to gradient descent if needed."""
    try:
        energy, relaxed_pos = relax(pos, potential, parameter)
        if _check_min_dist(relaxed_pos):
            return energy, relaxed_pos
    except Exception:
        pass
    
    # Fallback: L-BFGS
    try:
        def objective(x):
            pos = x.reshape(-1, 3)
            return _dzugutov_energy(pos)
        
        def gradient(x):
            pos = x.reshape(-1, 3)
            return _dzugutov_gradient(pos).reshape(-1)
        
        x0 = pos.flatten()
        result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
        relaxed_pos = result.x.reshape(-1, 3)
        if _check_min_dist(relaxed_pos):
            return result.fun, relaxed_pos
    except Exception:
        pass
    
    return _dzugutov_energy(pos), pos


def _l_bfgs_refine(pos, max_iter=1000):
    """Refine positions using L-BFGS."""
    try:
        def objective(x):
            p = x.reshape(-1, 3)
            return _dzugutov_energy(p)
        
        def gradient(x):
            p = x.reshape(-1, 3)
            return _dzugutov_gradient(p).reshape(-1)
        
        x0 = pos.flatten()
        result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
        refined_pos = result.x.reshape(-1, 3)
        if _check_min_dist(refined_pos):
            return result.fun, refined_pos
    except Exception:
        pass
    return _dzugutov_energy(pos), pos


def _basin_hopping(pos, n, potential, parameter, n_steps=15, step_size=0.15, 
                   temperature=0.05, seed=None):
    """Perform basin hopping starting from given positions."""
    if _time_remaining() < 20:
        return pos, _dzugutov_energy(pos)
    
    rng = np.random.default_rng(seed)
    current_energy, current_pos = _relax_with_fallback(pos, potential, parameter)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    for step in range(n_steps):
        if _time_remaining() < 10:
            break
        
        # Perturb with Gaussian noise
        noise = rng.normal(0, step_size, current_pos.shape)
        perturbed = current_pos + noise
        
        # Relax
        energy, relaxed_pos = _relax_with_fallback(perturbed, potential, parameter)
        
        # Check minimum distance
        if not _check_min_dist(relaxed_pos):
            continue
        
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
                accept_prob = np.exp(-delta / temperature)
                if rng.random() < accept_prob:
                    current_energy = energy
                    current_pos = relaxed_pos
    
    return best_pos, best_energy


def _simulated_annealing(pos, n, potential, parameter, n_steps=20, 
                          t_start=0.2, t_end=0.01, seed=None):
    """Perform simulated annealing."""
    if _time_remaining() < 20:
        return pos, _dzugutov_energy(pos)
    
    rng = np.random.default_rng(seed)
    current_energy, current_pos = _relax_with_fallback(pos, potential, parameter)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    for step in range(n_steps):
        if _time_remaining() < 10:
            break
        
        # Temperature schedule
        t = t_start * (t_end / t_start) ** (step / max(n_steps - 1, 1))
        
        # Perturb
        noise = rng.normal(0, 0.1, current_pos.shape)
        perturbed = current_pos + noise
        
        # Relax
        energy, relaxed_pos = _relax_with_fallback(perturbed, potential, parameter)
        
        if not _check_min_dist(relaxed_pos):
            continue
        
        # Metropolis acceptance
        if energy < current_energy:
            current_energy = energy
            current_pos = relaxed_pos
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos.copy()
        else:
            delta = energy - current_energy
            accept_prob = np.exp(-delta / t)
            if rng.random() < accept_prob:
                current_energy = energy
                current_pos = relaxed_pos
    
    return best_pos, best_energy


def _kick_and_relax(pos, n, potential, parameter, n_kicks=5, kick_size=0.2, seed=None):
    """Multiple random kicks followed by relaxation."""
    if _time_remaining() < 20:
        return pos, _dzugutov_energy(pos)
    
    rng = np.random.default_rng(seed)
    best_energy, best_pos = _relax_with_fallback(pos, potential, parameter)
    
    for _ in range(n_kicks):
        if _time_remaining() < 10:
            break
        
        noise = rng.normal(0, kick_size, pos.shape)
        kicked = pos + noise
        energy, relaxed_pos = _relax_with_fallback(kicked, potential, parameter)
        
        if _check_min_dist(relaxed_pos) and energy < best_energy:
            best_energy = energy
            best_pos = relaxed_pos
    
    return best_pos, best_energy


def _generate_seeds(n, base_scale):
    """Generate diverse initial seeds for optimization."""
    seeds = []
    
    # Mackay icosahedron
    seeds.append(_generate_macky_icosahedron(n, scale=base_scale))
    
    # Compressed icosahedron
    seeds.append(_generate_macky_icosahedron(n, scale=base_scale * 0.9))
    
    # Expanded icosahedron
    seeds.append(_generate_macky_icosahedron(n, scale=base_scale * 1.1))
    
    # FCC
    seeds.append(_generate_fcc_cluster(n, scale=base_scale))
    
    # BCC
    seeds.append(_generate_bcc_cluster(n, scale=base_scale))
    
    # Random compact
    seeds.append(_random_compact(n, scale=base_scale, seed=n))
    
    # Another random
    seeds.append(_random_compact(n, scale=base_scale * 1.1, seed=n + 100))
    
    return seeds


def construct(instance):
    """Construct low-energy Dzugutov cluster using multi-strategy optimization."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    def _update_best(energy, positions):
        nonlocal best_energy, best_positions
        if positions is None or len(positions) != size:
            return
        if not _check_min_dist(positions):
            return
        if energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()
    
    base_scale = size ** (1.0 / 3.0) * 0.38
    
    # Generate seeds
    seeds = _generate_seeds(size, base_scale)
    
    # Process each seed with multiple optimization strategies
    for seed_idx, seed in enumerate(seeds):
        if _time_remaining() < 30:
            break
        
        # First, relax the seed
        energy, relaxed_pos = _relax_with_fallback(seed, potential, parameter)
        _update_best(energy, relaxed_pos)
        
        if relaxed_pos is None or not _check_min_dist(relaxed_pos):
            continue
        
        # Basin hopping from this seed
        if _time_remaining() > 40:
            bh_pos, bh_energy = _basin_hopping(relaxed_pos, size, potential, parameter,
                                                 n_steps=12, step_size=0.12,
                                                 temperature=0.05, seed=seed_idx)
            _update_best(bh_energy, bh_pos)
        
        # Simulated annealing from this seed
        if _time_remaining() > 40:
            sa_pos, sa_energy = _simulated_annealing(relaxed_pos, size, potential, parameter,
                                                       n_steps=15, t_start=0.15, t_end=0.01,
                                                       seed=seed_idx + 1000)
            _update_best(sa_energy, sa_pos)
        
        # Kick and relax
        if _time_remaining() > 30:
            kr_pos, kr_energy = _kick_and_relax(relaxed_pos, size, potential, parameter,
                                                   n_kicks=4, kick_size=0.15,
                                                   seed=seed_idx + 2000)
            _update_best(kr_energy, kr_pos)
    
    # Additional refinement from best found
    if best_positions is not None and _time_remaining() > 30:
        # More basin hopping from best
        for _ in range(2):
            if _time_remaining() < 20:
                break
            bh_pos, bh_energy = _basin_hopping(best_positions, size, potential, parameter,
                                                 n_steps=15, step_size=0.10,
                                                 temperature=0.03, seed=5000)
            _update_best(bh_energy, bh_pos)
        
        # L-BFGS refinement
        if _time_remaining() > 15:
            refined_energy, refined_pos = _l_bfgs_refine(best_positions, max_iter=500)
            _update_best(refined_energy, refined_pos)
    
    # Final relaxation
    if best_positions is not None and _time_remaining() > 10:
        energy, relaxed_pos = _relax_with_fallback(best_positions, potential, parameter)
        _update_best(energy, relaxed_pos)
    
    # Fallback: return ideal icosahedron
    if best_positions is None:
        best_positions = _generate_macky_icosahedron(size, scale=base_scale)
    
    return best_positions
# EVOLVE-BLOCK-END

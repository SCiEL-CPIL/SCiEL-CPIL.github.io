"""
Strategy: Basin hopping with vectorized Morse energy/gradient and diverse crystalline starts.
Uses vectorized energy/gradient computation for speed, multiple diverse initializations
(FCC, BCC, HCP, icosahedral, shell, random), and basin hopping to escape local minima.
"""
import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy_and_grad(positions, rho):
    """Compute Morse energy and gradient using vectorized operations."""
    n = len(positions)
    # Pairwise differences: (n, n, 3)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)  # (n, n)
    # Mask diagonal and use only upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    r_upper = r[mask]
    diff_upper = diff[mask]  # (n*(n-1)/2, 3)
    
    if len(r_upper) == 0:
        return 0.0, np.zeros_like(positions)
    
    exp_term = np.exp(rho * (1.0 - r_upper))
    pair_e = exp_term * (exp_term - 2.0)
    energy = np.sum(pair_e)
    
    # dE/dr = -rho * exp_term * (2*exp_term - 2)
    dE_dr = -rho * exp_term * (2.0 * exp_term - 2.0)
    factor = dE_dr / r_upper  # (n*(n-1)/2,)
    
    # Scatter gradient contributions
    grad = np.zeros_like(positions)
    idx_i, idx_j = np.where(mask)
    np.add.at(grad, idx_i, factor[:, np.newaxis] * diff_upper)
    np.add.at(grad, idx_j, -factor[:, np.newaxis] * diff_upper)
    
    return float(energy), grad


def _morse_energy(positions, rho):
    """Compute Morse energy only (vectorized)."""
    e, _ = _morse_energy_and_grad(positions, rho)
    return e


def _is_valid(positions, min_dist=0.5):
    """Check if all pairwise distances are >= min_dist."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return np.min(r[mask]) >= min_dist


def _repair_distances(positions, min_dist=0.5, max_iter=100):
    """Push overlapping atoms apart."""
    positions = positions.copy()
    n = len(positions)
    for _ in range(max_iter):
        if _is_valid(positions, min_dist):
            break
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        r = np.linalg.norm(diff, axis=2)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        for i in range(n):
            for j in range(i + 1, n):
                d = r[i, j]
                if d < min_dist:
                    if d < 1e-10:
                        push = np.array([0.01, 0.01, 0.01])
                    else:
                        push = (positions[i] - positions[j]) / d * (min_dist - d) / 2.0
                    positions[i] += push
                    positions[j] -= push
    return positions


def _generate_bcc(n, spacing=1.0):
    """Generate BCC-like cluster."""
    points = []
    # BCC lattice points
    lattice = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                lattice.append([i * spacing, j * spacing, k * spacing])
                lattice.append([(i + 0.5) * spacing, (j + 0.5) * spacing, (k + 0.5) * spacing])
    
    # Sort by distance from center and take closest n
    lattice.sort(key=lambda p: np.linalg.norm(p))
    points = lattice[:n]
    return np.array(points, dtype=float)


def _generate_fcc(n, spacing=1.0):
    """Generate FCC-like cluster."""
    points = []
    lattice = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                # FCC: corners + face centers
                lattice.append([i * spacing, j * spacing, k * spacing])
                lattice.append([(i + 0.5) * spacing, (j + 0.5) * spacing, k * spacing])
                lattice.append([(i + 0.5) * spacing, j * spacing, (k + 0.5) * spacing])
                lattice.append([i * spacing, (j + 0.5) * spacing, (k + 0.5) * spacing])
    
    # Remove duplicates by rounding
    unique = []
    seen = set()
    for p in lattice:
        key = (round(p[0], 3), round(p[1], 3), round(p[2], 3))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    
    unique.sort(key=lambda p: np.linalg.norm(p))
    points = unique[:n]
    return np.array(points, dtype=float)


def _generate_hcp(n, spacing=1.0):
    """Generate HCP-like cluster."""
    points = []
    a = spacing
    c = spacing * np.sqrt(8.0 / 3.0)
    lattice = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-3, 4):
                # A layer
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3.0) / 2.0 * j
                z = k * c
                lattice.append([x, y, z])
                # B layer (offset)
                x2 = a * (i + 0.5 * ((j + 1) % 2)) + a / 2.0
                y2 = a * np.sqrt(3.0) / 2.0 * j + a * np.sqrt(3.0) / 6.0
                z2 = (k + 0.5) * c
                lattice.append([x2, y2, z2])
    
    unique = []
    seen = set()
    for p in lattice:
        key = (round(p[0], 3), round(p[1], 3), round(p[2], 3))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    
    unique.sort(key=lambda p: np.linalg.norm(p))
    points = unique[:n]
    return np.array(points, dtype=float)


def _generate_icosahedral(n, spacing=1.0):
    """Generate icosahedral-like cluster using nested shells."""
    points = [[0.0, 0.0, 0.0]]
    remaining = n - 1
    
    # Icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = [
        [1, 0, t], [-1, 0, t], [1, 0, -t], [-1, 0, -t],
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [t, 1, 0], [-t, 1, 0], [t, -1, 0], [-t, -1, 0],
        [0, t, 1], [0, -t, 1], [0, t, -1], [0, -t, -1]
    ]
    ico_verts = np.array(ico_verts, dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # First shell: 12 vertices
    shell1 = ico_verts * spacing
    for v in shell1:
        if remaining > 0:
            points.append(list(v))
            remaining -= 1
    
    # Second shell: 30 edge midpoints
    if remaining > 0:
        edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(ico_verts[i] - ico_verts[j])
                if d < 0.6:  # connected vertices
                    edges.append((i, j))
        edge_mids = []
        for i, j in edges:
            mid = (ico_verts[i] + ico_verts[j]) / 2.0
            mid = mid / np.linalg.norm(mid)
            edge_mids.append(mid)
        edge_mids = np.array(edge_mids, dtype=float)
        for v in edge_mids:
            if remaining > 0:
                points.append(list(v * spacing * 1.5))
                remaining -= 1
    
    # Third shell: face centers
    if remaining > 0:
        face_centers = []
        # Use a grid of points on a sphere
        golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
        for i in range(20):
            z = 1.0 - 2.0 * (i + 0.5) / 20
            radius_xy = np.sqrt(max(0, 1.0 - z * z))
            theta = 2.0 * np.pi * i / golden_ratio
            v = np.array([radius_xy * np.cos(theta), radius_xy * np.sin(theta), z])
            face_centers.append(v)
        face_centers = np.array(face_centers, dtype=float)
        for v in face_centers:
            if remaining > 0:
                points.append(list(v * spacing * 2.0))
                remaining -= 1
    
    # Fill remaining with random points on expanding shells
    rng = np.random.default_rng(7)
    while remaining > 0:
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v)
        r = spacing * (2.0 + 0.3 * remaining ** 0.5)
        points.append(list(v * r))
        remaining -= 1
    
    return np.array(points[:n], dtype=float)


def _generate_shell(n, spacing=1.0):
    """Generate spherical shell-like cluster using Fibonacci sphere."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    radius = spacing * (n / (4.0 * np.pi / (spacing ** 2 * 3.0 / 4.0))) ** (1.0 / 3.0) * 0.8
    
    points = []
    for i in range(n):
        z = 1.0 - 2.0 * (i + 0.5) / n
        radius_xy = np.sqrt(max(0, 1.0 - z * z))
        theta = 2.0 * np.pi * i / golden_ratio
        x = radius_xy * np.cos(theta)
        y = radius_xy * np.sin(theta)
        points.append([x * radius, y * radius, z * radius])
    return np.array(points, dtype=float)


def _generate_random_compact(n, spacing=1.0, seed=42):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.75
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            min_dist = min([np.linalg.norm(p - q) for q in points]) if points else np.inf
            if min_dist >= 0.7 * spacing:
                points.append(p)
        attempts += 1
    # If we couldn't place all, fill remaining randomly
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points, dtype=float)


def _local_minimize(positions, rho, maxiter=200):
    """Relax using L-BFGS-B with Morse energy and gradient."""
    def objective_and_grad(x):
        pos = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(pos, rho)
        return e, g.flatten()
    
    x0 = positions.flatten()
    result = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B',
                      options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8})
    return result.fun, result.x.reshape(-1, 3)


def _basin_hopping(positions, rho, n_steps=10, step_scale=0.3, deadline=None, seed=42):
    """Basin hopping: perturb, relax, accept if better or with Metropolis."""
    rng = np.random.default_rng(seed)
    best_pos = positions.copy()
    best_e = _morse_energy(positions, rho)
    current_pos = positions.copy()
    current_e = best_e
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Perturb
        perturbation = rng.normal(0, step_scale, size=current_pos.shape)
        new_pos = current_pos + perturbation
        
        # Relax
        try:
            new_e, new_pos_relaxed = _local_minimize(new_pos, rho, maxiter=100)
        except Exception:
            continue
        
        # Check validity
        if not _is_valid(new_pos_relaxed):
            continue
        
        # Accept if better, or with Metropolis criterion
        if new_e < current_e:
            current_pos = new_pos_relaxed.copy()
            current_e = new_e
            if current_e < best_e:
                best_e = current_e
                best_pos = current_pos.copy()
        else:
            # Metropolis accept (simple: only accept if better for this task)
            pass
    
    return best_e, best_pos





def construct(instance):
    """Construct lowest-energy Morse cluster using basin hopping with diverse crystalline starts."""
    start_time = time.time()
    deadline = start_time + 285.0  # Safety margin before 330s limit
    
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    # Generate diverse starting points
    generators = [
        lambda: _generate_bcc(size),
        lambda: _generate_fcc(size),
        lambda: _generate_hcp(size),
        lambda: _generate_icosahedral(size),
        lambda: _generate_shell(size),
        lambda: _generate_random_compact(size, seed=42),
        lambda: _generate_random_compact(size, seed=123),
        lambda: _generate_bcc(size, spacing=0.95),
        lambda: _generate_bcc(size, spacing=1.05),
    ]
    
    best_energy = np.inf
    best_positions = None
    
    for gen_idx, gen_func in enumerate(generators):
        if time.time() > deadline:
            break
        
        try:
            init_pos = gen_func()
            if len(init_pos) < size:
                continue
            
            # First do a local minimization
            energy, relaxed_pos = _local_minimize(init_pos, rho, maxiter=150)
            
            # Check validity
            if not _is_valid(relaxed_pos):
                relaxed_pos = _repair_distances(relaxed_pos)
                energy, relaxed_pos = _local_minimize(relaxed_pos, rho, maxiter=100)
            
            if _is_valid(relaxed_pos) and energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos.copy()
                
        except Exception:
            continue
    
    # Basin hopping on the best configuration found
    if best_positions is not None and time.time() < deadline:
        try:
            bh_energy, bh_pos = _basin_hopping(
                best_positions, rho,
                n_steps=15,
                step_scale=0.25,
                deadline=deadline,
                seed=42
            )
            if _is_valid(bh_pos) and bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_pos.copy()
        except Exception:
            pass
    
    # If no valid result, fall back to the library's relax
    if best_positions is None:
        init_pos = _generate_random_compact(size, seed=42)
        energy, best_positions = relax(init_pos, potential, rho)
    
    return best_positions

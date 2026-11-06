"""Morse cluster global minima: multi-start with structured initializations and L-BFGS-B."""
import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy_and_grad(pos, rho):
    """Compute Morse energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    x = rho * (1.0 - r)
    exp_x = np.exp(x)
    pair_e = exp_x * (exp_x - 2.0)
    energy = np.sum(np.triu(pair_e, k=1))
    
    # Gradient: dE/dr_ij = 2*rho*exp_x*(exp_x - 1)
    # dE/dpos_i = sum_j (dE/dr_ij) * (r_ij_vec / r_ij)
    grad = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            rij = pos[i] - pos[j]
            rij_norm = np.linalg.norm(rij)
            if rij_norm < 1e-12:
                continue
            x_ij = rho * (1.0 - rij_norm)
            exp_x_ij = np.exp(x_ij)
            dEdr = 2.0 * rho * exp_x_ij * (exp_x_ij - 1.0)
            grad[i] += dEdr * rij / rij_norm
            grad[j] -= dEdr * rij / rij_norm
    return energy, grad


def _generate_fcc(n):
    """Generate FCC-like cluster of n atoms."""
    points = []
    # FCC lattice vectors
    a = 1.0
    lattice = [
        [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1, 0], [1, 0, 1], [0, 1, 1],
        [1, 1, 1], [1.5, 0.5, 0], [1.5, 0, 0.5], [1, 0.5, 0.5],
        [0.5, 1.5, 0], [0, 1.5, 0.5], [0.5, 0, 1.5],
        [2, 0, 0], [0, 2, 0], [0, 0, 2],
        [2, 1, 0], [1, 2, 0], [2, 0, 1], [1, 0, 2], [0, 2, 1], [0, 1, 2],
        [2, 2, 0], [2, 0, 2], [0, 2, 2],
        [2, 2, 1], [2, 1, 2], [1, 2, 2],
        [3, 0, 0], [0, 3, 0], [0, 0, 3],
        [3, 1, 0], [1, 3, 0], [3, 0, 1], [1, 0, 3], [0, 3, 1], [0, 1, 3],
        [3, 2, 0], [2, 3, 0], [3, 0, 2], [2, 0, 3], [0, 3, 2], [0, 2, 3],
        [3, 3, 0], [3, 0, 3], [0, 3, 3],
    ]
    for vec in lattice:
        if len(points) >= n:
            break
        points.append(np.array(vec, dtype=float) * a)
    
    # If we don't have enough, add more
    if len(points) < n:
        rng = np.random.default_rng(42)
        while len(points) < n:
            p = rng.uniform(-0.5, 3.5, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
    
    points = np.array(points[:n])
    # Center
    points -= np.mean(points, axis=0)
    return points


def _generate_icosahedral(n):
    """Generate icosahedral-like cluster."""
    # Icosahedron vertices (normalized to edge length 1)
    phi = (1 + np.sqrt(5)) / 2
    ico_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    # Normalize so edge length is 1
    edge = np.linalg.norm(ico_vertices[0] - ico_vertices[1])
    ico_vertices /= edge
    
    points = [np.zeros(3)]  # center
    if n > 1:
        points.extend(ico_vertices.tolist())
    
    if n > 13:
        # Second shell: octahedral directions
        oct_vertices = np.array([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
        ], dtype=float)
        # Scale to be at distance ~1.5 from center (beyond first shell)
        first_shell_dist = np.linalg.norm(ico_vertices[0])
        second_shell_dist = first_shell_dist * 1.4
        oct_vertices *= second_shell_dist / np.linalg.norm(oct_vertices[0])
        points.extend(oct_vertices.tolist())
    
    if n > 19:
        # Third shell: more points on sphere
        rng = np.random.default_rng(123)
        third_shell_dist = first_shell_dist * 1.8
        while len(points) < n:
            p = rng.normal(0, 1, 3)
            p = p / np.linalg.norm(p) * third_shell_dist
            if all(np.linalg.norm(p - np.array(q)) >= 0.5 for q in points):
                points.append(p)
    
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points


def _generate_random_compact(n, seed=0):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.55 for q in points):
            points.append(p)
    
    # If we couldn't pack enough, force add with jitter
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points


def _generate_fibonacci_sphere(n):
    """Generate points on a Fibonacci sphere, then scale inward."""
    points = []
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        z = 1 - 2 * i / (n - 1) if n > 1 else 0
        radius_z = np.sqrt(1 - z * z)
        theta = golden_angle * i
        x = np.cos(theta) * radius_z
        y = np.sin(theta) * radius_z
        points.append([x, y, z])
    
    points = np.array(points, dtype=float)
    # Scale so that typical inter-particle distance is ~1
    if n > 1:
        # Average nearest neighbor distance on unit sphere is ~2*pi/sqrt(n)
        scale = 1.0 / (2 * np.pi / np.sqrt(n)) * 0.8
        points *= scale
    points -= np.mean(points, axis=0)
    return points


def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 6))
    
    start_time = time.time()
    deadline = start_time + 280  # 50s safety margin
    
    # Generate diverse starting configurations
    generators = [
        lambda: _generate_fcc(size),
        lambda: _generate_icosahedral(size),
        lambda: _generate_random_compact(size, seed=size),
        lambda: _generate_fibonacci_sphere(size),
        lambda: _generate_random_compact(size, seed=size + 100),
        lambda: _generate_random_compact(size, seed=size + 200),
    ]
    
    best_energy = np.inf
    best_positions = None
    
    # First pass: quick relax on all generators
    for gen_func in generators:
        if time.time() > deadline - 30:
            break
        try:
            init_pos = gen_func()
            if len(init_pos) < size:
                continue
            energy, positions = relax(init_pos, potential, rho)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue
    
    # Second pass: L-BFGS-B on the best few starting points
    # Use exact energy and gradient for faster convergence
    if best_positions is not None and time.time() < deadline - 20:
        try:
            res = minimize(
                _morse_energy_and_grad,
                best_positions,
                args=(rho,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'gtol': 1e-8}
            )
            if res.success or res.fun < best_energy:
                best_energy = res.fun
                best_positions = res.x.copy()
        except Exception:
            pass
    
    # Third pass: try L-BFGS-B on other good starting points if time permits
    if time.time() < deadline - 10:
        for gen_func in generators[:3]:  # Just the structured ones
            if time.time() > deadline - 5:
                break
            try:
                init_pos = gen_func()
                res = minimize(
                    _morse_energy_and_grad,
                    init_pos,
                    args=(rho,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 300, 'gtol': 1e-8}
                )
                if res.fun < best_energy:
                    best_energy = res.fun
                    best_positions = res.x.copy()
            except Exception:
                continue
    
    # Final safety check: ensure minimum distance constraint
    if best_positions is not None:
        # Verify distances
        for i in range(size):
            for j in range(i + 1, size):
                if np.linalg.norm(best_positions[i] - best_positions[j]) < 0.5:
                    # Apply small perturbation
                    best_positions[j] += 0.5 * (best_positions[j] - best_positions[i]) / np.linalg.norm(best_positions[j] - best_positions[i])
    
    if best_positions is None:
        # Fallback
        best_positions = _generate_random_compact(size, seed=999)
    
    return best_positions

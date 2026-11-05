"""Dzugutov cluster global minima search using polytetrahedral/icosahedral initial guesses."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient


# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16.0


def _dzugutov_energy_and_grad(positions):
    """Compute Dzugutov potential energy and gradient efficiently."""
    n = len(positions)
    energy = 0.0
    grad = np.zeros_like(positions)
    
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[i] - positions[j]
            r = np.linalg.norm(rij)
            if r < 1e-12:
                continue
            
            dV_dr = 0.0
            if r < a:
                r_inv_m = r ** (-m)
                r_inv_m1 = r ** (-(m + 1))
                exp_term = np.exp(c / (r - a))
                V = A * (r_inv_m - B) * exp_term
                dV_dr = A * (-m * r_inv_m1 - B * r_inv_m1) * exp_term + \
                        A * (r_inv_m - B) * exp_term * (-c / (r - a) ** 2)
            elif r < b:
                exp_term = np.exp(d / (r - b))
                V = B * exp_term
                dV_dr = B * exp_term * (-d / (r - b) ** 2)
            else:
                continue
            
            energy += V
            # Force on atom i is -dV/dr * rij/r
            force = -dV_dr * rij / r
            grad[i] += force
            grad[j] -= force
    
    return energy, grad


def _refine(positions, time_budget=15.0):
    """Refine positions using L-BFGS-B with the Dzugutov potential."""
    start = time.time()
    
    def objective(x):
        if time.time() - start > time_budget:
            raise TimeoutError()
        pos = x.reshape(-1, 3)
        # Check minimum distance constraint
        n = len(pos)
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(pos[i] - pos[j]) < 0.5:
                    return 1e10  # Penalty for violating constraint
        e, _ = _dzugutov_energy_and_grad(pos)
        return e
    
    def gradient(x):
        if time.time() - start > time_budget:
            raise TimeoutError()
        pos = x.reshape(-1, 3)
        # Check minimum distance constraint
        n = len(pos)
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(pos[i] - pos[j]) < 0.5:
                    return np.ones_like(x) * 1e8
        _, g = _dzugutov_energy_and_grad(pos)
        return g.flatten()
    
    try:
        result = minimize(
            objective,
            positions.flatten(),
            jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if result.success or result.fun < 1e9:
            return result.fun, result.x.reshape(-1, 3)
        else:
            return result.fun, result.x.reshape(-1, 3)
    except TimeoutError:
        return float('inf'), positions
    except Exception:
        return float('inf'), positions


def _icosahedron_vertices(radius=1.0):
    """Generate 12 vertices of a regular icosahedron with given radius."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms * radius
    return verts


def _build_mackay_icosahedron(n, scale=1.1):
    """Build a Mackay icosahedral cluster with n atoms."""
    if n <= 12:
        return _icosahedron_vertices(scale)[:n].copy()
    
    # 1st shell: 12 vertices of icosahedron
    positions = _icosahedron_vertices(scale).tolist()
    positions.append([0.0, 0.0, 0.0])  # Central atom (13 total)
    
    if n <= 13:
        return np.array(positions[:n])
    
    # 2nd shell: 30 atoms at edge midpoints, scaled outward
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts_unit = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    norms = np.linalg.norm(verts_unit, axis=1, keepdims=True)
    verts_unit = verts_unit / norms
    
    # Compute edge midpoints (30 edges of icosahedron)
    edges = [
        (0,1),(0,2),(0,4),(0,5),(0,8),(0,11),
        (1,3),(1,4),(1,5),(1,9),(1,10),
        (2,3),(2,6),(2,7),(2,8),(2,11),
        (3,6),(3,7),(3,9),(3,10),
        (4,5),(4,6),(4,7),(4,9),(4,11),
        (5,6),(5,7),(5,8),(5,9),
        (6,8),(6,10),
        (7,8),(7,10),
        (8,11),(9,11)
    ]
    
    second_shell = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        mid = mid / np.linalg.norm(mid) * scale * 1.55
        second_shell.append(mid)
    
    # Add second shell atoms
    for pos in second_shell:
        if len(positions) >= n:
            break
        positions.append(pos)
    
    # If we need more than 43 atoms, add third shell
    if n > 43:
        # 3rd shell: 20 atoms at face centers, scaled further out
        faces = [
            (0,1,4),(0,1,5),(0,2,4),(0,2,11),(0,4,5),(0,5,11),
            (1,3,4),(1,3,5),(1,4,9),(1,5,9),(1,9,10),
            (2,3,6),(2,3,7),(2,6,8),(2,7,8),(2,8,11),
            (3,6,7),(3,7,10),(3,9,10),
            (4,5,6),(4,5,7),(4,6,9),(4,7,11),
            (5,6,11),(5,7,11),
            (6,8,10),(6,10,11),
            (7,8,10),(7,10,11)
        ]
        
        third_shell = []
        for i, j, k in faces:
            center = (verts_unit[i] + verts_unit[j] + verts_unit[k]) / 3.0
            center = center / np.linalg.norm(center) * scale * 2.0
            third_shell.append(center)
        
        for pos in third_shell:
            if len(positions) >= n:
                break
            positions.append(pos)
    
    # If still not enough, add atoms on a spherical shell
    while len(positions) < n:
        r = scale * 2.3
        theta = np.random.uniform(0, 2 * np.pi)
        cos_phi = np.random.uniform(-1, 1)
        sin_phi = np.sqrt(1 - cos_phi ** 2)
        p = np.array([r * sin_phi * np.cos(theta), r * sin_phi * np.sin(theta), r * cos_phi])
        positions.append(p)
    
    return np.array(positions[:n])


def _build_polytetrahedral_cluster(n, rng, scale=1.1):
    """
    Build a cluster by growing an icosahedral core and adding atoms
    in positions that favor polytetrahedral coordination.
    """
    if n <= 12:
        verts = _icosahedron_vertices(scale)
        return verts[:n].copy()
    
    # Start with icosahedral core (12 atoms) + center
    positions = _icosahedron_vertices(scale).tolist()
    positions.append([0.0, 0.0, 0.0])
    
    if n <= 13:
        return np.array(positions[:n])
    
    # Generate candidate positions for additional atoms
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts_unit = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    norms = np.linalg.norm(verts_unit, axis=1, keepdims=True)
    verts_unit = verts_unit / norms
    
    # Face centers of icosahedron (20 faces)
    faces = [
        (0,1,4),(0,1,5),(0,2,4),(0,2,11),(0,4,5),(0,5,11),
        (1,3,4),(1,3,5),(1,4,9),(1,5,9),(1,9,10),
        (2,3,6),(2,3,7),(2,6,8),(2,7,8),(2,8,11),
        (3,6,7),(3,7,10),(3,9,10),
        (4,5,6),(4,5,7),(4,6,9),(4,7,11),
        (5,6,11),(5,7,11),
        (6,8,10),(6,10,11),
        (7,8,10),(7,10,11)
    ]
    
    # Edge midpoints (30 edges)
    edges = [
        (0,1),(0,2),(0,4),(0,5),(0,8),(0,11),
        (1,3),(1,4),(1,5),(1,9),(1,10),
        (2,3),(2,6),(2,7),(2,8),(2,11),
        (3,6),(3,7),(3,9),(3,10),
        (4,5),(4,6),(4,7),(4,9),(4,11),
        (5,6),(5,7),(5,8),(5,9),
        (6,8),(6,10),
        (7,8),(7,10),
        (8,11),(9,11)
    ]
    
    # Generate candidate positions
    candidates = []
    
    # Face centers at various radii
    for r_factor in [1.55, 1.7, 1.85, 2.0, 2.15]:
        for i, j, k in faces:
            center = (verts_unit[i] + verts_unit[j] + verts_unit[k]) / 3.0
            center = center / np.linalg.norm(center) * scale * r_factor
            candidates.append(center)
    
    # Edge midpoints at various radii
    for r_factor in [1.55, 1.7, 1.85, 2.0, 2.15]:
        for i, j in edges:
            mid = (verts_unit[i] + verts_unit[j]) / 2.0
            mid = mid / np.linalg.norm(mid) * scale * r_factor
            candidates.append(mid)
    
    # Random positions on shells
    for r_factor in [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4]:
        for _ in range(20):
            r = scale * r_factor
            theta = rng.uniform(0, 2 * np.pi)
            cos_phi = rng.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi ** 2)
            p = np.array([r * sin_phi * np.cos(theta), r * sin_phi * np.sin(theta), r * cos_phi])
            candidates.append(p)
    
    # Select candidates that are well-separated from existing atoms
    min_dist = 0.55
    selected = []
    
    # Sort candidates by distance from origin (prefer inner positions)
    candidates = sorted(candidates, key=lambda p: np.linalg.norm(p))
    
    for cand in candidates:
        if len(positions) + len(selected) >= n:
            break
        if all(np.linalg.norm(np.array(pos) - cand) >= min_dist for pos in positions + selected):
            selected.append(cand)
    
    positions.extend(selected)
    
    # If we still don't have enough atoms, add more randomly
    while len(positions) < n:
        r = scale * (1.5 + 0.5 * rng.random())
        theta = rng.uniform(0, 2 * np.pi)
        cos_phi = rng.uniform(-1, 1)
        sin_phi = np.sqrt(1 - cos_phi ** 2)
        p = np.array([r * sin_phi * np.cos(theta), r * sin_phi * np.sin(theta), r * cos_phi])
        positions.append(p)
    
    return np.array(positions[:n])


def _build_random_compact(n, rng, scale=1.1):
    """Generate a random compact cluster."""
    positions = []
    radius = scale * n ** (1.0 / 3.0) * 0.9
    min_dist = 0.55
    
    attempts = 0
    max_attempts = 30000
    
    while len(positions) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - np.array(pos)) >= min_dist for pos in positions):
                positions.append(p)
    
    # If we couldn't place all atoms, add remaining randomly
    while len(positions) < n:
        p = rng.uniform(-radius, radius, 3)
        positions.append(p)
    
    return np.array(positions)


def _ensure_min_distance(positions, min_dist=0.5, max_iters=100):
    """Ensure all pairs are at least min_dist apart."""
    positions = positions.copy()
    n = len(positions)
    
    for _ in range(max_iters):
        violated = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist:
                    violated = True
                    if d < 1e-6:
                        positions[i] += 0.1 * np.random.standard_normal(3)
                    else:
                        direction = (positions[i] - positions[j]) / d
                        push = (min_dist - d) / 2 + 0.01
                        positions[i] += push * direction
                        positions[j] -= push * direction
        if not violated:
            break
    
    return positions


def construct(instance):
    """
    Find low-energy structures for Dzugutov clusters using multiple
    polytetrahedral/icosahedral initial guesses and local relaxation.
    """
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    
    rng = np.random.default_rng(n * 7919)
    
    best_energy = float('inf')
    best_positions = None
    
    # Define initialization strategies
    def init_mackay():
        return _build_mackay_icosahedron(n, scale=1.1)
    
    def init_polytetrahedral():
        return _build_polytetrahedral_cluster(n, rng, scale=1.1)
    
    def init_random():
        return _build_random_compact(n, rng, scale=1.1)
    
    def init_mackay_perturbed():
        pos = _build_mackay_icosahedron(n, scale=1.1)
        return pos + 0.05 * rng.standard_normal(pos.shape)
    
    def init_polytetrahedral_perturbed():
        pos = _build_polytetrahedral_cluster(n, rng, scale=1.1)
        return pos + 0.05 * rng.standard_normal(pos.shape)
    
    strategies = [init_mackay, init_polytetrahedral, init_random, 
                  init_mackay_perturbed, init_polytetrahedral_perturbed]
    
    # First pass: try each strategy once
    for i, strategy in enumerate(strategies):
        if time.time() > deadline - 30:
            break
        
        positions = strategy()
        positions = _ensure_min_distance(positions)
        
        # Try both relax and L-BFGS-B refinement
        # First try the built-in relax
        try:
            energy, relaxed = relax(positions, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed.copy()
        except Exception:
            pass
        
        # Then try L-BFGS-B refinement
        if time.time() < deadline - 25:
            e2, pos2 = _refine(positions, time_budget=10.0)
            if e2 < best_energy:
                best_energy = e2
                best_positions = pos2.copy()
    
    # Second pass: if we have a good result, try perturbed versions
    if best_positions is not None and time.time() < deadline - 40:
        for _ in range(3):
            if time.time() > deadline - 25:
                break
            
            noise_scale = 0.1
            positions = best_positions + noise_scale * rng.standard_normal(best_positions.shape)
            positions = _ensure_min_distance(positions)
            
            # Try relax
            try:
                energy, relaxed = relax(positions, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed.copy()
            except Exception:
                pass
            
            # Try L-BFGS-B
            if time.time() < deadline - 20:
                e2, pos2 = _refine(positions, time_budget=8.0)
                if e2 < best_energy:
                    best_energy = e2
                    best_positions = pos2.copy()
    
    # Final pass: try more diverse initializations if time permits
    if time.time() < deadline - 35:
        for i in range(3):
            if time.time() > deadline - 20:
                break
            
            # Use different scales
            scale = 1.0 + 0.1 * i
            positions = _build_mackay_icosahedron(n, scale=scale)
            positions = _ensure_min_distance(positions)
            
            try:
                energy, relaxed = relax(positions, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed.copy()
            except Exception:
                pass
            
            if time.time() < deadline - 15:
                e2, pos2 = _refine(positions, time_budget=8.0)
                if e2 < best_energy:
                    best_energy = e2
                    best_positions = pos2.copy()
    
    if best_positions is None:
        # Fallback: return a simple compact arrangement
        positions = _build_random_compact(n, rng, scale=1.1)
        positions = _ensure_min_distance(positions)
        try:
            energy, relaxed = relax(positions, potential, parameter)
            best_positions = relaxed
        except Exception:
            best_positions = positions
    
    return best_positions
# EVOLVE-BLOCK-END

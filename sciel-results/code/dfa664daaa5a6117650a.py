"""Dzugutov cluster global minima search using topological optimization with polytetrahedral constraints."""
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax
from scipy.optimize import minimize

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c_true = 1.10
d_true = 0.27
m = 16.0

def _find_r_min():
    """Find the minimum energy distance r_min for the Dzugutov potential."""
    r = np.linspace(0.5, 2.0, 1000)
    energies = []
    for ri in r:
        if ri < a:
            e = A * (ri ** (-m) - B) * np.exp(c_true / (ri - a))
        elif ri < b:
            e = B * np.exp(d_true / (ri - b))
        else:
            e = 0.0
        energies.append(e)
    energies = np.array(energies)
    idx = np.argmin(energies)
    return r[idx]

R_MIN = _find_r_min()

def _icosahedron_vertices(radius=1.0):
    """Generate 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis] * radius
    return verts

def _build_mackay_icosahedron(n, rng, r_min=R_MIN):
    """
    Build a Mackay icosahedral structure.
    For n=13: center + 12 vertices
    For n=55: 1 + 12 + 42 (Mackay icosahedron)
    For other n: start with 13, add atoms on second shell
    """
    points = [np.zeros(3)]
    
    # Shell 1: 12 atoms at icosahedron vertices
    ico = _icosahedron_vertices(r_min)
    points.extend(list(ico))
    
    if n <= 13:
        if n < 13:
            idx = rng.choice(13, n, replace=False)
            return np.array([points[i] for i in idx])
        return np.array(points)
    
    # Shell 2: Add atoms
    # For Mackay 55, the second shell has 42 atoms at specific positions
    # For general n, we'll place atoms on a sphere of radius ~1.7*r_min
    r2 = 1.7 * r_min
    needed = n - 13
    added = 0
    attempts = 0
    min_dist = 0.5
    
    # For Mackay 55, try to place atoms in a more structured way
    if n == 55:
        # Place 42 atoms on second shell
        # Use a quasi-random distribution
        for _ in range(needed):
            u = rng.random()
            v = rng.random()
            theta = 2 * np.pi * u
            phi = np.arccos(2 * v - 1)
            
            x = r2 * np.sin(phi) * np.cos(theta)
            y = r2 * np.sin(phi) * np.sin(theta)
            z = r2 * np.cos(phi)
            p = np.array([x, y, z])
            
            # Check minimum distance
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            
            if valid:
                points.append(p)
                added += 1
            else:
                # Try again with small offset
                p = p + 0.05 * rng.standard_normal(3)
                points.append(p)
                added += 1
    else:
        while added < needed and attempts < 5000:
            u = rng.random()
            v = rng.random()
            theta = 2 * np.pi * u
            phi = np.arccos(2 * v - 1)
            
            x = r2 * np.sin(phi) * np.cos(theta)
            y = r2 * np.sin(phi) * np.sin(theta)
            z = r2 * np.cos(phi)
            p = np.array([x, y, z])
            
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            
            if valid:
                points.append(p)
                added += 1
            attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-r2, r2, 3)
        if np.linalg.norm(p) > 1e-6:
            p = p / np.linalg.norm(p) * r2
        points.append(p)
        
    return np.array(points)

def _build_random_compact(n, rng, r_min=R_MIN):
    """Build a random compact cluster."""
    radius = r_min * (n / 13) ** (1/3) * 1.2
    points = []
    attempts = 0
    min_dist = 0.5
    
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
        
    while len(points) < n:
        base = points[len(points) % max(1, len(points))] if points else np.zeros(3)
        p = base + 0.2 * rng.standard_normal(3)
        points.append(p)
        
    return np.array(points)

def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 1e-12:
                continue
            if r < a:
                energy += A * (r ** (-m) - B) * np.exp(c_true / (r - a))
            elif r < b:
                energy += B * np.exp(d_true / (r - b))
    return energy

def _dzugutov_energy_flat(x):
    """Flat energy function for scipy.optimize."""
    n = len(x) // 3
    positions = x.reshape(n, 3)
    return _dzugutov_energy(positions)

def _dzugutov_gradient_flat(x):
    """Flat gradient function for scipy.optimize."""
    n = len(x) // 3
    positions = x.reshape(n, 3)
    grad = np.zeros_like(x)
    
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[i] - positions[j]
            r = np.linalg.norm(rij)
            if r < 1e-12:
                continue
            if r < a:
                exp_term = np.exp(c_true / (r - a))
                dVdr = A * (-m * r ** (-m - 1) * exp_term + (r ** (-m) - B) * exp_term * (-c_true / (r - a) ** 2))
            elif r < b:
                exp_term = np.exp(d_true / (r - b))
                dVdr = B * exp_term * (-d_true / (r - b) ** 2)
            else:
                continue
            grad[i] += dVdr * rij / r
            grad[j] -= dVdr * rij / r
    
    return grad

def _topological_energy(positions, bond_pairs, r_target=R_MIN, barrier_k=100.0):
    """
    Compute energy with topological constraints.
    bond_pairs: list of (i, j) pairs that should be bonded
    barrier_k: barrier strength for bond length deviations
    """
    n = len(positions)
    
    # Standard Dzugutov energy
    energy = _dzugutov_energy(positions)
    
    # Add barrier for bond length deviations
    if bond_pairs is not None and len(bond_pairs) > 0:
        barrier = 0.0
        for i, j in bond_pairs:
            r = np.linalg.norm(positions[i] - positions[j])
            barrier += barrier_k * (r - r_target) ** 2
        energy += barrier
    
    return energy

def _topological_energy_flat(x, bond_pairs, r_target=R_MIN, barrier_k=100.0):
    """Flat topological energy for scipy.optimize."""
    n = len(x) // 3
    positions = x.reshape(n, 3)
    return _topological_energy(positions, bond_pairs, r_target, barrier_k)

def _topological_gradient_flat(x, bond_pairs, r_target=R_MIN, barrier_k=100.0):
    """Flat topological gradient for scipy.optimize."""
    n = len(x) // 3
    positions = x.reshape(n, 3)
    grad = np.zeros_like(x)
    
    # Dzugutov gradient
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[i] - positions[j]
            r = np.linalg.norm(rij)
            if r < 1e-12:
                continue
            if r < a:
                exp_term = np.exp(c_true / (r - a))
                dVdr = A * (-m * r ** (-m - 1) * exp_term + (r ** (-m) - B) * exp_term * (-c_true / (r - a) ** 2))
            elif r < b:
                exp_term = np.exp(d_true / (r - b))
                dVdr = B * exp_term * (-d_true / (r - b) ** 2)
            else:
                continue
            grad[i] += dVdr * rij / r
            grad[j] -= dVdr * rij / r
    
    # Barrier gradient
    if bond_pairs is not None and len(bond_pairs) > 0:
        for i, j in bond_pairs:
            rij = positions[i] - positions[j]
            r = np.linalg.norm(rij)
            if r < 1e-12:
                continue
            # d/dr [k * (r - r_target)^2] = 2k * (r - r_target)
            d_barrier_dr = 2 * barrier_k * (r - r_target)
            grad[i] += d_barrier_dr * rij / r
            grad[j] -= d_barrier_dr * rij / r
    
    return grad

def _find_bonds(positions, r_min=R_MIN, tol=0.3):
    """Find bonds: pairs of atoms within r_min + tol."""
    n = len(positions)
    bonds = []
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < r_min + tol:
                bonds.append((i, j))
    return bonds

def _topological_optimize(positions, bond_pairs, n, start_time, time_limit=285.0, max_iter=100):
    """
    Optimize positions with topological constraints using L-BFGS-B.
    """
    current_pos = positions.copy()
    x0 = current_pos.flatten()
    
    try:
        result = minimize(
            _topological_energy_flat,
            x0,
            jac=_topological_gradient_flat,
            args=(bond_pairs, R_MIN, 100.0),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        if result.success or result.fun < _topological_energy(current_pos, bond_pairs, R_MIN, 100.0):
            current_pos = result.x.reshape(n, 3)
    except Exception:
        pass
    
    return current_pos

def _annealed_energy(positions, c_param, d_param):
    """Compute energy with annealed softness parameters."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 1e-12:
                continue
            if r < a:
                energy += A * (r ** (-m) - B) * np.exp(c_param / (r - a))
            elif r < b:
                energy += B * np.exp(d_param / (r - b))
    return energy

def _annealed_energy_flat(x, c_param, d_param):
    """Flat energy function for scipy.optimize."""
    n = len(x) // 3
    positions = x.reshape(n, 3)
    return _annealed_energy(positions, c_param, d_param)

def _annealed_gradient_flat(x, c_param, d_param):
    """Flat gradient function for scipy.optimize."""
    n = len(x) // 3
    positions = x.reshape(n, 3)
    grad = np.zeros_like(x)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[i] - positions[j]
            r = np.linalg.norm(rij)
            if r < 1e-12:
                continue
            if r < a:
                exp_term = np.exp(c_param / (r - a))
                dVdr = A * (-m * r ** (-m - 1) * exp_term + (r ** (-m) - B) * exp_term * (-c_param / (r - a) ** 2))
            elif r < b:
                exp_term = np.exp(d_param / (r - b))
                dVdr = B * exp_term * (-d_param / (r - b) ** 2)
            else:
                continue
            grad[i] += dVdr * rij / r
            grad[j] -= dVdr * rij / r
    return grad

def _continuous_annealing(positions, n, start_time, time_limit=285.0):
    """
    Perform continuous annealing on Dzugutov softness parameters.
    """
    current_pos = positions.copy()
    
    # Annealing schedule
    c_start = 0.1
    d_start = 0.02
    c_end = c_true
    d_end = d_true
    n_steps = 20
    
    for step in range(n_steps):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        t = step / max(1, n_steps - 1)
        c_current = c_start + t * (c_end - c_start)
        d_current = d_start + t * (d_end - d_start)
        
        try:
            result = minimize(
                _annealed_energy_flat,
                current_pos.flatten(),
                jac=_annealed_gradient_flat,
                args=(c_current, d_current),
                method='L-BFGS-B',
                options={'maxiter': 30, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if result.success or result.fun < _annealed_energy(current_pos, c_current, d_current):
                current_pos = result.x.reshape(n, 3)
        except Exception:
            pass
    
    return current_pos

def construct(instance):
    """Find low-energy Dzugutov clusters using topological optimization."""
    start_time = time.time()
    time_limit = 300.0
    safety_margin = 15.0
    n = instance["n"]
    potential = instance["potential"]
    param = None
    
    rng = np.random.default_rng(n + 42)
    
    best_energy = float('inf')
    best_positions = None
    
    def check_time():
        return (time.time() - start_time) < (time_limit - safety_margin)
    
    def try_relax(pos, label=""):
        nonlocal best_energy, best_positions
        try:
            # Add small noise to break symmetry
            pos_noisy = pos + 0.01 * rng.standard_normal((n, 3))
            energy, relaxed_pos = relax(pos_noisy, potential, param)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos.copy()
                return True
        except Exception:
            pass
        return False
    
    # Phase 1: Generate diverse initial guesses
    initial_guesses = []
    
    # Mackay icosahedral guesses
    for i in range(3):
        if not check_time():
            break
        pos = _build_mackay_icosahedron(n, rng)
        initial_guesses.append(pos)
    
    # Random compact guesses
    for i in range(3):
        if not check_time():
            break
        pos = _build_random_compact(n, rng)
        initial_guesses.append(pos)
    
    # Phase 2: Topological optimization on each initial guess
    for i, pos_guess in enumerate(initial_guesses):
        if not check_time():
            break
        
        # Find bonds from initial guess
        bonds = _find_bonds(pos_guess, R_MIN, tol=0.3)
        
        # Topological optimization
        pos_topo = _topological_optimize(pos_guess, bonds, n, start_time, time_limit, max_iter=50)
        
        # Continue with annealing
        pos_annealed = _continuous_annealing(pos_topo, n, start_time, time_limit)
        
        # Final relaxation
        try_relax(pos_annealed, f"topo_{i}")
    
    # Phase 3: Basin hopping on best solution
    if best_positions is not None:
        kicks = 10
        for i in range(kicks):
            if not check_time():
                break
            kick_size = 0.15 + 0.03 * i
            pos_kicked = best_positions + kick_size * rng.standard_normal((n, 3))
            try_relax(pos_kicked, f"kick_{i}")
    
    # Phase 4: Additional diverse restarts
    extra_attempts = 8
    for i in range(extra_attempts):
        if not check_time():
            break
        
        if i % 2 == 0:
            pos = _build_mackay_icosahedron(n, rng)
        else:
            pos = _build_random_compact(n, rng)
        
        # Quick topological + annealing
        bonds = _find_bonds(pos, R_MIN, tol=0.3)
        pos_topo = _topological_optimize(pos, bonds, n, start_time, time_limit, max_iter=30)
        pos_annealed = _continuous_annealing(pos_topo, n, start_time, time_limit)
        try_relax(pos_annealed, f"extra_{i}")
    
    # Fallback
    if best_positions is None:
        best_positions = _build_random_compact(n, rng)
    
    return best_positions

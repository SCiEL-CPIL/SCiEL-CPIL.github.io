"""Dzugutov cluster global minima search using continuous annealing on Dzugutov softness parameters."""
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
                # V(r) = A * (r^(-m) - B) * exp(c/(r-a))
                # dV/dr = A * (-m * r^(-m-1) * exp(c/(r-a)) + (r^(-m) - B) * exp(c/(r-a)) * (-c/(r-a)^2))
                exp_term = np.exp(c_param / (r - a))
                dVdr = A * (-m * r ** (-m - 1) * exp_term + (r ** (-m) - B) * exp_term * (-c_param / (r - a) ** 2))
            elif r < b:
                # V(r) = B * exp(d/(r-b))
                # dV/dr = B * exp(d/(r-b)) * (-d/(r-b)^2)
                exp_term = np.exp(d_param / (r - b))
                dVdr = B * exp_term * (-d_param / (r - b) ** 2)
            else:
                continue
            grad[i] += dVdr * rij / r
            grad[j] -= dVdr * rij / r
    return grad

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

def _build_icosahedral_core(n, rng, r_min=R_MIN):
    """
    Build an initial guess based on icosahedral symmetry.
    1. Place 1 atom at center.
    2. Place 12 atoms at icosahedron vertices (radius r_min).
    3. If n > 13, add atoms on a second shell (radius ~ 1.7 * r_min) 
       using a random rejection method to keep them separated.
    """
    points = [np.zeros(3)]
    
    # Shell 1
    ico = _icosahedron_vertices(r_min)
    points.extend(list(ico))
    
    if n <= 13:
        if n < 13:
             idx = rng.choice(13, n, replace=False)
             return np.array([points[i] for i in idx])
        return np.array(points)
    
    # Shell 2
    r2 = 1.7 * r_min
    needed = n - 13
    added = 0
    attempts = 0
    min_dist = 0.5
    
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

def _continuous_annealing_trajectory(positions, n, rng, start_time, time_limit=290.0):
    """
    Perform continuous annealing on the Dzugutov potential's softness parameters.
    Start with reduced c and d (smoother landscape), gradually increase to true values.
    Use gradient-based optimization at each annealing step.
    """
    current_pos = positions.copy()
    n_atoms = len(current_pos)
    
    # Annealing schedule: start with very smooth (small c, d), end with true values
    # c_true = 1.10, d_true = 0.27
    # Start with c_start = 0.1, d_start = 0.02 (much smoother)
    c_start = 0.1
    d_start = 0.02
    c_end = c_true
    d_end = d_true
    
    # Number of annealing steps
    n_steps = 30
    step_times = np.linspace(0, 1, n_steps)
    
    for step in range(n_steps):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        t = step_times[step]
        # Linear interpolation of c and d
        c_current = c_start + t * (c_end - c_start)
        d_current = d_start + t * (d_end - d_start)
        
        # Perform a few gradient steps at this annealing level
        # Use L-BFGS-B for robustness
        try:
            result = minimize(
                _annealed_energy_flat,
                current_pos.flatten(),
                jac=_annealed_gradient_flat,
                args=(c_current, d_current),
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            if result.success or result.fun < _annealed_energy(current_pos, c_current, d_current):
                current_pos = result.x.reshape(n_atoms, 3)
        except Exception:
            pass
    
    return current_pos

def construct(instance):
    """Find low-energy Dzugutov clusters using continuous annealing."""
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
            # Add small noise to break symmetry if needed
            pos_noisy = pos + 0.01 * rng.standard_normal((n, 3))
            energy, relaxed_pos = relax(pos_noisy, potential, param)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos.copy()
                return True
        except Exception as e:
            pass
        return False

    # Phase 1: Generate diverse initial guesses
    initial_guesses = []
    
    # Icosahedral guesses
    ico_attempts = 5
    if n <= 13:
        ico_attempts = 3
    
    for i in range(ico_attempts):
        if not check_time(): break
        pos = _build_icosahedral_core(n, rng)
        initial_guesses.append(pos)
        
    # Random compact guesses
    rand_attempts = 5
    for i in range(rand_attempts):
        if not check_time(): break
        pos = _build_random_compact(n, rng)
        initial_guesses.append(pos)
    
    # Phase 2: Apply continuous annealing to each initial guess
    annealed_positions = []
    for i, pos in enumerate(initial_guesses):
        if not check_time(): break
        try:
            annealed = _continuous_annealing_trajectory(pos, n, rng, start_time)
            annealed_positions.append(annealed)
        except Exception:
            annealed_positions.append(pos)
    
    # Phase 3: Relax all annealed positions with the true Dzugutov potential
    for i, pos in enumerate(annealed_positions):
        if not check_time(): break
        try_relax(pos, f"annealed_{i}")
    
    # Phase 4: Basin hopping on best solution
    kicks = 10
    if best_positions is not None:
        for i in range(kicks):
            if not check_time(): break
            kick_size = 0.15 + 0.04 * i
            pos_kicked = best_positions + kick_size * rng.standard_normal((n, 3))
            
            # Apply a short annealing trajectory to the kicked position
            try:
                annealed_kicked = _continuous_annealing_trajectory(
                    pos_kicked, n, rng, start_time, time_limit=280.0
                )
                try_relax(annealed_kicked, f"kick_annealed_{i}")
            except Exception:
                try_relax(pos_kicked, f"kick_{i}")
    
    # Phase 5: Additional diverse attempts if time permits
    extra_attempts = 5
    for i in range(extra_attempts):
        if not check_time(): break
        if i % 2 == 0:
            pos = _build_icosahedral_core(n, rng)
        else:
            pos = _build_random_compact(n, rng)
        
        # Apply annealing
        try:
            annealed = _continuous_annealing_trajectory(pos, n, rng, start_time, time_limit=280.0)
            try_relax(annealed, f"extra_annealed_{i}")
        except Exception:
            try_relax(pos, f"extra_{i}")
    
    # Fallback
    if best_positions is None:
        best_positions = _build_random_compact(n, rng)
        
    return best_positions

# EVOLVE-BLOCK-END

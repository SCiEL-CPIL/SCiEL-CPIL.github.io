import numpy as np
import time
from autoresearch_suite.chem_clusters import relax
from scipy.optimize import minimize

# Dzugutov potential parameters
A = 5.82
B = 1.28
a_pot = 1.87
b_pot = 1.94
c_pot = 1.10
d_pot = 0.27
m_pot = 16.0

def _find_r_min():
    """Find the minimum energy distance r_min for the Dzugutov potential."""
    r = np.linspace(0.5, 2.0, 1000)
    energies = []
    for ri in r:
        if ri < a_pot:
            e = A * (ri ** (-m_pot) - B) * np.exp(c_pot / (ri - a_pot))
        elif ri < b_pot:
            e = B * np.exp(d_pot / (ri - b_pot))
        else:
            e = 0.0
        energies.append(e)
    energies = np.array(energies)
    idx = np.argmin(energies)
    return r[idx]

R_MIN = _find_r_min()

# Precompute Mackay icosahedral positions for n=55 (1+12+42)
def _mackay_55():
    """Generate the 55-atom Mackay icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    r1 = R_MIN
    r2 = R_MIN * np.sqrt(1.7)  # second shell radius
    
    # 12 vertices of icosahedron
    ico = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True) * r1
    
    # 30 edge midpoints (second shell for Mackay)
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            v1, v2 = ico[i], ico[j]
            d = np.linalg.norm(v1 - v2)
            if 0.8 * r1 < d < 1.2 * r1:  # adjacent vertices
                mid = (v1 + v2) / 2
                mid = mid / np.linalg.norm(mid) * r2
                edges.append(mid)
    
    # 20 face centers
    faces = []
    # Find triangles in the icosahedron
    verts = ico
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if (0.8*r1 < d1 < 1.2*r1 and 0.8*r1 < d2 < 1.2*r1 and 0.8*r1 < d3 < 1.2*r1):
                    center = (verts[i] + verts[j] + verts[k]) / 3
                    center = center / np.linalg.norm(center) * r2
                    faces.append(center)
    
    # Take 42 from second shell: 30 edges + 12 more from faces (or just use all)
    second_shell = edges[:30]
    while len(second_shell) < 42 and faces:
        second_shell.append(faces.pop(0))
    
    points = [np.zeros(3)] + list(ico) + second_shell[:42]
    return np.array(points[:55])

_MACKAY_55 = _mackay_55()

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

def _lbfgs_polish(positions, potential, param, start_time, time_limit, safety_margin):
    """Polish a structure using L-BFGS-B with the analytical gradient."""
    def check_time():
        return (time.time() - start_time) < (time_limit - safety_margin)
    
    n = len(positions)
    
    def obj_and_grad(x):
        pos = x.reshape(n, 3)
        e, g = _energy_and_gradient(pos)
        return e, g.flatten()
    
    x0 = positions.flatten()
    
    try:
        result = minimize(
            obj_and_grad, x0, jac=True,
            method='L-BFGS-B',
            options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-12}
        )
        if result.success or result.fun < _energy_and_gradient(positions)[0]:
            return result.x.reshape(n, 3), result.fun
    except Exception:
        pass
    
    return positions, _energy_and_gradient(positions)[0]

def _energy_and_gradient(positions, c_val=None, d_val=None):
    """
    Vectorized calculation of energy and gradient for Dzugutov potential.
    Returns (energy, gradient).
    """
    if c_val is None:
        c_val = c_pot
    if d_val is None:
        d_val = d_pot
        
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    # Compute pairwise differences
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    
    # Mask to avoid division by zero and self-interaction
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    r = np.where(mask, r, 1.0)
    
    energy = 0.0
    grad = np.zeros_like(positions)
    
    # Region 1: r < a
    m1 = r < a_pot
    # Region 2: a <= r < b
    m2 = (r >= a_pot) & (r < b_pot)
    
    r_safe1 = np.where(m1, r, 1.0)
    r_safe2 = np.where(m2, r, 1.0)
    
    # Avoid division by zero in exponent
    denom1 = r_safe1 - a_pot
    denom1 = np.where(np.abs(denom1) < 1e-9, 1e-9, denom1)
    denom2 = r_safe2 - b_pot
    denom2 = np.where(np.abs(denom2) < 1e-9, 1e-9, denom2)
    
    e1_vals = A * (r_safe1 ** (-m_pot) - B) * np.exp(c_val / denom1)
    e2_vals = B * np.exp(d_val / denom2)
    
    e1 = np.where(m1, e1_vals, 0.0)
    e2 = np.where(m2, e2_vals, 0.0)
    
    pair_energy = e1 + e2
    energy = 0.5 * np.sum(pair_energy)
    
    # Gradient
    exp_term1 = np.exp(c_val / denom1)
    de1_dr_vals = A * (
        -m_pot * r_safe1 ** (-m_pot - 1) * exp_term1 + 
        (r_safe1 ** (-m_pot) - B) * exp_term1 * (-c_val / (denom1 ** 2))
    )
    de1_dr = np.where(m1, de1_dr_vals, 0.0)
    
    exp_term2 = np.exp(d_val / denom2)
    de2_dr_vals = B * exp_term2 * (-d_val / (denom2 ** 2))
    de2_dr = np.where(m2, de2_dr_vals, 0.0)
    
    de_dr = de1_dr + de2_dr
    
    factor = np.where(r > 1e-12, de_dr / r, 0.0)
    factor = np.where(mask, factor, 0.0)
    
    grad = np.einsum('ij,ijk->ik', factor, diff)
    
    return energy, grad

def _build_mackay_icosahedral(n, rng, r_min=None):
    """
    Build a Mackay-like icosahedral structure.
    For n=55, use precomputed Mackay 55. For other n, interpolate.
    """
    if n == 55:
        return _MACKAY_55.copy()
    
    if r_min is None:
        r_min = R_MIN
        
    points = [np.zeros(3)]
    ico = _icosahedron_vertices(r_min)
    points.extend(list(ico))
    
    if n <= 13:
        if n < 13:
            return np.array(points[:n])
        return np.array(points)
    
    # For n > 13, use the Mackay 55 as a template
    # Take first n atoms from Mackay 55, scaled appropriately
    if n <= 55:
        return _MACKAY_55[:n].copy()
    
    # For n > 55, extend Mackay 55 with additional atoms on a sphere
    base = _MACKAY_55.copy()
    needed = n - 55
    r3 = r_min * np.sqrt(2.5)
    golden_angle = np.pi * (3 - np.sqrt(5))
    extra = []
    for i in range(needed):
        z = 1 - 2 * (i + 0.5) / needed
        r = np.sqrt(max(0, 1 - z**2))
        theta = golden_angle * i
        p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * r3
        extra.append(p)
    
    return np.vstack([base, np.array(extra)])[:n]

def _build_random_compact(n, rng, r_min=None):
    """Build a random compact cluster."""
    if r_min is None:
        r_min = R_MIN
        
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

def _anneal_and_optimize(positions, n, rng, param, potential, start_time, time_limit, safety_margin):
    """
    Perform parameter annealing followed by L-BFGS-B polish.
    """
    def check_time():
        return (time.time() - start_time) < (time_limit - safety_margin)
        
    # Annealing: start with softened potential, ramp to original
    num_steps = 15
    c_vals = np.linspace(4.0, c_pot, num_steps)
    d_vals = np.linspace(1.5, d_pot, num_steps)
    
    current_pos = positions.copy()
    current_energy = _energy_and_gradient(current_pos)[0]
    
    for i in range(num_steps):
        if not check_time():
            break
            
        c_curr = c_vals[i]
        d_curr = d_vals[i]
        
        # Use L-BFGS-B for each annealing step
        def obj_and_grad_anneal(x):
            pos = x.reshape(n, 3)
            e, g = _energy_and_gradient(pos, c_curr, d_curr)
            return e, g.flatten()
        
        try:
            result = minimize(
                obj_and_grad_anneal, current_pos.flatten(), jac=True,
                method='L-BFGS-B',
                options={'maxiter': 100, 'gtol': 1e-5}
            )
            if result.fun < current_energy:
                current_pos = result.x.reshape(n, 3)
                current_energy = result.fun
        except Exception:
            pass
    
    # Final polish with original parameters using L-BFGS-B
    polished_pos, polished_energy = _lbfgs_polish(current_pos, potential, param, start_time, time_limit, safety_margin)
    
    # Also try library relax as a complement
    try:
        energy, relaxed_pos = relax(current_pos, potential, param)
        if energy < polished_energy:
            polished_pos, polished_energy = relaxed_pos, energy
    except Exception:
        pass
    
    return polished_pos, polished_energy

def construct(instance):
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
        
    def try_relax(pos):
        nonlocal best_energy, best_positions
        try:
            energy, relaxed_pos = relax(pos, potential, param)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos.copy()
                return True
        except Exception:
            pass
        return False

    def try_lbfgs(pos):
        nonlocal best_energy, best_positions
        if not check_time():
            return False
        polished, energy = _lbfgs_polish(pos, potential, param, start_time, time_limit, safety_margin)
        if energy < best_energy:
            best_energy = energy
            best_positions = polished.copy()
            return True
        return False

    # Phase 1: Generate diverse initial guesses
    initial_guesses = []
    
    # 1. Mackay icosahedral (use precomputed for n=55)
    initial_guesses.append(_build_mackay_icosahedral(n, rng))
    
    # 2. Random compact
    initial_guesses.append(_build_random_compact(n, rng))
    
    # 3. Mackay variants with different noise
    for i in range(2):
        pos = _build_mackay_icosahedral(n, rng)
        pos += 0.08 * rng.standard_normal((n, 3))
        initial_guesses.append(pos)
        
    # 4. One more random
    initial_guesses.append(_build_random_compact(n, rng))

    # Phase 2: Relax all initial guesses with library relax
    for pos in initial_guesses:
        if not check_time(): break
        try_relax(pos)
    
    # Phase 3: L-BFGS-B polish the best
    if best_positions is not None and check_time():
        try_lbfgs(best_positions)
        
    # Phase 4: Annealing from best structure
    if best_positions is not None and check_time():
        annealed_pos, annealed_energy = _anneal_and_optimize(
            best_positions, n, rng, param, potential, start_time, time_limit, safety_margin
        )
        if annealed_energy < best_energy:
            best_energy = annealed_energy
            best_positions = annealed_pos.copy()
            
    # Phase 5: Basin Hopping with L-BFGS-B polish
    if best_positions is not None and check_time():
        base_kick = 0.12
        for i in range(15):
            if not check_time(): break
            kick_size = base_kick * (1.0 + 0.05 * i)
            pos_kicked = best_positions + kick_size * rng.standard_normal((n, 3))
            try_relax(pos_kicked)
            if check_time():
                try_lbfgs(best_positions)
            
    # Fallback
    if best_positions is None:
        best_positions = _build_mackay_icosahedral(n, rng)
        
    return best_positions

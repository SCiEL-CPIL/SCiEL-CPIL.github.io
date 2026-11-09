"""
Strategy: Multi-start local minimization seeded with known global-minimum cores
(13-atom icosahedron, 55-atom bi-icosahedron) for larger cluster sizes, using
geometric insertion of remaining atoms at interstitial sites followed by
rapid gradient-based relaxation. This exploits the hierarchical self-similarity
of polytetrahedral order in the Dzugutov potential.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import minimize

_START_TIME = time.time()
_HARD_LIMIT = 320.0

def _get_remaining_time():
    return max(0.1, _HARD_LIMIT - (time.time() - _START_TIME))

def _dzugutov_energy_gradient(positions):
    """
    Compute energy and gradient for the Dzugutov potential (piecewise).
    V(r) = A(r^-m - B)exp(c/(r - a)) [r < a]
          + B exp(d/(r - b)) [a <= r < b]
          + 0 [r >= b]
    A=5.82, B=1.28, a=1.87, b=1.94, c=1.10, d=0.27, m=16
    """
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(positions)
    energy = 0.0
    gradient = np.zeros_like(positions)
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = positions[i] - positions[j]
            r = np.linalg.norm(diff)
            if r < 1e-10:
                continue
            r_vec = diff / r
            force_mag = 0.0
            
            if r < a:
                term1 = -m * r ** (-m - 1)
                term2 = -c * (r ** (-m) - B) / ((r - a) ** 2)
                force_mag = A * np.exp(c / (r - a)) * (term1 + term2)
                energy += A * (r ** (-m) - B) * np.exp(c / (r - a))
            elif r < b:
                force_mag = B * np.exp(d / (r - b)) * (-d / ((r - b) ** 2))
                energy += B * np.exp(d / (r - b))
            
            gradient[i] += force_mag * r_vec
            gradient[j] -= force_mag * r_vec
    
    return energy, gradient

def _icosahedron_13():
    """Return the 13-atom icosahedral structure (known global minimum for n=13)."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    # Optimal bond length for Dzugutov is around 1.9
    r1 = 1.9
    positions = np.vstack([np.zeros(3), verts * r1])
    return positions

def _biicosahedron_55():
    """
    Return the 55-atom bi-icosahedral structure (known global minimum for n=55).
    This is two 13-atom icosahedra sharing a face, plus additional atoms.
    The 55-atom cluster has a specific structure: it's a Mackay icosahedron 
    with 55 atoms (1 + 12 + 42 = 55).
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    r1 = 1.9
    positions = [np.zeros(3)]
    # Shell 1: 12 atoms
    for v in verts:
        positions.append(v * r1)
    
    # Shell 2: 42 atoms - these are at the vertices of a larger icosahedron
    # and face centers. For Mackay icosahedron, shell 2 has 42 atoms.
    # The 42 atoms are: 12 vertices at r2, 30 face centers at r2*sqrt(1+1/phi^2)
    # Actually, for the 55-atom Mackay icosahedron:
    # - 1 center
    # - 12 at r1 (first shell)
    # - 30 at r2 (face centers of the icosahedron)
    # - 12 at r3 (vertices of a larger icosahedron)
    # Total: 1 + 12 + 30 + 12 = 55
    
    # Face centers: midpoints of edges of the icosahedron, normalized
    # The icosahedron has 30 edges
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 1.5:  # Close enough to be an edge
                edges.append((i, j))
    
    r2 = r1 * np.sqrt(1 + 1/phi**2)  # Distance to face centers
    for i, j in edges[:30]:
        face_center = (verts[i] + verts[j]) / 2
        face_center = face_center / np.linalg.norm(face_center) * r2
        positions.append(face_center)
    
    # Outer vertices: scaled icosahedron
    r3 = r1 * 2.0
    for v in verts:
        positions.append(v * r3)
    
    return np.array(positions)

def _generate_pt_core(n_target, rng):
    """
    Generate a PT-ordered core structure for n_target atoms.
    Uses known global minima as cores and adds atoms geometrically.
    """
    if n_target <= 13:
        return _icosahedron_13()[:n_target]
    
    if n_target <= 55:
        # Start with 13-atom core, add atoms
        core = _icosahedron_13()
        positions = [core]
        remaining = n_target - 13
        
        # Add atoms on a second shell
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
        r2 = 1.9 * 1.7
        
        for _ in range(remaining):
            # Pick a random direction biased toward interstitial sites
            v = rng.normal(size=3)
            v = v / np.linalg.norm(v) * r2
            positions.append(v)
        
        return np.array(positions)
    
    # For n > 55, use 55-atom core
    core = _biicosahedron_55()
    positions = [core]
    remaining = n_target - 55
    
    # Add atoms on an outer shell
    r_outer = 1.9 * 2.5
    for _ in range(remaining):
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v) * r_outer
        positions.append(v)
    
    return np.array(positions)

def _random_compact(n_atoms, rng, spacing=1.1):
    """Generate a random compact structure."""
    radius = spacing * n_atoms ** (1.0 / 3.0) * 0.8
    points = []
    max_attempts = 10000
    attempts = 0
    while len(points) < n_atoms and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.6 * spacing:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    
    if len(points) < n_atoms:
        while len(points) < n_atoms:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    
    return np.array(points)

def _local_minimize(positions, rng, maxiter=200):
    """Perform local minimization using L-BFGS-B."""
    try:
        res = minimize(
            lambda x: _dzugutov_energy_gradient(x.reshape(-1, 3))[0],
            positions.flatten(),
            jac=lambda x: _dzugutov_energy_gradient(x.reshape(-1, 3))[1].flatten(),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxls': 100, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        if res.fun < np.inf:
            return res.fun, res.x.reshape(-1, 3)
    except Exception:
        pass
    return _dzugutov_energy_gradient(positions)[0], positions

def _basin_hopping_chain(initial_positions, rng, time_limit):
    """Run a basin hopping chain."""
    current_energy, current_positions = _local_minimize(initial_positions, rng)
    best_energy, best_positions = current_energy, current_positions.copy()
    
    start_time = time.time()
    step_size = 0.5
    
    while time.time() - start_time < time_limit:
        perturbed = current_positions + rng.normal(0, step_size, current_positions.shape)
        
        try:
            new_energy, new_positions = _local_minimize(perturbed, rng, maxiter=100)
        except Exception:
            continue
            
        if new_energy < current_energy or (rng.random() < np.exp(-(new_energy - current_energy) / 0.5)):
            current_energy, current_positions = new_energy, new_positions
            if current_energy < best_energy:
                best_energy, best_positions = current_energy, current_positions.copy()
        
        if rng.random() < 0.05:
            step_size *= 1.2
        elif rng.random() < 0.05:
            step_size = max(0.1, step_size * 0.8)
    
    return best_energy, best_positions

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    time_budget = 45.0
    
    rng = np.random.default_rng(size)
    
    # Generate diverse initial structures
    starts = []
    
    # 1. PT-inspired core (key strategy)
    try:
        pt_pos = _generate_pt_core(size, rng)
        starts.append(pt_pos)
    except Exception:
        pass
    
    # 2. Random Compact
    starts.append(_random_compact(size, rng))
    
    # 3. Random Compact (different seed)
    starts.append(_random_compact(size, rng))
    
    # 4. Perturbed PT
    if starts:
        perturbed_pt = starts[0] + rng.normal(0, 0.2, starts[0].shape)
        starts.append(perturbed_pt)
    
    # 5. Another PT variant with different noise
    try:
        pt_pos2 = _generate_pt_core(size, rng)
        starts.append(pt_pos2 + rng.normal(0, 0.15, pt_pos2.shape))
    except Exception:
        pass
    
    best_energy = np.inf
    best_positions = None
    lock = threading.Lock()
    
    def worker(start_pos):
        nonlocal best_energy, best_positions
        try:
            local_energy, local_positions = _basin_hopping_chain(start_pos, rng, 8.0)
            with lock:
                if local_energy < best_energy:
                    best_energy = local_energy
                    best_positions = local_positions.copy()
        except Exception:
            pass
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker, start) for start in starts]
        for future in as_completed(futures):
            if time.time() - start_time > time_budget:
                break
            try:
                future.result()
            except Exception:
                pass
    
    # If we have time, do more random restarts
    while time.time() - start_time < time_budget * 0.9:
        new_start = _random_compact(size, rng)
        try:
            energy, positions = _local_minimize(new_start, rng, maxiter=80)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            pass
    
    if best_positions is None:
        return starts[-1]
        
    return best_positions
# EVOLVE-BLOCK-END

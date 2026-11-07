"""Sutton-Chen cluster global minima search using Simulated Annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _fcc_positions(n, a=0.72):
    """Generate n positions from an FCC lattice fragment."""
    basis = [
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0],
    ]
    side = int(np.ceil((n / 4) ** (1/3))) + 2
    candidates = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for b in basis:
                    p = np.array([i * a + b[0] * a, j * a + b[1] * a, k * a + b[2] * a])
                    candidates.append(p)
    candidates = np.array(candidates)
    center = candidates.mean(axis=0)
    candidates -= center
    dists = np.linalg.norm(candidates, axis=1)
    idx = np.argsort(dists)[:n]
    return candidates[idx]


def _bcc_positions(n, a=0.72):
    """Generate n positions from a BCC lattice fragment."""
    basis = [
        [0, 0, 0],
        [0.5, 0.5, 0.5],
    ]
    side = int(np.ceil((n / 2) ** (1/3))) + 2
    candidates = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for b in basis:
                    p = np.array([i * a + b[0] * a, j * a + b[1] * a, k * a + b[2] * a])
                    candidates.append(p)
    candidates = np.array(candidates)
    center = candidates.mean(axis=0)
    candidates -= center
    dists = np.linalg.norm(candidates, axis=1)
    idx = np.argsort(dists)[:n]
    return candidates[idx]


def _hcp_positions(n, a=0.72):
    """Generate n positions from an HCP lattice fragment."""
    c_over_a = np.sqrt(8.0/3.0)
    basis = [
        [0, 0, 0],
        [1/3, 1/np.sqrt(3), 0],
        [2/3, -1/np.sqrt(3), 0],
        [0, 0, c_over_a/2],
        [1/3, 1/np.sqrt(3), c_over_a/2],
        [2/3, -1/np.sqrt(3), c_over_a/2],
    ]
    side = int(np.ceil((n / 6) ** (1/3))) + 2
    candidates = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for b in basis:
                    x = i * a + b[0] * a
                    y = j * a + b[1] * a
                    z = k * a * c_over_a + b[2] * a * c_over_a
                    candidates.append([x, y, z])
    candidates = np.array(candidates)
    center = candidates.mean(axis=0)
    candidates -= center
    dists = np.linalg.norm(candidates, axis=1)
    idx = np.argsort(dists)[:n]
    return candidates[idx]


def _icosahedron_positions(n, r=1.0):
    """Generate positions on/inside an icosahedral structure."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * r
    
    if n <= 13:
        pts = [np.zeros(3)] + list(verts[:n-1])
        return np.array(pts)
    
    pts = [np.zeros(3)]
    for v in verts:
        pts.append(v)
    if n <= 13:
        return np.array(pts[:n])
    
    def fib_sphere(n_pts, radius):
        pts = []
        golden = (1 + np.sqrt(5)) / 2
        for i in range(n_pts):
            z = 1 - 2 * (i + 0.5) / n_pts
            r = np.sqrt(1 - z*z)
            theta = 2 * np.pi * i / golden
            pts.append([r * np.cos(theta), r * np.sin(theta), z])
        return np.array(pts) * radius
    
    # Approximate Mackay shells
    shell1 = fib_sphere(12, 1.0)
    pts.extend([p for p in shell1])
    if n <= 13:
        return np.array(pts[:n])
        
    shell2 = fib_sphere(42, 1.8)
    pts.extend([p for p in shell2])
    if n <= 55:
        return np.array(pts[:n])
        
    shell3 = fib_sphere(92, 2.6)
    pts.extend([p for p in shell3])
    
    return np.array(pts[:n])


def _random_compact(n, spacing=0.72, seed=42):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    points = [np.zeros(3)]
    radius = spacing * n ** (1.0/3.0) * 0.9
    
    attempts = 0
    while len(points) < n and attempts < n * 200:
        attempts += 1
        p = rng.normal(0, radius/2, 3)
        if np.linalg.norm(p) > radius:
            continue
        min_dist = np.min([np.linalg.norm(p - q) for q in points]) if points else np.inf
        if min_dist >= 0.6 * spacing:
            points.append(p)
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            points.append(p)
    
    return np.array(points)


def _sa_optimize(positions, potential, kind, max_iters=5000, t_start=1.0, t_end=0.01, rng=None, time_limit=None):
    """Simulated Annealing optimization with adaptive step size and periodic relaxation."""
    if rng is None:
        rng = np.random.default_rng()
    
    n = len(positions)
    current_energy, _ = sutton_chen_energy_gradient(positions, kind)
    best_energy = current_energy
    best_positions = positions.copy()
    
    # First, relax the initial structure to get into a local minimum
    try:
        current_energy, positions = relax(positions, potential, kind)
        best_energy = current_energy
        best_positions = positions.copy()
    except:
        pass
    
    temperature = t_start
    cooling_rate = (t_end / t_start) ** (1.0 / max_iters)
    
    # Track acceptance rate to adapt step size
    accepted = 0
    total = 0
    step_scale = 1.0
    
    for i in range(max_iters):
        if time_limit is not None and time.time() > time_limit:
            break
            
        # Adaptive step size based on recent acceptance rate
        if total > 50:
            acceptance = accepted / total
            if acceptance < 0.2:
                step_scale *= 0.8
            elif acceptance > 0.8:
                step_scale *= 1.2
            step_scale = np.clip(step_scale, 0.1, 10.0)
            accepted = 0
            total = 0
        
        # Generate a random perturbation
        step_size = 0.1 * temperature * step_scale + 0.01
        delta = rng.normal(0, step_size, (n, 3))
        new_positions = positions + delta
        
        try:
            new_energy, _ = sutton_chen_energy_gradient(new_positions, kind)
            if not np.isfinite(new_energy):
                temperature *= cooling_rate
                continue
        except:
            temperature *= cooling_rate
            continue
            
        delta_e = new_energy - current_energy
        total += 1
        
        # Metropolis criterion
        if delta_e < 0 or rng.random() < np.exp(-delta_e / temperature):
            accepted += 1
            current_energy = new_energy
            positions = new_positions
            
            # Periodically relax to local minimum to stay in basin
            if i % 30 == 0:
                try:
                    relaxed_energy, relaxed_positions = relax(positions, potential, kind)
                    if relaxed_energy < current_energy:
                        current_energy = relaxed_energy
                        positions = relaxed_positions
                except:
                    pass
                    
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = positions.copy()
                
        temperature *= cooling_rate
        
    return best_energy, best_positions


def construct(instance):
    """Construct Sutton-Chen cluster using Simulated Annealing."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance["kind"]
    
    start_time = time.time()
    time_limit = 280.0  # Leave buffer for 300s total
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Random compact
    inits.append(_random_compact(size, spacing=0.72, seed=1))
    
    # 2. FCC fragment
    try:
        inits.append(_fcc_positions(size, a=0.72))
    except:
        pass
        
    # 3. Icosahedron
    try:
        inits.append(_icosahedron_positions(size, r=0.72 * size**(1/3)))
    except:
        pass
        
    # 4. BCC fragment
    try:
        inits.append(_bcc_positions(size, a=0.72))
    except:
        pass
        
    # 5. HCP fragment
    try:
        inits.append(_hcp_positions(size, a=0.72))
    except:
        pass
        
    # 6. Another random with different seed
    inits.append(_random_compact(size, spacing=0.72, seed=2))
    
    # 7. Random with larger radius
    inits.append(_random_compact(size, spacing=0.85, seed=3))
    
    best_energy = np.inf
    best_positions = None
    
    # Run SA on each initialization
    for i, init in enumerate(inits):
        if time.time() - start_time > time_limit:
            break
            
        # Adjust max_iters based on time remaining and number of inits left
        time_remaining = time_limit - (time.time() - start_time)
        inits_remaining = len(inits) - i
        max_iters = int(4000 * (time_remaining / time_limit) / inits_remaining)
        if max_iters < 50:
            max_iters = 50
            
        try:
            energy, positions = _sa_optimize(init, potential, kind, max_iters=max_iters, 
                                             time_limit=time_limit - 10)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except:
            continue
            
    if best_positions is None:
        # Fallback
        rng = np.random.default_rng(size)
        radius = 0.72 * size ** (1/3)
        best_positions = rng.uniform(-radius, radius, (size, 3))
        try:
            best_positions = relax(best_positions, potential, kind)[1]
        except:
            pass

    return best_positions
# EVOLVE-BLOCK-END

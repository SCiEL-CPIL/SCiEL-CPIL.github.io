"""Sutton-Chen transition-metal cluster global minima - SA + Structural Initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _fcc_fragment(n, seed=42):
    rng = np.random.default_rng(seed)
    a = 1.0
    R = (3.0 * n / 4.0) ** (1.0 / 3.0) * a * 1.2
    points = []
    max_idx = int(np.ceil(R / (a/2))) + 2
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                for b in [(0,0,0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    x = (i + b[0]) * a
                    y = (j + b[1]) * a
                    z = (k + b[2]) * a
                    if np.linalg.norm([x, y, z]) <= R:
                        points.append([x, y, z])
    if len(points) < n:
        return None
    points = np.array(points)
    idx = rng.permutation(len(points))[:n]
    return points[idx]

def _hcp_fragment(n, seed=42):
    rng = np.random.default_rng(seed)
    a = 1.0
    c = 1.633 * a
    R = (3.0 * n / 4.0) ** (1.0 / 3.0) * a * 1.2
    points = []
    max_idx = int(np.ceil(R / (a/2))) + 2
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                z = k * c
                x_a = i * a + j * (-0.5 * a)
                y_a = j * (np.sqrt(3)/2 * a)
                if np.linalg.norm([x_a, y_a, z]) <= R:
                    points.append([x_a, y_a, z])
                x_b = x_a + 0.0
                y_b = y_a + a * np.sqrt(3) / 3.0
                z_b = z + c / 2.0
                if np.linalg.norm([x_b, y_b, z_b]) <= R:
                    points.append([x_b, y_b, z_b])
    if len(points) < n:
        return None
    points = np.array(points)
    idx = rng.permutation(len(points))[:n]
    return points[idx]

def _icosahedral_fragment(n, seed=42):
    rng = np.random.default_rng(seed)
    phi = (1 + np.sqrt(5)) / 2
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    points = [[0, 0, 0]]
    points.extend(v * 1.0)
    if n > 13:
        while len(points) < n:
            z = rng.uniform(-1, 1)
            theta = rng.uniform(0, 2 * np.pi)
            r = np.sqrt(1 - z*z)
            p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * 2.0
            if all(np.linalg.norm(p - np.array(q)) > 0.8 for q in points):
                points.append(p)
    if len(points) < n:
        return None
    return np.array(points[:n])

def _random_packing(n, seed=42):
    rng = np.random.default_rng(seed)
    points = []
    radius = 1.0 * n ** (1.0/3.0) * 0.8
    min_dist = 0.8
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - np.array(q)) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    if len(points) < n:
        return None
    return np.array(points[:n])

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    start_time = time.time()
    deadline = start_time + 280.0
    
    best_energy = np.inf
    best_positions = None
    
    initializers = [
        _fcc_fragment,
        _hcp_fragment,
        _icosahedral_fragment,
        _random_packing
    ]
    
    T0 = 5.0
    cooling = 0.99
    step_size = 0.5
    
    seed_counter = 0
    
    while time.time() < deadline:
        init_func = initializers[seed_counter % len(initializers)]
        positions = init_func(size, seed=seed_counter)
        
        if positions is None:
            seed_counter += 1
            continue
            
        T = T0
        curr_energy, _ = sutton_chen_energy_gradient(positions, kind)
        
        if not np.isfinite(curr_energy):
            seed_counter += 1
            continue
            
        for step in range(1000):
            if time.time() > deadline:
                break
                
            atom_idx = np.random.randint(size)
            move = np.random.normal(0, step_size, 3)
            new_positions = positions.copy()
            new_positions[atom_idx] += move
            
            # Check min distance
            diffs = new_positions[:, np.newaxis, :] - new_positions[np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=-1)
            min_d = np.min(dists[np.triu_indices(size, k=1)])
            
            if min_d < 0.35:
                continue
                
            new_energy, _ = sutton_chen_energy_gradient(new_positions, kind)
            
            if new_energy < curr_energy:
                curr_energy = new_energy
                positions = new_positions
            else:
                delta = new_energy - curr_energy
                if np.random.rand() < np.exp(-delta / T):
                    curr_energy = new_energy
                    positions = new_positions
            
            T *= cooling
            
        energy, positions = relax(positions, potential, kind)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = positions
            
        seed_counter += 1
        
    if best_positions is None:
        return np.random.default_rng(0).uniform(-1, 1, (size, 3))
        
    return best_positions
# EVOLVE-BLOCK-END

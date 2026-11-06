"""Dzugutov cluster global minima: Mackay icosahedral seeds + basin hopping with parallel restarts."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START = time.time()
TIME_LIMIT = 285.0  # safety margin under 300s kill

# Dzugutov parameters
_A, _B = 5.82, 1.28
_a, _b = 1.87, 1.94
_c, _d = 1.10, 0.27
_m = 16


def _elapsed():
    return time.time() - _START


def _remaining():
    return TIME_LIMIT - _elapsed()


def _energy(positions):
    """Direct Dzugutov energy (vectorised)."""
    pos = np.asarray(positions, dtype=float)
    n = pos.shape[0]
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(n, k=1)
    r_ij = r[iu]
    
    # V(r) = A(r^-m - B)exp(c/(r-a)) [r < a] + B exp(d/(r-b)) [r < b]
    # For r >= b, V = 0
    # For a <= r < b, only the second term: B exp(d/(r-b))
    # For r < a, both terms: A(r^-m - B)exp(c/(r-a)) + B exp(d/(r-b))
    
    e = np.zeros_like(r_ij)
    
    # Second term: B exp(d/(r-b)) for r < b
    mask_b = r_ij < _b
    if np.any(mask_b):
        r_b = r_ij[mask_b]
        # Avoid division by zero or negative
        r_b_safe = np.maximum(r_b, 1e-10)
        e[mask_b] += _B * np.exp(_d / (r_b_safe - _b))
    
    # First term: A(r^-m - B)exp(c/(r-a)) for r < a
    mask_a = r_ij < _a
    if np.any(mask_a):
        r_a = r_ij[mask_a]
        r_a_safe = np.maximum(r_a, 1e-10)
        e[mask_a] += _A * (r_a_safe ** (-_m) - _B) * np.exp(_c / (r_a_safe - _a))
    
    return float(np.sum(e))


def _mckay_icosahedron(n):
    """Generate a Mackay icosahedral cluster of n atoms.
    
    Mackay icosahedra have shells:
    - Shell 0: 1 atom at center
    - Shell 1: 12 atoms (vertices of icosahedron)
    - Shell 2: 42 atoms
    - Shell 3: 92 atoms
    Total: 1, 13, 55, 147...
    
    For arbitrary n, we build up shells and truncate.
    """
    # Icosahedron vertices (unit icosahedron)
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    # 12 vertices of regular icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Shell 1: 12 atoms at distance d1 from center
    # For Dzugutov, optimal nearest-neighbor distance is around 1.0-1.1
    d1 = 1.05  # first shell radius
    
    atoms = [np.zeros(3)]  # center atom
    
    # Add shell 1 atoms
    if n > 1:
        shell1 = verts * d1
        atoms.extend(shell1.tolist())
    
    # If we need more atoms, add shell 2
    # Shell 2 of Mackay icosahedron: 42 atoms
    # These are at the midpoints of edges and face centers, scaled appropriately
    if n > 13:
        # Generate shell 2 positions
        # Edge midpoints: 30 edges -> 30 points at distance ~d1*sqrt(3/2)
        # Face centers: 20 faces -> 20 points at distance ~d1*sqrt(3/5)
        
        # Edge midpoints (normalize to appropriate radius)
        edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                v1, v2 = verts[i], verts[j]
                dist = np.linalg.norm(v1 - v2)
                if 0.5 < dist < 1.5:  # connected vertices
                    edges.append((v1 + v2) / 2)
        
        # Face centers
        faces = []
        # Find triangular faces of icosahedron
        from itertools import combinations
        for tri in combinations(range(12), 3):
            i, j, k = tri
            v1, v2, v3 = verts[i], verts[j], verts[k]
            # Check if it's a face (all three connected)
            d12 = np.linalg.norm(v1 - v2)
            d13 = np.linalg.norm(v1 - v3)
            d23 = np.linalg.norm(v2 - v3)
            if all(0.5 < d < 1.5 for d in [d12, d13, d23]):
                face_center = (v1 + v2 + v3) / 3
                faces.append(face_center)
        
        # Scale edge midpoints and face centers to appropriate radii
        # Edge midpoint distance from center
        if edges:
            edge_mids = np.array(edges)
            edge_mids /= np.linalg.norm(edge_mids, axis=1, keepdims=True)
            d2_edge = d1 * np.sqrt(3.0 / 2.0)  # approximate
            shell2_edge = edge_mids * d2_edge
        else:
            shell2_edge = []
        
        if faces:
            face_centers = np.array(faces)
            face_centers /= np.linalg.norm(face_centers, axis=1, keepdims=True)
            d2_face = d1 * np.sqrt(3.0 / 5.0)  # approximate
            shell2_face = face_centers * d2_face
        else:
            shell2_face = []
        
        # Add shell 2 atoms
        remaining = n - len(atoms)
        shell2_atoms = list(shell2_edge) + list(shell2_face)
        # Shuffle to get a consistent ordering
        rng = np.random.default_rng(42)
        rng.shuffle(shell2_atoms)
        atoms.extend(shell2_atoms[:remaining])
    
    # If we still need more atoms, add random positions in a larger shell
    if len(atoms) < n:
        remaining = n - len(atoms)
        rng = np.random.default_rng(123)
        for _ in range(remaining):
            # Random direction
            dir_vec = rng.normal(size=3)
            dir_vec /= np.linalg.norm(dir_vec)
            # Random radius between d1 and 2*d1
            r = rng.uniform(d1 * 1.2, d1 * 2.0)
            atoms.append(dir_vec * r)
    
    return np.array(atoms[:n])


def _basin_hopping(positions, potential, parameter, max_iter=50, step_size=0.3, seed=0):
    """Basin hopping: random perturbations + local relaxation."""
    rng = np.random.default_rng(seed)
    pos = np.array(positions, dtype=float)
    best_energy, best_pos = _energy(pos), pos.copy()
    
    # Initial relaxation
    energy, pos = relax(pos, potential, parameter)
    best_energy, best_pos = energy, pos.copy()
    
    for it in range(max_iter):
        if _remaining() < 5.0:
            break
        
        # Random perturbation
        noise = rng.normal(0, step_size, pos.shape)
        new_pos = pos + noise
        
        # Relax
        energy, new_pos = relax(new_pos, potential, parameter)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = new_pos.copy()
        
        pos = new_pos
    
    return best_energy, best_pos


def _single_search(n, potential, parameter, seed, time_budget):
    """Run a single search thread with time budget."""
    start_time = time.time()
    
    # Generate initial icosahedral structure
    init_pos = _mckay_icosahedron(n)
    
    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(seed)
    init_pos += rng.normal(0, 0.05, init_pos.shape)
    
    # Basin hopping
    max_iter = 30
    step_size = 0.25
    best_energy, best_pos = _basin_hopping(init_pos, potential, parameter, 
                                            max_iter=max_iter, step_size=step_size, seed=seed)
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy Dzugutov cluster structure using icosahedral seeds and basin hopping."""
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Time budget for this instance
    time_budget = min(_remaining() - 10.0, 45.0)  # Leave time for other instances
    if time_budget < 5.0:
        time_budget = 5.0
    
    # Run multiple parallel searches with different seeds
    n_threads = 4
    seeds = list(range(n_threads))
    
    best_energy = float('inf')
    best_pos = None
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        for seed in seeds:
            future = executor.submit(_single_search, n, potential, parameter, seed, time_budget)
            futures.append(future)
        
        for future in futures:
            try:
                energy, pos = future.result(timeout=_remaining() - 2.0)
                if energy < best_energy:
                    best_energy = energy
                    best_pos = pos
            except Exception:
                continue
    
    # If no valid result, use a simple relaxed icosahedron
    if best_pos is None:
        init_pos = _mckay_icosahedron(n)
        best_energy, best_pos = relax(init_pos, potential, parameter)
    
    return best_pos
# EVOLVE-BLOCK-END

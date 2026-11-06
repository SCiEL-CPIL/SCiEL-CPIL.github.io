"""Dzugutov cluster global minima: Mackay icosahedral seeds + basin hopping with parallelism."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Global deadline tracking
_START_TIME = None
_DEADLINE = 300.0  # seconds, with safety margin
_THREAD_LOCK = threading.Lock()


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _dzugutov_energy(pos):
    """Compute Dzugutov energy vectorized."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = dist[i, j]
            if r < a:
                energy += A * (r ** (-m) - B) * np.exp(c / (r - a))
            elif r < b:
                energy += B * np.exp(d / (r - b))
    return energy


def _icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit sphere
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    vertices = vertices / norms
    return vertices


def _generate_mackay_seed(n, rng):
    """Generate a Mackay icosahedral seed for n atoms."""
    ico_verts = _icosahedron_vertices()
    
    # Mackay icosahedron layers: center (1), inner shell (12), outer shell (42), next (92)
    # For 38-60 atoms, we use center + inner + part of outer
    layers = []
    
    # Center atom
    layers.append(np.array([[0.0, 0.0, 0.0]]))
    remaining = n - 1
    
    # Inner shell: 12 atoms at radius r1
    if remaining >= 12:
        r1 = 1.0
        inner = ico_verts * r1
        # Add small random perturbations to break perfect symmetry
        inner = inner + rng.normal(0, 0.05, size=inner.shape)
        layers.append(inner)
        remaining -= 12
    else:
        # Take only 'remaining' atoms from inner shell
        r1 = 1.0
        inner = ico_verts[:remaining] * r1
        inner = inner + rng.normal(0, 0.05, size=inner.shape)
        layers.append(inner)
        remaining = 0
    
    # Outer shell: 42 atoms (positions derived from icosahedron)
    if remaining > 0:
        # Generate outer shell positions using icosahedral symmetry
        # The outer shell of a Mackay icosahedron has 42 atoms
        # We can approximate by placing atoms on a larger icosahedron with additional points
        r2 = 1.9  # approximate radius for outer shell
        
        # Create additional points by interpolating and adding to icosahedron
        outer_positions = []
        # Use the 12 vertices scaled
        for v in ico_verts:
            outer_positions.append(v * r2)
        
        # Add 30 more positions from edge midpoints of icosahedron (approximate)
        # Icosahedron has 30 edges
        phi = (1 + np.sqrt(5)) / 2
        # Edge midpoints of unit icosahedron
        edge_midpoints = []
        ico_edges = [
            (0,1),(0,2),(0,4),(0,6),(0,8),(0,10),
            (1,3),(1,4),(1,6),(1,9),(1,11),
            (2,3),(2,5),(2,7),(2,8),(2,10),
            (3,5),(3,7),(3,9),(3,11),
            (4,5),(4,7),(4,9),
            (5,6),(5,8),(5,10),
            (6,7),(6,8),(6,10),
            (7,9),(7,11),
            (8,9),(8,11),
            (9,10),(10,11)
        ]
        for i, j in ico_edges:
            mid = (ico_verts[i] + ico_verts[j]) / 2
            norm = np.linalg.norm(mid)
            if norm > 0:
                mid = mid / norm
            edge_midpoints.append(mid * r2)
        
        outer_positions = np.array(outer_positions[:42])
        # Take only 'remaining' atoms
        outer_positions = outer_positions[:remaining]
        outer_positions = outer_positions + rng.normal(0, 0.05, size=outer_positions.shape)
        layers.append(outer_positions)
        remaining = 0
    
    # If still need more atoms, add random compact positions
    if remaining > 0:
        radius = 2.5
        extra = []
        while len(extra) < remaining:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                extra.append(p)
        extra = np.array(extra)
        layers.append(extra)
    
    positions = np.vstack(layers)
    return positions


def _fix_min_distance(positions, min_dist=0.5):
    """Ensure all atoms are at least min_dist apart by pushing overlapping atoms."""
    n = len(positions)
    for _ in range(100):  # max iterations
        min_found = min_dist
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist:
                    # Push atoms apart
                    if d < 1e-10:
                        direction = np.random.randn(3)
                        direction = direction / np.linalg.norm(direction)
                    else:
                        direction = (positions[j] - positions[i]) / d
                    push = (min_dist - d) / 2 + 0.01
                    positions[i] -= direction * push
                    positions[j] += direction * push
        # Check if all distances are satisfied
        all_ok = True
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(positions[i] - positions[j]) < min_dist - 0.01:
                    all_ok = False
                    break
            if not all_ok:
                break
        if all_ok:
            break
    return positions


def _basin_hopping(positions, rng, max_iterations=50, perturbation_scale=0.5):
    """Perform basin hopping: perturb, relax, accept/reject."""
    current_pos = positions.copy()
    current_energy = _dzugutov_energy(current_pos)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    for _ in range(max_iterations):
        if _remaining() < 5.0:
            break
        
        # Perturb
        perturbation = rng.normal(0, perturbation_scale, size=current_pos.shape)
        new_pos = current_pos + perturbation
        new_pos = _fix_min_distance(new_pos)
        
        # Relax
        try:
            relaxed_energy, relaxed_pos = relax(new_pos, 'dzugutov', None)
        except Exception:
            continue
        
        # Metropolis acceptance
        delta_e = relaxed_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / 0.1):
            current_pos = relaxed_pos
            current_energy = relaxed_energy
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy Dzugutov cluster using Mackay seeds and basin hopping."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n = instance["n"]
    potential = instance["potential"]
    rng = np.random.default_rng(42 + n)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate multiple Mackay-based seeds and run basin hopping
    n_seeds = min(8, max(3, int(_remaining() / 20)))
    
    seeds = []
    for i in range(n_seeds):
        seed = _generate_mackay_seed(n, rng)
        seed = _fix_min_distance(seed)
        seeds.append(seed)
    
    # Also add some random compact seeds for diversity
    for i in range(3):
        radius = 1.1 * n ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 for q in points):
                points.append(p)
        if len(points) == n:
            points = np.array(points)
            points = _fix_min_distance(points)
            seeds.append(points)
    
    # Process seeds in parallel using threads
    def process_seed(seed):
        if _remaining() < 10.0:
            return None
        try:
            # First, do a quick local relaxation
            rel_energy, rel_pos = relax(seed, 'dzugutov', None)
            
            # Then basin hopping
            bh_energy, bh_pos = _basin_hopping(rel_pos, rng, max_iterations=30, perturbation_scale=0.3)
            return bh_energy, bh_pos
        except Exception:
            return None
    
    # Use thread pool for parallelism
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_seed, seed): seed for seed in seeds}
        for future in as_completed(futures):
            if _remaining() < 5.0:
                break
            result = future.result()
            if result is not None:
                energy, positions = result
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
    
    # If no valid result, fall back to simple relaxation
    if best_positions is None:
        seed = _generate_mackay_seed(n, rng)
        seed = _fix_min_distance(seed)
        try:
            _, best_positions = relax(seed, 'dzugutov', None)
        except Exception:
            # Last resort: random compact
            radius = 1.1 * n ** (1.0 / 3.0) * 0.75
            points = []
            while len(points) < n:
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 for q in points):
                    points.append(p)
            best_positions = np.array(points)
            best_positions = _fix_min_distance(best_positions)
    
    return best_positions
# EVOLVE-BLOCK-END

"""Sutton-Chen transition-metal cluster global minima - multi-start with basin hopping."""
import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Global deadline: 300s wall-clock limit, use 280s for safety
_DEADLINE = time.time() + 280.0


def _energy_grad(positions, kind):
    """Compute energy and gradient."""
    try:
        return sutton_chen_energy_gradient(positions, kind)
    except Exception:
        return np.inf, np.zeros_like(positions)


def _min_dist(positions):
    """Compute minimum pairwise distance."""
    n = len(positions)
    if n < 2:
        return np.inf
    min_d = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_d:
                min_d = d
    return min_d


def _valid(positions, min_sep=0.35):
    """Check if positions are valid (finite, no atoms too close)."""
    if not np.all(np.isfinite(positions)):
        return False
    if _min_dist(positions) < min_sep:
        return False
    return True


def _fcc_cluster(n, spacing=1.0):
    """Generate n positions on an FCC lattice, packed compactly."""
    a = spacing * np.sqrt(2.0)  # lattice constant for nn distance = spacing
    points = []
    nmax = 4
    for i in range(-nmax, nmax + 1):
        for j in range(-nmax, nmax + 1):
            for k in range(-nmax, nmax + 1):
                for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    x = (i + dx) * a
                    y = (j + dy) * a
                    z = (k + dz) * a
                    points.append([x, y, z])
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:min(n, len(points))]
    return points[idx]


def _hcp_cluster(n, spacing=1.0):
    """Generate n positions on an HCP lattice, packed compactly."""
    a = spacing  # nearest neighbor distance
    c = a * np.sqrt(8.0 / 3.0)  # ideal c/a ratio
    points = []
    nmax = 4
    for layer in range(-nmax, nmax + 1):
        z = layer * c
        for i in range(-nmax, nmax + 1):
            for j in range(-nmax, nmax + 1):
                if layer % 2 == 0:
                    x = i * a + (j % 2) * a / 2.0
                    y = j * a * np.sqrt(3.0) / 2.0
                else:
                    x = i * a + a / 2.0 + (j % 2) * a / 2.0
                    y = j * a * np.sqrt(3.0) / 2.0
                points.append([x, y, z])
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:min(n, len(points))]
    return points[idx]


def _mackay_cluster(n, spacing=1.0):
    """Generate a Mackay icosahedral cluster."""
    phi = (1 + np.sqrt(5)) / 2.0
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # Shell radii for Mackay icosahedron
    R1 = spacing / (2.0 * np.sin(np.pi / 5.0))  # 12 vertices
    R2 = R1 * np.sqrt(2.0)  # 30 edges
    
    points = [np.array([0.0, 0.0, 0.0])]  # center
    
    # First shell: 12 icosahedron vertices
    for v in ico_verts:
        points.append(v * R1)
    
    # Second shell: 30 edge midpoints (scaled)
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(ico_verts[i] - ico_verts[j])
            if d < 1.5:  # adjacent vertices
                mid = (ico_verts[i] + ico_verts[j]) / 2.0
                mid = mid / np.linalg.norm(mid) * R2
                points.append(mid)
    
    points = np.array(points)
    if len(points) > n:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:n]
        return points[idx]
    else:
        return points


def _random_compact(n, rng, radius=None):
    """Generate n random positions in a compact sphere."""
    if radius is None:
        radius = 0.8 * n ** (1.0 / 3.0)
    positions = rng.uniform(-radius, radius, (n, 3))
    return positions


def _spherical_shell(n, rng, radius=None):
    """Generate n positions on a spherical shell."""
    if radius is None:
        radius = 1.0 * n ** (1.0 / 3.0)
    theta = rng.uniform(0, 2 * np.pi, n)
    cos_phi = rng.uniform(-1, 1, n)
    sin_phi = np.sqrt(1 - cos_phi**2)
    positions = np.column_stack([
        radius * sin_phi * np.cos(theta),
        radius * sin_phi * np.sin(theta),
        radius * cos_phi
    ])
    return positions


def _de_step(positions, kind, rng, step_size=0.5):
    """Perform a DE mutation step."""
    n = len(positions)
    if n < 4:
        return positions + rng.normal(0, step_size, positions.shape)
    
    i = rng.integers(0, n)
    a, b, c = rng.choice(n, 3, replace=False)
    while a == b or a == c or b == c:
        a, b, c = rng.choice(n, 3, replace=False)
    
    mutant = positions.copy()
    mutant[i] = positions[a] + 0.5 * (positions[b] - positions[c])
    return mutant


def _basin_hopping(positions, kind, rng, max_iter=50, step_size=0.5, time_budget=30.0):
    """Basin hopping optimization."""
    current_pos = positions.copy()
    current_energy, _ = _energy_grad(current_pos, kind)
    
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    start_time = time.time()
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5:
            break
        if time.time() - start_time > time_budget:
            break
        
        # Generate perturbation
        perturbed = _de_step(current_pos, kind, rng, step_size)
        
        # Relax perturbed configuration
        try:
            new_energy, new_pos = relax(perturbed, 'sutton_chen', kind)
        except Exception:
            continue
        
        if not _valid(new_pos):
            continue
        
        # Accept if lower energy or with Metropolis criterion
        if new_energy < current_energy:
            current_pos = new_pos
            current_energy = new_energy
        else:
            # Metropolis acceptance
            delta_e = new_energy - current_energy
            if delta_e < 0 or rng.random() < np.exp(-delta_e / 0.1):
                current_pos = new_pos
                current_energy = new_energy
        
        # Update best
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
    
    return best_energy, best_pos


def construct(instance):
    """
    Multi-start with basin hopping for Sutton-Chen clusters.
    """
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    rng = np.random.default_rng(size * 7919 + 12345)
    
    # Generate diverse initial structures
    initial_structures = []
    
    # FCC
    try:
        fcc = _fcc_cluster(size, spacing=1.0)
        if len(fcc) >= size:
            initial_structures.append(fcc[:size])
    except Exception:
        pass
    
    # HCP
    try:
        hcp = _hcp_cluster(size, spacing=1.0)
        if len(hcp) >= size:
            initial_structures.append(hcp[:size])
    except Exception:
        pass
    
    # Mackay icosahedral
    try:
        mackay = _mackay_cluster(size, spacing=1.0)
        if len(mackay) >= size:
            initial_structures.append(mackay[:size])
    except Exception:
        pass
    
    # Random compact
    try:
        random_compact = _random_compact(size, rng, radius=0.8 * size ** (1.0 / 3.0))
        initial_structures.append(random_compact)
    except Exception:
        pass
    
    # Spherical shell
    try:
        sphere = _spherical_shell(size, rng, radius=1.0 * size ** (1.0 / 3.0))
        initial_structures.append(sphere)
    except Exception:
        pass
    
    # Perturbed FCC
    try:
        fcc_pert = _fcc_cluster(size, spacing=1.0) + rng.normal(0, 0.2, (size, 3))
        initial_structures.append(fcc_pert)
    except Exception:
        pass
    
    # Perturbed random
    try:
        random_pert = _random_compact(size, rng, radius=1.0 * size ** (1.0 / 3.0))
        initial_structures.append(random_pert)
    except Exception:
        pass
    
    best_energy = np.inf
    best_positions = None
    
    # Try each initial structure with basin hopping
    for init_pos in initial_structures:
        if time.time() > _DEADLINE - 10:
            break
        
        if len(init_pos) != size:
            continue
        
        # First relax the initial structure
        try:
            energy, relaxed = relax(init_pos, 'sutton_chen', kind)
        except Exception:
            continue
        
        if not _valid(relaxed):
            continue
        
        if energy < best_energy:
            best_energy = energy
            best_positions = relaxed.copy()
        
        # Basin hopping from this relaxed structure
        try:
            bh_energy, bh_pos = _basin_hopping(
                relaxed, kind, rng, 
                max_iter=30, step_size=0.4, 
                time_budget=min(20.0, (_DEADLINE - time.time()) * 0.3)
            )
        except Exception:
            continue
        
        if _valid(bh_pos) and bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_pos.copy()
    
    # If we haven't found a valid solution, try a simple approach
    if best_positions is None or not _valid(best_positions):
        # Generate a simple compact arrangement
        radius = 0.7 * size ** (1.0 / 3.0)
        positions = rng.uniform(-radius, radius, (size, 3))
        try:
            energy, relaxed = relax(positions, 'sutton_chen', kind)
            if _valid(relaxed):
                best_energy = energy
                best_positions = relaxed
        except Exception:
            pass
    
    # Final safety check
    if best_positions is not None and len(best_positions) == size:
        if not _valid(best_positions):
            # Try to fix by spreading out
            for _ in range(20):
                min_d = _min_dist(best_positions)
                if min_d >= 0.35:
                    break
                for i in range(size):
                    for j in range(i + 1, size):
                        d = np.linalg.norm(best_positions[i] - best_positions[j])
                        if d < 0.35:
                            direction = best_positions[i] - best_positions[j]
                            if np.linalg.norm(direction) < 1e-10:
                                direction = rng.normal(0, 1, 3)
                            direction = direction / np.linalg.norm(direction)
                            best_positions[i] += 0.2 * direction
                            best_positions[j] -= 0.2 * direction
    
    # Ensure we return exactly size positions
    if best_positions is not None:
        if len(best_positions) > size:
            best_positions = best_positions[:size]
        elif len(best_positions) < size:
            # Pad with random positions far from cluster
            center = np.mean(best_positions, axis=0)
            max_dist = np.max(np.linalg.norm(best_positions - center, axis=1))
            while len(best_positions) < size:
                new_pos = center + (max_dist + 0.5) * rng.normal(0, 1, 3)
                best_positions = np.vstack([best_positions, new_pos])
    
    return best_positions

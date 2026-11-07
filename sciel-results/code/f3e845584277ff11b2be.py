"""Sutton-Chen transition-metal cluster global minima - growth + multi-start + basin hopping."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Global deadline: 320s (10s safety margin before 330s kill)
_DEADLINE = time.time() + 320.0


def _energy_grad(pos, kind):
    """Compute energy and gradient."""
    try:
        return sutton_chen_energy_gradient(pos, kind)
    except Exception:
        return np.inf, np.zeros_like(pos)


def _min_dist(pos):
    """Compute minimum pairwise distance."""
    if len(pos) < 2:
        return np.inf
    return pdist(pos, 'euclidean').min()


def _valid(pos, min_sep=0.35):
    """Check if positions are valid (finite, min distance > min_sep)."""
    if not np.all(np.isfinite(pos)):
        return False
    if len(pos) < 2:
        return True
    return _min_dist(pos) > min_sep


def _fcc_cluster(n, spacing=1.0):
    """Generate n positions on an FCC lattice, packed compactly."""
    a = spacing * np.sqrt(2.0)
    max_coord = int(np.ceil((n * 4) ** (1.0/3.0))) + 3
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * a / 2.0
                    points.append(p)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    return points[order[:n]]


def _hcp_cluster(n, spacing=1.0):
    """Generate n positions on an HCP lattice, packed compactly."""
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    points = []
    max_layer = int(np.ceil((n / 3) ** 0.5)) + 3
    for layer in range(-max_layer, max_layer + 1):
        z = layer * c
        for i in range(-max_layer, max_layer + 1):
            for j in range(-max_layer, max_layer + 1):
                if layer % 2 == 0:
                    x = (i + 0.5 * (j % 2)) * a
                    y = j * a * np.sqrt(3.0) / 2.0
                else:
                    x = i * a
                    y = (j + 0.5 * (i % 2)) * a * np.sqrt(3.0) / 2.0
                p = np.array([x, y, z])
                points.append(p)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    return points[order[:n]]


def _mackay_cluster(n, spacing=1.0):
    """Generate Mackay icosahedral cluster positions."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    verts = verts / np.linalg.norm(verts[0]) * spacing
    
    points = [np.zeros(3)]
    
    # Shell 1: 12 vertices
    if n > 1:
        for v in verts:
            points.append(v.copy())
    
    # Shell 2: 42 positions
    if n > 13:
        r2 = spacing * np.sqrt(3.0)
        oct_verts = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=float) * r2
        for v in oct_verts:
            points.append(v.copy())
        
        # Add rotated icosahedron vertices
        for angle in [np.pi/5, 2*np.pi/5, 3*np.pi/5]:
            R = np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
            rotated = verts @ R.T * (r2 / spacing)
            for v in rotated:
                points.append(v.copy())
                if len(points) >= min(n, 55):
                    break
            if len(points) >= min(n, 55):
                break
    
    # Fill remaining with golden ratio spiral on sphere
    while len(points) < n:
        i = len(points)
        r = spacing * (i + 1) ** (1.0/3.0) * 1.0
        golden = (1 + np.sqrt(5)) / 2
        theta = 2 * np.pi * i / golden
        phi = np.arccos(1 - 2 * (i + 0.5) / max(n, 1))
        p = r * np.array([np.sin(phi) * np.cos(theta),
                          np.sin(phi) * np.sin(theta),
                          np.cos(phi)])
        points.append(p)
    
    return np.array(points[:n])


def _random_compact(n, rng, spacing=1.0):
    """Generate n random positions in a compact sphere."""
    radius = spacing * n ** (1.0/3.0) * 0.9
    points = []
    attempts = 0
    min_d = 0.7 * spacing
    while len(points) < n and attempts < n * 200:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if len(points) == 0 or np.min(np.linalg.norm(np.array(points) - p, axis=1)) >= min_d:
                points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _growth_cluster(n, kind, spacing=1.0, rng=None, time_budget=60.0):
    """Build cluster by growing from small relaxed cores."""
    if rng is None:
        rng = np.random.default_rng(42)
    
    start_time = time.time()
    
    if n <= 3:
        if n == 1:
            return np.zeros((1, 3))
        elif n == 2:
            return np.array([[0, 0, 0], [spacing, 0, 0]])
        else:
            return np.array([[0, 0, 0], [spacing, 0, 0], [spacing/2, spacing*np.sqrt(3)/2, 0]])
    
    # Start with 3-atom triangle
    current = np.array([[0, 0, 0], [spacing, 0, 0], [spacing/2, spacing*np.sqrt(3)/2, 0]])
    try:
        current, _ = relax(current, 'sutton_chen', kind)
    except Exception:
        pass
    
    for target in range(4, n + 1):
        if time.time() - start_time > time_budget:
            break
        if len(current) >= target:
            break
        
        best_pos = None
        best_e = np.inf
        
        # Try positions around existing atoms
        for atom in current:
            for r in [spacing * 0.9, spacing, spacing * 1.1, spacing * 1.2]:
                for theta in np.linspace(0, 2*np.pi, 12, endpoint=False):
                    for phi in [0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]:
                        new_p = atom + r * np.array([
                            np.sin(phi) * np.cos(theta),
                            np.sin(phi) * np.sin(theta),
                            np.cos(phi)
                        ])
                        trial = np.vstack([current, new_p])
                        if _valid(trial):
                            e, _ = _energy_grad(trial, kind)
                            if e < best_e:
                                best_e = e
                                best_pos = new_p
        
        # Also try random positions
        radius = spacing * target ** (1.0/3.0) * 1.2
        for _ in range(30):
            new_p = rng.uniform(-radius, radius, 3)
            trial = np.vstack([current, new_p])
            if _valid(trial):
                e, _ = _energy_grad(trial, kind)
                if e < best_e:
                    best_e = e
                    best_pos = new_p
        
        if best_pos is not None:
            current = np.vstack([current, best_pos])
            try:
                current, _ = relax(current, 'sutton_chen', kind)
            except Exception:
                pass
        
        if time.time() - start_time > time_budget:
            break
    
    # If growth didn't reach n, pad with random positions
    while len(current) < n:
        radius = spacing * n ** (1.0/3.0) * 1.0
        new_p = rng.uniform(-radius, radius, 3)
        current = np.vstack([current, new_p])
    
    return current[:n]


def _de_step(positions, rng, step_size=0.5):
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


def _basin_hopping(positions, kind, rng, max_iter=40, step_size=0.4, time_budget=30.0):
    """Basin hopping optimization."""
    current_pos = positions.copy()
    current_energy, _ = _energy_grad(current_pos, kind)
    
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    start_time = time.time()
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break
        if time.time() - start_time > time_budget:
            break
        
        perturbed = _de_step(current_pos, rng, step_size)
        
        try:
            new_energy, new_pos = relax(perturbed, 'sutton_chen', kind)
        except Exception:
            continue
        
        if not _valid(new_pos):
            continue
        
        if new_energy < current_energy:
            current_pos = new_pos
            current_energy = new_energy
        else:
            delta_e = new_energy - current_energy
            if delta_e < 0 or rng.random() < np.exp(-delta_e / 0.1):
                current_pos = new_pos
                current_energy = new_energy
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
    
    return best_energy, best_pos


def construct(instance):
    """
    Multi-start with growth + basin hopping for Sutton-Chen clusters.
    """
    size = instance["n"]
    kind = instance.get("kind", "12-6")
    
    rng = np.random.default_rng(size * 7919 + 12345)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse initial structures
    initial_structures = []
    
    # 1. Growth algorithm (deterministic, exploits structural hierarchy)
    try:
        growth_time = min(40.0, (_DEADLINE - time.time()) * 0.25)
        grown = _growth_cluster(size, kind, spacing=1.0, rng=rng, time_budget=growth_time)
        if len(grown) == size and _valid(grown):
            initial_structures.append(grown)
    except Exception:
        pass
    
    # 2. FCC
    try:
        fcc = _fcc_cluster(size, spacing=1.0)
        if len(fcc) >= size:
            initial_structures.append(fcc[:size])
    except Exception:
        pass
    
    # 3. HCP
    try:
        hcp = _hcp_cluster(size, spacing=1.0)
        if len(hcp) >= size:
            initial_structures.append(hcp[:size])
    except Exception:
        pass
    
    # 4. Mackay icosahedral
    try:
        mackay = _mackay_cluster(size, spacing=1.0)
        if len(mackay) >= size:
            initial_structures.append(mackay[:size])
    except Exception:
        pass
    
    # 5. Random compact
    try:
        random_compact = _random_compact(size, rng, spacing=1.0)
        initial_structures.append(random_compact)
    except Exception:
        pass
    
    # 6. Perturbed FCC
    try:
        fcc_pert = _fcc_cluster(size, spacing=1.0) + rng.normal(0, 0.2, (size, 3))
        initial_structures.append(fcc_pert)
    except Exception:
        pass
    
    # Try each initial structure
    for init_pos in initial_structures:
        if time.time() > _DEADLINE - 15:
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
        remaining = _DEADLINE - time.time() - 15
        if remaining > 20:
            try:
                bh_energy, bh_pos = _basin_hopping(
                    relaxed, kind, rng,
                    max_iter=25, step_size=0.4,
                    time_budget=min(25.0, remaining * 0.4)
                )
            except Exception:
                continue
            
            if _valid(bh_pos) and bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_pos.copy()
    
    # If we haven't found a valid solution, try a simple approach
    if best_positions is None or not _valid(best_positions):
        radius = 0.7 * size ** (1.0/3.0)
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
            for _ in range(30):
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
            center = np.mean(best_positions, axis=0)
            max_dist = np.max(np.linalg.norm(best_positions - center, axis=1))
            while len(best_positions) < size:
                new_pos = center + (max_dist + 0.5) * rng.normal(0, 1, 3)
                best_positions = np.vstack([best_positions, new_pos])
    
    return best_positions

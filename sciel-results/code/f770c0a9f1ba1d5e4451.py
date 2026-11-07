"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Time management
_START_TIME = time.time()
_TOTAL_TIME_LIMIT = 300.0  # seconds
_SAFETY_MARGIN = 10.0  # seconds

def _time_left():
    return _TOTAL_TIME_LIMIT - (time.time() - _START_TIME) - _SAFETY_MARGIN

# --- Structural initialization helpers ---

def _bcc_positions(n, spacing=1.0):
    """Generate n positions in a BCC-like arrangement."""
    if n == 0:
        return np.zeros((0, 3))
    
    # BCC unit cell: corners + body center
    # For a cube of side L, BCC has 8 corners + 1 center = 9 atoms per cell
    # But we want compact clusters, so let's build up from a central atom outward
    
    positions = [np.array([0.0, 0.0, 0.0])]
    if n == 1:
        return np.array(positions)
    
    # Add neighbors in order of distance from center
    # BCC neighbors: 8 at distance sqrt(3)*a/2, 6 at distance a, etc.
    # Let's use a simpler approach: generate points in a cubic grid with BCC offset
    
    a = spacing
    # Generate candidate points in a cube
    max_coord = int(np.ceil((n / 4) ** (1/3))) + 2  # rough estimate
    candidates = []
    
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                # Corner of BCC cell
                candidates.append(np.array([ix * a, iy * a, iz * a]))
                # Body center of BCC cell
                candidates.append(np.array([(ix + 0.5) * a, (iy + 0.5) * a, (iz + 0.5) * a]))
    
    # Sort by distance from origin
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    candidates = candidates[order]
    
    # Greedily select points with minimum distance constraint
    selected = [candidates[0]]
    min_dist = 0.5 * a
    for p in candidates[1:]:
        if len(selected) >= n:
            break
        if all(np.linalg.norm(p - s) >= min_dist for s in selected):
            selected.append(p)
    
    # If we didn't get enough, just take the closest n
    if len(selected) < n:
        selected = [candidates[i] for i in range(n)]
    
    return np.array(selected[:n])


def _fcc_positions(n, spacing=1.0):
    """Generate n positions in an FCC-like arrangement."""
    if n == 0:
        return np.zeros((0, 3))
    
    a = spacing
    max_coord = int(np.ceil((n / 2) ** (1/3))) + 2
    candidates = []
    
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                # FCC: corners + face centers
                candidates.append(np.array([ix * a, iy * a, iz * a]))
                candidates.append(np.array([(ix + 0.5) * a, (iy + 0.5) * a, iz * a]))
                candidates.append(np.array([(ix + 0.5) * a, iy * a, (iz + 0.5) * a]))
                candidates.append(np.array([ix * a, (iy + 0.5) * a, (iz + 0.5) * a]))
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    candidates = candidates[order]
    
    selected = [candidates[0]]
    min_dist = 0.5 * a
    for p in candidates[1:]:
        if len(selected) >= n:
            break
        if all(np.linalg.norm(p - s) >= min_dist for s in selected):
            selected.append(p)
    
    if len(selected) < n:
        selected = [candidates[i] for i in range(n)]
    
    return np.array(selected[:n])


def _random_compact(n, spacing=0.72, rng=None):
    """Generate a random compact cluster."""
    if rng is None:
        rng = np.random.default_rng()
    
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    max_attempts = n * 500
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.4 * spacing for q in points):
                points.append(p)
        attempts += 1
    
    # Fill remaining with random positions if needed
    while len(points) < n:
        points.append(rng.uniform(-radius, radius, 3))
    
    return np.array(points)


def _icosahedron(n, spacing=0.72):
    """Generate an icosahedral cluster."""
    if n <= 1:
        return np.zeros((n, 3))
    
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * spacing
    
    positions = [np.array([0.0, 0.0, 0.0])]
    if n <= 1:
        return np.array(positions)
    
    # Add icosahedron vertices
    for v in verts:
        if len(positions) >= n:
            break
        positions.append(v)
    
    # Add second shell if needed
    if len(positions) < n:
        # Octahedron vertices for second shell
        octa = np.array([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
        ])
        octa = octa / np.linalg.norm(octa, axis=1, keepdims=True) * spacing * 1.6
        for v in octa:
            if len(positions) >= n:
                break
            positions.append(v)
    
    # Fill with random if still needed
    while len(positions) < n:
        p = np.random.uniform(-spacing * 2, spacing * 2, 3)
        if all(np.linalg.norm(p - q) >= 0.4 * spacing for q in positions):
            positions.append(p)
    
    return np.array(positions[:n])


def _octahedron(n, spacing=0.72):
    """Generate an octahedral cluster."""
    if n <= 1:
        return np.zeros((n, 3))
    
    octa = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ])
    octa = octa / np.linalg.norm(octa, axis=1, keepdims=True) * spacing
    
    positions = [np.array([0.0, 0.0, 0.0])]
    for v in octa:
        if len(positions) >= n:
            break
        positions.append(v)
    
    # Second shell: cube vertices
    if len(positions) < n:
        cube = np.array(list(product([-1, 1], repeat=3)))
        cube = cube / np.linalg.norm(cube, axis=1, keepdims=True) * spacing * 1.4
        for v in cube:
            if len(positions) >= n:
                break
            positions.append(v)
    
    while len(positions) < n:
        p = np.random.uniform(-spacing * 2, spacing * 2, 3)
        if all(np.linalg.norm(p - q) >= 0.4 * spacing for q in positions):
            positions.append(p)
    
    return np.array(positions[:n])


def _cuboctahedron(n, spacing=0.72):
    """Generate a cuboctahedral cluster."""
    if n <= 1:
        return np.zeros((n, 3))
    
    # Cuboctahedron: 12 vertices
    verts = []
    for perm in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
        for sign in [1, -1]:
            v = np.zeros(3)
            v[perm[0]] = sign * 1
            v[perm[1]] = sign * 1
            v[perm[2]] = 0
            verts.append(v)
    # Remove duplicates
    verts = np.array(verts)
    unique_verts = []
    for v in verts:
        if not any(np.allclose(v, u) for u in unique_verts):
            unique_verts.append(v)
    verts = np.array(unique_verts)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * spacing
    
    positions = [np.array([0.0, 0.0, 0.0])]
    for v in verts:
        if len(positions) >= n:
            break
        positions.append(v)
    
    while len(positions) < n:
        p = np.random.uniform(-spacing * 2, spacing * 2, 3)
        if all(np.linalg.norm(p - q) >= 0.4 * spacing for q in positions):
            positions.append(p)
    
    return np.array(positions[:n])


def _generate_starts(n, kind, rng):
    """Generate multiple diverse starting configurations."""
    spacing = 0.72
    starts = []
    
    # BCC-based
    try:
        bcc = _bcc_positions(n, spacing)
        if len(bcc) == n and np.all(np.isfinite(bcc)):
            starts.append(bcc)
    except:
        pass
    
    # FCC-based
    try:
        fcc = _fcc_positions(n, spacing)
        if len(fcc) == n and np.all(np.isfinite(fcc)):
            starts.append(fcc)
    except:
        pass
    
    # Icosahedral
    try:
        icosa = _icosahedron(n, spacing)
        if len(icosa) == n and np.all(np.isfinite(icosa)):
            starts.append(icosa)
    except:
        pass
    
    # Octahedral
    try:
        octa = _octahedron(n, spacing)
        if len(octa) == n and np.all(np.isfinite(octa)):
            starts.append(octa)
    except:
        pass
    
    # Cuboctahedral
    try:
        cubocta = _cuboctahedron(n, spacing)
        if len(cubocta) == n and np.all(np.isfinite(cubocta)):
            starts.append(cubocta)
    except:
        pass
    
    # Random compact (multiple)
    for _ in range(4):
        try:
            rand = _random_compact(n, spacing, rng)
            if len(rand) == n and np.all(np.isfinite(rand)):
                starts.append(rand)
        except:
            pass
    
    # Jittered versions of BCC
    if len(starts) > 0:
        for _ in range(2):
            jittered = starts[0] + rng.normal(0, 0.05, (n, 3))
            if np.all(np.isfinite(jittered)):
                starts.append(jittered)
    
    return starts


def _validate(positions):
    """Check if positions are valid."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(positions[i] - positions[j]) < 0.35:
                return False
    return True


def construct(instance):
    """Find near-global minimum for Sutton-Chen transition-metal clusters."""
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    rng = np.random.default_rng(n * 12345 + 67890)
    
    # Generate diverse starting configurations
    starts = _generate_starts(n, kind, rng)
    
    if not starts:
        # Fallback: single random start
        starts = [_random_compact(n, 0.72, rng)]
    
    best_energy = np.inf
    best_positions = starts[0]
    
    # Try each start with relaxation
    for i, start in enumerate(starts):
        if _time_left() < 5.0:
            break
        
        if not _validate(start):
            continue
        
        try:
            energy, positions = relax(start, 'sutton_chen', kind)
            
            if not np.all(np.isfinite(positions)):
                continue
            
            if not _validate(positions):
                continue
            
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception as e:
            continue
    
    # If we have time, try a few more random starts
    attempts = 0
    max_attempts = 20
    while _time_left() > 10.0 and attempts < max_attempts:
        attempts += 1
        try:
            start = _random_compact(n, 0.72, rng)
            if not _validate(start):
                continue
            
            energy, positions = relax(start, 'sutton_chen', kind)
            
            if not np.all(np.isfinite(positions)):
                continue
            
            if not _validate(positions):
                continue
            
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except:
            continue
    
    # Final validation
    if not _validate(best_positions):
        # Return a simple valid configuration
        best_positions = _random_compact(n, 0.72, rng)
        # Ensure minimum distance
        for i in range(len(best_positions)):
            for j in range(i + 1, len(best_positions)):
                d = np.linalg.norm(best_positions[i] - best_positions[j])
                if d < 0.35:
                    direction = best_positions[i] - best_positions[j]
                    if np.linalg.norm(direction) > 1e-10:
                        direction = direction / np.linalg.norm(direction)
                    else:
                        direction = np.array([1.0, 0, 0])
                    best_positions[j] = best_positions[i] + direction * 0.36
    
    return best_positions
# EVOLVE-BLOCK-END

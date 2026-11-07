import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _fcc_positions(n):
    """Generate n points from an FCC lattice closest to the origin."""
    max_coord = int(np.ceil((n ** (1/3)) * 2.5)) + 5
    pts = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    pts.append([0.5 * i, 0.5 * j, 0.5 * k])
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)
    return pts[idx[:n]]

def _hcp_positions(n):
    """Generate n points from an HCP lattice closest to the origin."""
    a = 1.0
    c = a * np.sqrt(8.0/3.0)
    pts = []
    max_coord = int(np.ceil((n ** (1/3)) * 2.5)) + 5
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            x = i + 0.5 * j
            y = np.sqrt(3.0)/2.0 * j
            pts.append([x, y, 0.0])
            x_shift = x + a/3.0
            y_shift = y + np.sqrt(3.0)*a/6.0
            pts.append([x_shift, y_shift, c/2.0])
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)
    return pts[idx[:n]]

def _bcc_positions(n):
    """Generate n points from a BCC lattice closest to the origin."""
    max_coord = int(np.ceil((n ** (1/3)) * 2.5)) + 5
    pts = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                pts.append([0.5*i, 0.5*j, 0.5*k])
                if (i + j + k) % 2 == 0:
                    pts.append([0.5*i + 0.5, 0.5*j + 0.5, 0.5*k + 0.5])
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)
    return pts[idx[:n]]

def _icosahedron_positions(n):
    """Generate n points in icosahedral shells."""
    # icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
        [1, phi, 0], [-1, phi, 0]
    ])
    # Normalize to unit sphere
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    pts = [np.zeros(3)]  # center
    # Shell 1: 12 vertices
    pts.extend(verts.tolist())
    # Shell 2: 30 vertices (edge midpoints scaled)
    # For simplicity, use more icosahedron-like points
    # Actually, let's just use concentric icosahedra
    shell2 = verts * 2.0
    pts.extend(shell2.tolist())
    # Shell 3
    shell3 = verts * 3.0
    pts.extend(shell3.tolist())
    
    pts = np.array(pts)
    if len(pts) < n:
        # Fill with more points
        extra = verts * np.linspace(1.5, 4.0, n - len(pts))
        pts = np.vstack([pts, extra])
    return pts[:n]

def _random_compact(n, seed=42, min_dist=0.8):
    """Generate n random points in a compact region."""
    rng = np.random.default_rng(seed)
    radius = (n / (4/3 * np.pi)) ** (1/3) * 1.2
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 500:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    if len(points) < n:
        # Fallback: just place remaining randomly
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)

def _small_cluster_template(n, seed=42):
    """Use known small cluster structures as templates."""
    # For small n, use specific structures
    if n <= 13:
        # Icosahedron
        return _icosahedron_positions(n)
    elif n <= 55:
        # Mackay icosahedron
        return _icosahedron_positions(n)
    else:
        return _fcc_positions(n)

def _ensure_valid(pos, min_dist=0.35):
    """Ensure no two atoms are closer than min_dist."""
    n = len(pos)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < min_dist:
                # Push apart
                if d < 1e-10:
                    pos[j] += np.array([0.01, 0.01, 0.01])
                else:
                    direction = (pos[j] - pos[i]) / d
                    pos[j] = pos[i] + direction * min_dist
    return pos

def construct(instance):
    """
    Construct a low-energy cluster configuration.
    """
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    
    start_time = time.time()
    deadline = 280.0  # Safety margin within 300s
    
    # Generate diverse candidates
    candidates = []
    
    # Structured initializations
    try:
        candidates.append(_fcc_positions(n))
    except Exception:
        pass
    try:
        candidates.append(_hcp_positions(n))
    except Exception:
        pass
    try:
        candidates.append(_bcc_positions(n))
    except Exception:
        pass
    try:
        candidates.append(_icosahedron_positions(n))
    except Exception:
        pass
    try:
        candidates.append(_small_cluster_template(n))
    except Exception:
        pass
    
    # Random compact starts with different seeds
    for seed in [1, 2, 3, 4, 5]:
        try:
            candidates.append(_random_compact(n, seed=seed))
        except Exception:
            pass
    
    # Ensure all candidates are valid
    valid_candidates = []
    for pos in candidates:
        try:
            pos = np.array(pos, dtype=float)
            pos = _ensure_valid(pos)
            if np.all(np.isfinite(pos)):
                valid_candidates.append(pos)
        except Exception:
            continue
    
    if not valid_candidates:
        # Fallback
        rng = np.random.default_rng(0)
        pos = rng.uniform(-2, 2, (n, 3))
        pos = _ensure_valid(pos)
        valid_candidates.append(pos)
    
    # Relax each candidate and keep the best
    best_energy = np.inf
    best_positions = None
    
    for pos in valid_candidates:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        try:
            energy, relaxed_pos = relax(pos, "sutton_chen", kind)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos
        except Exception:
            continue
    
    if best_positions is None:
        # Last resort: use first valid candidate
        best_positions = valid_candidates[0]
    
    return best_positions

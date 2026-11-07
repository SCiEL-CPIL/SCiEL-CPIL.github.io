import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _fcc_positions(n):
    """Generate n points from an FCC lattice closest to the origin."""
    # FCC basis vectors
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    c = np.array([0.0, 0.0, 1.0])
    # Generate grid of FCC points
    max_coord = int(np.ceil((n ** (1/3)) * 2.0)) + 5
    pts = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC: points at (i,j,k) where i+j+k is even, plus basis offsets
                base = np.array([i, j, k]) * 0.5
                # FCC has 4 atoms per unit cell
                offsets = [
                    np.array([0.0, 0.0, 0.0]),
                    np.array([0.5, 0.5, 0.0]),
                    np.array([0.5, 0.0, 0.5]),
                    np.array([0.0, 0.5, 0.5]),
                ]
                for off in offsets:
                    pts.append(base + off)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)
    return pts[idx[:n]]

def _hcp_positions(n):
    """Generate n points from an HCP lattice closest to the origin."""
    a = 1.0
    c = a * np.sqrt(8.0/3.0)
    pts = []
    max_coord = int(np.ceil((n ** (1/3)) * 2.0)) + 5
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
    max_coord = int(np.ceil((n ** (1/3)) * 2.0)) + 5
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
    """Generate n points in icosahedral shells (Mackay icosahedron)."""
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    pts = [np.zeros(3)]  # center
    
    # Shell 1: 12 vertices
    shell1 = verts * 1.0
    pts.extend(shell1.tolist())
    
    # Shell 2: 30 points (edge midpoints of icosahedron)
    # Generate all edges
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 1.2:  # adjacent vertices
                edges.append((i, j))
    
    shell2 = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid) * 1.732  # scale to proper shell
        shell2.append(mid)
    pts.extend(shell2)
    
    # Shell 3: 20 points (face centers)
    # Find triangular faces
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if d1 < 1.2 and d2 < 1.2 and d3 < 1.2:
                    faces.append((i, j, k))
    
    shell3 = []
    for i, j, k in faces:
        centroid = (verts[i] + verts[j] + verts[k]) / 3.0
        centroid = centroid / np.linalg.norm(centroid) * 2.0
        shell3.append(centroid)
    pts.extend(shell3)
    
    # Shell 4: 60 more points if needed
    if len(pts) < n:
        # Use more icosahedral symmetry points
        shell4 = []
        for v in verts:
            for scale in [2.5, 3.0]:
                shell4.append(v * scale)
        pts.extend(shell4)
    
    pts = np.array(pts)
    if len(pts) < n:
        # Fill with more points
        extra = verts * np.linspace(1.5, 5.0, n - len(pts))
        pts = np.vstack([pts, extra])
    return pts[:n]

def _random_compact(n, seed=42, min_dist=0.7):
    """Generate n random points in a compact region."""
    rng = np.random.default_rng(seed)
    radius = (n / (4/3 * np.pi)) ** (1/3) * 1.1
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 300:
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

def _ensure_valid(pos, min_dist=0.35):
    """Ensure no two atoms are closer than min_dist."""
    pos = pos.copy()
    n = len(pos)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < min_dist:
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
    deadline = 320.0  # Safety margin within 330s limit
    
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
    
    # Random compact starts with different seeds
    for seed in [1, 2, 3, 4, 5, 6, 7]:
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

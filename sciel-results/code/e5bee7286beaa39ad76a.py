"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _validate_positions(pos):
    """Check if positions are valid (finite, min distance > 0.35)."""
    if not np.all(np.isfinite(pos)):
        return False
    n = len(pos)
    if n < 2:
        return True
    # Vectorized min-pair-distance check
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) > 0.35

def _generate_fcc_cluster(n, lattice=1.0):
    """Generate a compact FCC-like cluster of n atoms."""
    fcc_offsets = np.array([
        [0, 0, 0],
        [lattice/2, lattice/2, 0],
        [lattice/2, 0, lattice/2],
        [0, lattice/2, lattice/2],
    ])
    size = int(np.ceil(n ** (1.0/3.0))) + 2
    all_points = []
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            for k in range(-size, size + 1):
                base = np.array([i * lattice, j * lattice, k * lattice])
                for off in fcc_offsets:
                    all_points.append(base + off)
    all_points = np.array(all_points)
    dists = np.linalg.norm(all_points, axis=1)
    max_r = 0.9 * n ** (1.0/3.0) + 1.0
    mask = dists <= max_r
    pts = all_points[mask]
    d2 = np.linalg.norm(pts, axis=1)
    order = np.argsort(d2)
    if len(pts) >= n:
        return pts[order[:n]].copy()
    else:
        # Fallback: random
        rng = np.random.default_rng(42)
        pts = rng.uniform(-max_r, max_r, size=(n, 3))
        return pts

def _generate_hcp_cluster(n, lattice=1.0):
    """Generate a compact HCP-like cluster of n atoms."""
    a = lattice
    c = a * np.sqrt(8.0 / 3.0)
    points = []
    size = int(np.ceil(n ** (1.0/3.0))) + 2
    for k in range(-size, size + 1):
        z = k * c / 2
        layer = k % 2
        for i in range(-size, size + 1):
            for j in range(-size, size + 1):
                if layer == 0:
                    x = i * a + j * a / 2
                    y = j * a * np.sqrt(3) / 2
                else:
                    x = i * a + j * a / 2 + a / 2
                    y = j * a * np.sqrt(3) / 2 + a * np.sqrt(3) / 6
                points.append([x, y, z])
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    max_r = 0.9 * n ** (1.0/3.0) + 1.0
    mask = dists <= max_r
    pts = points[mask]
    d2 = np.linalg.norm(pts, axis=1)
    order = np.argsort(d2)
    if len(pts) >= n:
        return pts[order[:n]].copy()
    else:
        rng = np.random.default_rng(43)
        pts = rng.uniform(-max_r, max_r, size=(n, 3))
        return pts

def _generate_icosahedral_cluster(n, scale=1.0):
    """
    Generate a Mackay icosahedron cluster.
    This is a strong candidate for clusters with icosahedral symmetry (n=13, 55, 147...).
    For other n, it fills shells in order of increasing radius.
    """
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of a regular icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ])
    verts = verts / np.linalg.norm(verts[0])
    
    points = [np.zeros(3)]  # Center
    
    # Shell 1: 12 vertices
    for v in verts:
        points.append(v * 1.0)
        
    # Shell 2: 30 edge midpoints (radius ~ 1.35)
    shell2 = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if 0.9 < d < 1.1: # Adjacent vertices
                mid = (verts[i] + verts[j]) / 2
                mid = mid / np.linalg.norm(mid) * 1.35
                shell2.append(mid)
    for p in shell2:
        points.append(p)
        
    # Shell 3: 20 face centers (radius ~ 1.7)
    shell3 = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if all(0.9 < d < 1.1 for d in [d1, d2, d3]):
                    fc = (verts[i] + verts[j] + verts[k]) / 3
                    fc = fc / np.linalg.norm(fc) * 1.7
                    shell3.append(fc)
    for p in shell3:
        points.append(p)
        
    # If we need more atoms, we can add a 4th shell (vertices of a larger icosahedron)
    # or just stop. For n up to 80, 1+12+30+20 = 63 atoms. 
    # For n > 63, we need a 4th shell.
    if n > 63:
        # 4th shell: 30 edge midpoints of the 2nd shell? Or just scale up vertices?
        # Let's add 30 points at radius ~2.1 (midpoints of shell 2 edges)
        # This is getting complex. A simpler fallback for n > 63 is to add random points 
        # in a shell or use a different structure. 
        # However, for the specific instances (38, 55, 59, 75, 79, 80), 
        # 55 is perfect. 38 is 1+12+25 (partial shell 2).
        # 59 is 1+12+30+16 (partial shell 3).
        # 75, 79, 80 need more.
        # Let's add a 4th shell of 60 points (midpoints of faces of shell 2?) 
        # Actually, let's just generate points on a sphere of radius 2.1 for the rest.
        rng = np.random.default_rng(123)
        while len(points) < n:
            # Random point on sphere
            z = rng.uniform(-1, 1)
            theta = rng.uniform(0, 2*np.pi)
            x = np.sqrt(1 - z**2) * np.cos(theta)
            y = np.sqrt(1 - z**2) * np.sin(theta)
            points.append(np.array([x, y, z]) * 2.1)
            
    points = np.array(points)
    # Scale to appropriate size. The ideal bond length is ~1.0.
    # The current scale has bond length 1.0.
    # But the cluster might be too big or small for the potential.
    # We will relax it, so exact scale isn't critical, but it helps.
    return points[:n].copy()

def _generate_deformed_icosahedron(n, scale=1.0):
    """
    Generate a deformed icosahedron.
    For many clusters (e.g., n=38, 59), the global minimum is a deformed icosahedron
    or a fragment of a Mackay icosahedron.
    We start with a Mackay icosahedron and slightly deform it.
    """
    pos = _generate_icosahedral_cluster(n, scale)
    # Add small noise to break symmetry slightly, allowing relaxation to find the minimum
    rng = np.random.default_rng(999)
    pos += rng.normal(0, 0.05, size=pos.shape)
    return pos

def _generate_bcc_cluster(n, lattice=1.0):
    """Generate a compact BCC-like cluster."""
    bcc_offsets = np.array([
        [0, 0, 0],
        [lattice/2, lattice/2, lattice/2],
    ])
    size = int(np.ceil(n ** (1.0/3.0))) + 2
    all_points = []
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            for k in range(-size, size + 1):
                base = np.array([i * lattice, j * lattice, k * lattice])
                for off in bcc_offsets:
                    all_points.append(base + off)
    all_points = np.array(all_points)
    dists = np.linalg.norm(all_points, axis=1)
    max_r = 0.9 * n ** (1.0/3.0) + 1.0
    mask = dists <= max_r
    pts = all_points[mask]
    d2 = np.linalg.norm(pts, axis=1)
    order = np.argsort(d2)
    if len(pts) >= n:
        return pts[order[:n]].copy()
    else:
        rng = np.random.default_rng(45)
        pts = rng.uniform(-max_r, max_r, size=(n, 3))
        return pts

def construct(instance):
    """
    Deterministic structure-based global search.
    Generates a library of physically-motivated prototypes, relaxes each,
    and returns the one with the lowest energy.
    """
    start_time = time.time()
    # 300s total for 6 instances. 50s per instance is safe.
    deadline = start_time + 45.0
    
    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    
    # Define a library of prototypes
    # We try multiple scales for each structure type to ensure we cover the right size
    prototypes = []
    
    # 1. FCC
    for lat in [0.7, 0.8, 0.9, 1.0]:
        prototypes.append(_generate_fcc_cluster(n, lattice=lat))
        
    # 2. HCP
    for lat in [0.7, 0.8, 0.9, 1.0]:
        prototypes.append(_generate_hcp_cluster(n, lattice=lat))
        
    # 3. Icosahedral
    for scale in [0.9, 1.0, 1.1]:
        prototypes.append(_generate_icosahedral_cluster(n, scale=scale))
        
    # 4. Deformed Icosahedral
    for scale in [0.9, 1.0, 1.1]:
        prototypes.append(_generate_deformed_icosahedron(n, scale=scale))
        
    # 5. BCC
    for lat in [0.7, 0.8, 0.9, 1.0]:
        prototypes.append(_generate_bcc_cluster(n, lattice=lat))
        
    # 6. Random compact (fallback)
    rng = np.random.default_rng(12345 + n)
    max_r = 0.9 * n ** (1.0/3.0) + 1.0
    prototypes.append(rng.uniform(-max_r, max_r, size=(n, 3)))
    
    best_energy = np.inf
    best_pos = None
    
    for pos in prototypes:
        if time.time() > deadline:
            break
            
        # Validate
        if not _validate_positions(pos):
            continue
            
        # Relax
        try:
            energy, relaxed = relax(pos, potential, kind)
        except Exception:
            continue
            
        if np.isfinite(energy) and _validate_positions(relaxed):
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed.copy()
                
    if best_pos is None:
        # Fallback: return the first valid prototype
        for pos in prototypes:
            if _validate_positions(pos):
                return pos
        # Last resort
        return np.zeros((n, 3))
        
    return best_pos
# EVOLVE-BLOCK-END

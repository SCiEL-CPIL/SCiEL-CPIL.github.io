"""Improved program for Sutton-Chen cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _generate_fcc_cluster(n):
    """Generate a compact FCC-like cluster by taking the n closest points to origin in FCC lattice."""
    # FCC lattice: conventional cell with basis at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    # We generate points on the FCC lattice and pick the n closest to origin
    max_idx = int(np.ceil(n ** (1/3.0))) + 3
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # FCC lattice points: integer combinations of basis vectors
                # Basis: a1=(0,0,0), a2=(0,0.5,0.5), a3=(0.5,0,0.5), a4=(0.5,0.5,0)
                # So point = i*a2 + j*a3 + k*a4 = (0.5*j+0.5*k, 0.5*i+0.5*k, 0.5*i+0.5*j)
                # But we need to include the (0,0,0) type too. Let's use a different approach.
                # Actually, FCC lattice points are at (i,j,k) where i+j+k is even, scaled by 0.5
                # Or equivalently: all (x,y,z) where x,y,z are all integers or all half-integers with even sum
                pass
    
    # Better approach: generate FCC points directly
    # FCC lattice vectors in conventional cubic cell of side a=1:
    # Primitive vectors: a1 = (0, 0.5, 0.5), a2 = (0.5, 0, 0.5), a3 = (0.5, 0.5, 0)
    # Lattice point = n1*a1 + n2*a2 + n3*a3
    pts = []
    for n1 in range(-max_idx, max_idx + 1):
        for n2 in range(-max_idx, max_idx + 1):
            for n3 in range(-max_idx, max_idx + 1):
                x = 0.5 * (n2 + n3)
                y = 0.5 * (n1 + n3)
                z = 0.5 * (n1 + n2)
                pts.append((x, y, z))
    
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _generate_hcp_cluster(n):
    """Generate a compact HCP-like cluster."""
    max_idx = int(np.ceil(n ** (1/3.0))) + 3
    pts = []
    # HCP: two interpenetrating triangular lattices
    a = 1.0  # in-plane lattice constant
    c = np.sqrt(8/3) * a  # ideal c/a ratio
    for layer in range(-max_idx, max_idx + 1):
        z = layer * c / 2
        # Triangular lattice in xy plane
        for i in range(-max_idx, max_idx + 1):
            for j in range(-max_idx, max_idx + 1):
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3) / 2 * j
                # A site
                pts.append((x, y, z))
                # B site (shifted)
                x2 = a * (i + 0.5 * ((j + 1) % 2)) + a / 3
                y2 = a * np.sqrt(3) / 2 * j - a * np.sqrt(3) / 6
                pts.append((x2, y2, z))
    
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _generate_icosahedral_cluster(n):
    """Generate an icosahedral cluster (Mackay icosahedron-like)."""
    # Start with icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit radius
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Mackay icosahedron: concentric shells
    # Shell 0: center
    # Shell 1: 12 vertices
    # Shell 2: 42 vertices
    # Shell 3: 92 vertices
    # Total: 1, 13, 55, 147...
    
    points = [np.zeros(3)]
    points.extend([v.copy() for v in verts])
    
    # Add second shell: 42 points
    # These are at positions derived from the icosahedral symmetry
    # For simplicity, we'll use a scaled version and add points on the faces
    # Actually, let's just generate a dense icosahedral-like structure
    
    # For larger n, we'll fill in with points on icosahedral symmetry axes
    if n > 13:
        # Add points on edges and faces
        # 30 edges: midpoints
        from itertools import combinations
        edge_midpoints = []
        for i, j in combinations(range(12), 2):
            mid = (verts[i] + verts[j]) / 2
            # Only add if it's a true edge (distance ~ 2*sin(pi/5) for unit icosahedron)
            d = np.linalg.norm(verts[i] - verts[j])
            if 0.8 < d < 1.2:  # approximate edge length
                edge_midpoints.append(mid)
        
        # Deduplicate
        unique_edges = []
        for p in edge_midpoints:
            if not any(np.linalg.norm(p - q) < 0.01 for q in unique_edges):
                unique_edges.append(p)
        
        points.extend([p.copy() for p in unique_edges])
        
        # Add more points by scaling
        for scale in [1.5, 2.0, 2.5]:
            for v in verts:
                points.append(v * scale)
            for p in unique_edges:
                points.append(p * scale)
    
    points = np.array(points[:n])
    if len(points) < n:
        # Pad with random points
        rng = np.random.default_rng(42)
        while len(points) < n:
            p = rng.uniform(-2, 2, 3)
            points = np.vstack([points, p])
    
    return points[:n]


def _generate_bcc_cluster(n):
    """Generate a BCC-like cluster."""
    max_idx = int(np.ceil(n ** (1/3.0))) + 3
    pts = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # BCC: corners + body center
                pts.append((float(i), float(j), float(k)))
                pts.append((i + 0.5, j + 0.5, k + 0.5))
    
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _generate_random_compact(n, rng):
    """Generate a random compact cluster."""
    spacing = 0.72
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
    
    # If we didn't get enough, fill with random
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)


def _generate_sphere_shell(n):
    """Generate points on a sphere shell (for larger clusters)."""
    if n <= 1:
        return np.zeros((n, 3))
    
    # Fibonacci sphere
    points = []
    phi = np.pi * (3 - np.sqrt(5))  # golden angle
    radius = 0.72 * n ** (1.0 / 3.0) * 0.8
    
    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2
        r = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * r
        z = np.sin(theta) * r
        points.append([x * radius, y * radius, z * radius])
    
    return np.array(points)


def construct(instance):
    """Structure-based global search for Sutton-Chen cluster minima."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    start_time = time.time()
    deadline = start_time + 290  # 290s safety margin
    
    # Generate diverse initial structures
    generators = [
        _generate_fcc_cluster,
        _generate_hcp_cluster,
        _generate_icosahedral_cluster,
        _generate_bcc_cluster,
        _generate_sphere_shell,
    ]
    
    rng = np.random.default_rng(size)
    
    candidates = []
    
    # Add structure-based candidates
    for gen in generators:
        try:
            if time.time() > deadline - 30:
                break
            pts = gen(size)
            if pts.shape == (size, 3) and np.all(np.isfinite(pts)):
                candidates.append(pts)
        except Exception:
            pass
    
    # Add random compact candidates
    for _ in range(6):
        if time.time() > deadline - 30:
            break
        try:
            pts = _generate_random_compact(size, rng)
            candidates.append(pts)
        except Exception:
            pass
    
    # Relax each candidate and keep the best
    best_energy = np.inf
    best_positions = None
    
    for i, pts in enumerate(candidates):
        if time.time() > deadline - 15:
            break
        try:
            energy, positions = relax(pts, potential, kind)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue
    
    # If we still have time, try a few more random restarts
    while time.time() < deadline - 15:
        try:
            pts = _generate_random_compact(size, rng)
            energy, positions = relax(pts, potential, kind)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            break
    
    if best_positions is None:
        # Fallback: random compact
        best_positions = _generate_random_compact(size, rng)
    
    return best_positions
# EVOLVE-BLOCK-END

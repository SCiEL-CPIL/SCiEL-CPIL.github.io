import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
import time

def _generate_fcc_like(n):
    """Generate an FCC-like cluster of size n."""
    # Generate points in a cubic lattice with FCC basis
    # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    # We generate a large enough block and then select the n closest to center.
    L = int(np.ceil(n ** (1/3))) + 2
    coords = []
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            for k in range(-L, L+1):
                # Add FCC basis vectors
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    x = i + dx
                    y = j + dy
                    z = k + dz
                    coords.append([x, y, z])
    coords = np.array(coords)
    # Center
    coords -= coords.mean(axis=0)
    # Sort by distance to origin
    dists = np.linalg.norm(coords, axis=1)
    indices = np.argsort(dists)
    # Take n closest
    return coords[indices[:n]]

def _generate_ico_like(n):
    """Generate an Icosahedral-like cluster of size n."""
    # Icosahedral shells: 1, 12, 30, 56, 92, 120...
    # We will construct shells manually.
    # Shell 1: 1 atom at center.
    # Shell 2: 12 atoms at icosahedron vertices.
    # Shell 3: 30 atoms at edge midpoints (approx) or specific icosahedral positions.
    # For simplicity and robustness, we use a "random" icosahedral packing:
    # Start with 1, then add 12 at distance r1, then add more on a spherical shell.
    
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of icosahedron
    ico_verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_verts /= np.linalg.norm(ico_verts[0])
    
    # Scale shell 2 to typical bond length ~ 1.0 (relative to lattice constant 1, SC equilibrium is ~0.8-1.0)
    # Let's assume ideal bond length r0 = 1.0 for generation, relaxation will adjust.
    r2 = 1.0
    points.extend([v * r2 for v in ico_verts])
    
    if n <= 13:
        return np.array(points[:n])
        
    # Shell 3: 30 atoms. These are typically at the midpoints of the icosahedron edges.
    # There are 30 edges in an icosahedron.
    # Let's approximate by placing 30 atoms on a sphere of radius r3.
    # To be more "structural", we can use the 30 edge midpoints of the icosahedron.
    # Indices of vertices connected by edges:
    # This is complex to hardcode. Alternative: Use a spherical Fibonacci distribution for remaining atoms?
    # No, that loses the "icosahedral" symmetry.
    # Let's use a simpler heuristic: Place atoms on a sphere of radius r3, using a random projection but constrained to be near the icosahedral symmetry axes?
    # Actually, for global minima search, a "random" icosahedral-like blob is often sufficient if the optimizer is good.
    # Let's just add atoms on a spherical shell with some jitter to break symmetry if needed, 
    # but keeping the first 13 exact.
    
    # For n > 13, we add atoms on a shell of radius r3 = 1.0 * np.sqrt(2) * 0.9? 
    # In a perfect icosahedron, the distance from center to edge midpoint is 1.0 * cos(pi/5) * 2? 
    # Distance to vertex is 1. Edge length is 2*sin(pi/5). Midpoint distance is cos(pi/5).
    # So r3 = cos(pi/5) * r2.
    r3 = r2 * np.cos(np.pi/5)
    
    # We need to generate 30 points on this shell.
    # We can use the 30 edge midpoints.
    # Vertices: 0-11.
    # Edges in icosahedron:
    # 0-1, 1-2, 2-3, 3-4, 4-0 (top pentagon? No, icosahedron has 5 triangles meeting at vertex 0)
    # Let's just use a generic "add to shell" method that tries to maintain minimal distance.
    
    current_points = np.array(points)
    target_count = n
    while len(current_points) < target_count:
        # Try to add a point on the sphere of radius r3
        # Random direction
        dir_vec = np.random.randn(3)
        dir_vec /= np.linalg.norm(dir_vec)
        new_p = dir_vec * r3
        # Check if it's too close to existing points
        dists = np.linalg.norm(current_points - new_p, axis=1)
        if np.min(dists) > 0.5 * r3: # Heuristic minimum distance
            current_points = np.vstack([current_points, new_p])
        # If we get stuck, lower the threshold or just accept
        if len(current_points) >= target_count:
            break
        # Fallback: if we can't place, just place at random position on shell
        if np.random.rand() > 0.9: # 10% chance to force placement to ensure we reach n
             current_points = np.vstack([current_points, new_p])
             
    return current_points[:target_count]

def _generate_random_compact(n, seed=None):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    # Start with a small sphere and add atoms
    points = [np.array([0.0, 0.0, 0.0])]
    radius = 0.5
    while len(points) < n:
        # Try to place a new point
        for _ in range(50):
            # Random direction
            dir_vec = rng.normal(0, 1, 3)
            dir_vec /= np.linalg.norm(dir_vec)
            # Random radius between 0.5 and current_max_radius
            r = rng.uniform(0.5, radius)
            p = dir_vec * r
            # Check distance to existing
            if all(np.linalg.norm(p - q) > 0.6 for q in points):
                points.append(p)
                break
        if len(points) < n:
            # If we couldn't place, expand radius slightly
            radius *= 1.05
            # Or just place it randomly
            if rng.rand() > 0.9:
                dir_vec = rng.normal(0, 1, 3)
                dir_vec /= np.linalg.norm(dir_vec)
                p = dir_vec * radius
                points.append(p)
    return np.array(points[:n])

def construct(instance):
    """
    Construct a cluster for the given instance.
    Uses multi-start relaxation with structured initial guesses (FCC and Icosahedral)
    to guide the optimizer towards the global minimum.
    """
    n = instance["n"]
    kind = instance["kind"]
    
    # Time limit per instance is ~50s (300s / 6 instances).
    # We will generate a few candidates and relax them.
    
    candidates = []
    
    # 1. FCC-like guess
    try:
        fcc_pts = _generate_fcc_like(n)
        candidates.append(fcc_pts)
    except Exception:
        pass
        
    # 2. Icosahedral-like guess
    try:
        ico_pts = _generate_ico_like(n)
        candidates.append(ico_pts)
    except Exception:
        pass
        
    # 3. Random compact guesses (3 variations)
    for i in range(3):
        try:
            rand_pts = _generate_random_compact(n, seed=42 + i)
            candidates.append(rand_pts)
        except Exception:
            pass

    best_energy = float('inf')
    best_positions = None
    
    # Relax each candidate
    for i, pts in enumerate(candidates):
        # Ensure finite
        if not np.all(np.isfinite(pts)):
            continue
        try:
            energy, relaxed_pts = relax(pts, 'sutton_chen', kind)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pts
        except Exception as e:
            # If relaxation fails, skip
            pass
            
    # If all failed, return a valid random cluster
    if best_positions is None:
        # Fallback: simple random
        rng = np.random.default_rng(0)
        best_positions = rng.normal(0, 1, (n, 3))
        
    return best_positions

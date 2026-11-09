"""
Strategy: Polytetrahedral (PT) Motif Seeding with Pruning/Growing and Multi-Start Relaxation.

The Dzugutov potential favors icosahedral order. We generate high-quality starting points
based on Mackay icosahedra (n=55) and modify them to fit other cluster sizes (n=38..60).
We then perform multi-start local relaxation to find the global minimum.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax

# Global time tracking
_START_TIME = time.time()
_HARD_LIMIT = 300.0
_SAFETY_MARGIN = 10.0

def _time_left():
    return _HARD_LIMIT - _SAFETY_MARGIN - (time.time() - _START_TIME)

def _generate_icosahedron_vertices():
    """Generate vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2.0
    # 12 vertices of an icosahedron
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    return verts

def _generate_mackay_55():
    """Generate the 55-atom Mackay icosahedron."""
    verts = _generate_icosahedron_vertices()
    
    # Radii for the three shells (center, vertices, edges, faces)
    # Typical Dzugutov equilibrium distances are around 1.0-1.2 for first shell.
    # We will scale the whole structure later or use relative ratios.
    # Geometric Mackay ratios: 
    # R1 (vertices) = 1
    # R2 (edges) = sqrt(1 + (1/phi)^2) ? No, distance from center to edge midpoint.
    # In a unit icosahedron (edge length 1), R1 = 1/sqrt(10+2*sqrt(5)) * 5? 
    # Let's just use the normalized vertices and define shell radii empirically or geometrically.
    
    # For a regular icosahedron with vertices on unit sphere:
    # R1 = 1.0
    # Edge length a = sqrt(2 - 2/phi^2) ? 
    # Distance from center to edge midpoint R2 = R1 * cos(theta/2) where theta is angle between adjacent vertices.
    # Angle between adjacent vertices of icosahedron: arccos(1/sqrt(5)) ? No.
    # Dot product of adjacent vertices (0,1,phi) and (1,phi,0) normalized:
    # (0*1 + 1*phi + phi*0) / (sqrt(1+phi^2) * sqrt(1+phi^2)) = phi / (1+phi^2) = 1/phi.
    # So cos(theta) = 1/phi. theta = arccos(1/phi).
    # R2 = R1 * cos(theta/2) = sqrt((1 + 1/phi)/2).
    # Face center R3: Distance from center to centroid of a face.
    # Face vertices v1, v2, v3. Centroid C = (v1+v2+v3)/3.
    
    R1 = 1.0
    R2 = np.sqrt((1 + 1/phi) / 2.0)
    
    # Calculate R3
    # Take first 3 vertices that form a face. 
    # In the standard icosahedron, faces are triangles.
    # Let's just compute it for one face.
    v1, v2, v3 = verts[0], verts[1], verts[4] # Check if these form a face? 
    # Actually, it's safer to just use a standard ratio or optimize. 
    # For Mackay icosahedron, the radii are often R1:R2:R3 = 1 : 1.236 : 1.528?
    # Let's use the geometric calculation.
    # A face of the icosahedron is an equilateral triangle.
    # The centroid of a face is at distance R3 from center.
    # R3 = R1 * cos(alpha) where alpha is angle from center to face normal.
    # In icosahedron, the angle from center to vertex is 0, to face normal is arccos(1/sqrt(15))? No.
    # Let's just compute the centroid of the first face defined by vertices 0, 1, 4? 
    # Vertices: 0:(0,1,phi), 1:(0,1,-phi), 4:(1,phi,0).
    # Are 0,1,4 connected?
    # Dist(0,1) = 2phi / norm. Dist(0,4) = ...
    # It's easier to just use the known ratio for Mackay icosahedron: 
    # R1 = 1
    # R2 = sqrt(1 + 1/phi^2) / 2 ? No.
    # Let's stick to the geometric derivation:
    # R2 = sqrt((1 + 1/phi)/2)
    # R3 = sqrt((1 + 1/phi^2) / 3) ? No.
    # Let's calculate R3 explicitly.
    # Face normal direction is the sum of the 3 vertex vectors.
    # n = v1 + v2 + v3. R3 = ||n|| / 3.
    # Let's pick a face. In the icosahedron, vertices 0, 1, 4 are NOT a face.
    # 0 is (0,1,phi). 1 is (0,1,-phi). They are adjacent? No, dist is 2phi.
    # The edge length is 2/sqrt(1+phi^2) * something.
    # Let's just use a heuristic: R3 = R2 * 1.2 or similar, or compute it properly.
    
    # Proper calculation:
    # Find a face. 
    # Vertices:
    # 0: (0, 1, phi)
    # 4: (1, phi, 0)
    # 8: (phi, 0, 1)
    # Check distances:
    # d(0,4) = sqrt(1 + (phi-1)^2 + phi^2) = sqrt(1 + 1 + 2phi + phi^2) = sqrt(2 + 2phi + phi^2)
    # Since phi^2 = phi+1, d^2 = 2 + 2phi + phi + 1 = 3 + 3phi = 3(1+phi).
    # d(0,8) = sqrt(phi^2 + 1 + (phi-1)^2) = same.
    # d(4,8) = sqrt((1-phi)^2 + phi^2 + 1) = same.
    # So 0, 4, 8 form a face.
    
    v_face = verts[[0, 4, 8]]
    centroid = np.mean(v_face, axis=0)
    R3 = np.linalg.norm(centroid)
    
    # Construct 55 atoms
    atoms = [np.zeros(3)] # Center
    
    # 12 vertices (Shell 1)
    for v in verts:
        atoms.append(R1 * v)
        
    # 30 edge midpoints (Shell 2)
    # We need to find all edges. An edge connects vertices with distance d_edge.
    d_edge = np.linalg.norm(verts[0] - verts[4])
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if np.linalg.norm(verts[i] - verts[j]) == d_edge:
                edges.append((i, j))
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        # Normalize and scale to R2
        norm_mid = np.linalg.norm(mid)
        if norm_mid > 0:
            mid = mid / norm_mid
            atoms.append(R2 * mid)
            
    # 12 face centers (Shell 3)
    # We need to find all faces. A face is a triangle of 3 vertices.
    # In an icosahedron, each face is an equilateral triangle.
    # We can find faces by checking triples of vertices that are mutually connected.
    faces = []
    # Build adjacency list
    adj = {i: [] for i in range(12)}
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
        
    for i in range(12):
        for j in adj[i]:
            if j > i:
                # Find common neighbors
                common = [k for k in adj[j] if k in adj[i] and k != i]
                for k in common:
                    if i < j and j < k:
                        faces.append((i, j, k))
                        
    for i, j, k in faces:
        centroid = (verts[i] + verts[j] + verts[k]) / 3.0
        norm_c = np.linalg.norm(centroid)
        if norm_c > 0:
            centroid = centroid / norm_c
            atoms.append(R3 * centroid)
            
    return np.array(atoms)

def _prune_or_grow(mackay_55, target_n, rng):
    """
    Modify the 55-atom Mackay structure to have target_n atoms.
    Pruning: Remove atoms from the outermost shells.
    Growing: Add atoms on the surface.
    """
    atoms = mackay_55.copy()
    current_n = len(atoms)
    
    if current_n == target_n:
        return atoms
    
    if current_n > target_n:
        # Prune
        # Sort atoms by distance from center. Outer atoms are removed first.
        # We want to keep the core.
        dists = np.linalg.norm(atoms, axis=1)
        # Indices sorted by distance (descending)
        idx_sorted = np.argsort(dists)[::-1]
        
        # We need to remove (current_n - target_n) atoms.
        # However, removing random outer atoms might break symmetry.
        # Better: Identify shell indices.
        # Shell 3 (faces): indices 43-54 (12 atoms)
        # Shell 2 (edges): indices 13-42 (30 atoms)
        # Shell 1 (verts): indices 1-12 (12 atoms)
        # Center: index 0
        
        remove_count = current_n - target_n
        # Prioritize removing from Shell 3, then Shell 2.
        # Shell 3 indices: 43 to 54
        # Shell 2 indices: 13 to 42
        
        to_remove = []
        # Randomly select from Shell 3
        shell3_indices = list(range(43, 55))
        rng.shuffle(shell3_indices)
        for idx in shell3_indices:
            if len(to_remove) < remove_count:
                to_remove.append(idx)
        
        # If still need to remove, select from Shell 2
        if len(to_remove) < remove_count:
            shell2_indices = list(range(13, 43))
            rng.shuffle(shell2_indices)
            for idx in shell2_indices:
                if len(to_remove) < remove_count:
                    to_remove.append(idx)
                    
        # If still need to remove (unlikely for n>=38), select from Shell 1
        if len(to_remove) < remove_count:
            shell1_indices = list(range(1, 13))
            rng.shuffle(shell1_indices)
            for idx in shell1_indices:
                if len(to_remove) < remove_count:
                    to_remove.append(idx)
                    
        # Remove these indices
        # Create mask
        mask = np.ones(current_n, dtype=bool)
        for idx in to_remove:
            mask[idx] = False
        atoms = atoms[mask]
        
    else:
        # Grow
        # Add atoms on the surface.
        add_count = target_n - current_n
        # Estimate max radius
        max_r = np.max(np.linalg.norm(atoms, axis=1))
        
        new_atoms = []
        attempts = 0
        while len(new_atoms) < add_count and attempts < 1000:
            attempts += 1
            # Random point on sphere of radius max_r * 1.1
            # Or random point in shell
            # Let's try adding atoms near the existing surface
            # Pick a random existing atom and offset it
            idx = rng.integers(0, len(atoms))
            p = atoms[idx]
            # Random direction
            dir = rng.normal(size=3)
            dir = dir / np.linalg.norm(dir)
            # Offset distance
            offset = 0.8 + 0.4 * rng.random()
            new_p = p + dir * offset
            
            # Check minimum distance
            valid = True
            for q in atoms:
                if np.linalg.norm(new_p - q) < 0.5:
                    valid = False
                    break
            for q in new_atoms:
                if np.linalg.norm(new_p - q) < 0.5:
                    valid = False
                    break
            
            if valid:
                new_atoms.append(new_p)
                
        if new_atoms:
            atoms = np.vstack([atoms, np.array(new_atoms)])
            
    return atoms

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    
    # Get time budget
    t_left = _time_left()
    if t_left < 5.0:
        # Not enough time, return a quick valid structure
        # Just a random cluster or the Mackay 55 pruned/grown without relaxation
        # But we must return something valid.
        # Let's just return the Mackay 55 pruned/grown.
        mackay = _generate_mackay_55()
        rng = np.random.default_rng(size)
        pos = _prune_or_grow(mackay, size, rng)
        return pos

    # Determine number of restarts based on time and size
    # Larger clusters take longer to relax.
    # Assume relax takes ~1-2s per restart for small, ~5-10s for large.
    # Let's allocate 20 restarts for n<50, 10 for n>=50.
    if size < 50:
        max_restarts = 20
    else:
        max_restarts = 10
        
    # Adjust if time is tight
    est_time_per_restart = 2.0 if size < 50 else 5.0
    max_restarts = min(max_restarts, int(t_left / est_time_per_restart))
    if max_restarts < 1:
        max_restarts = 1

    rng = np.random.default_rng(size)
    mackay = _generate_mackay_55()
    
    best_energy = float('inf')
    best_pos = None
    
    for i in range(max_restarts):
        # Check time
        if _time_left() < 2.0:
            break
            
        # Generate candidate
        if i == 0:
            # First try: exact PT structure (pruned/grown)
            pos = _prune_or_grow(mackay, size, rng)
        else:
            # Subsequent tries: Perturbed PT structure
            # Start with pruned/grown structure
            pos = _prune_or_grow(mackay, size, rng)
            
            # Apply random perturbations
            # Scale factor
            scale = 1.0 + 0.1 * rng.normal()
            pos = pos * scale
            
            # Random noise
            noise = 0.05 * rng.normal(size=pos.shape)
            pos = pos + noise
            
            # Ensure minimum distance is not violated badly before relax
            # Relax should handle it, but if atoms overlap, energy is inf.
            # We can skip this check for speed, relax will push them apart.
            
        try:
            energy, relaxed_pos = relax(pos, potential)
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos
        except Exception:
            continue
            
    if best_pos is None:
        # Fallback: return the unrelaxed PT structure
        best_pos = _prune_or_grow(mackay, size, rng)
        
    return best_pos
# EVOLVE-BLOCK-END

"""Sutton-Chen transition-metal cluster global minima: multi-start structured seeds + L-BFGS-B relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _fcc_cluster(n, spacing=0.72):
    """Generate an n-atom fcc-like cluster by filling fcc lattice sites nearest to origin."""
    if n <= 0:
        return np.empty((0, 3))
    
    # fcc basis vectors (conventional cubic cell side a=1, atoms at corners and face centers)
    # We generate points in a grid and filter by distance to origin.
    # To get n points, we need a shell large enough.
    # Approximation: volume ~ n * spacing^3. Side length L ~ (n)^(1/3) * spacing.
    # fcc density is higher. Let's just generate a large enough grid and sort by distance.
    
    # Use a slightly larger shell to ensure we have enough points
    max_shell = int(np.ceil(n ** (1.0 / 3.0) * 1.5)) + 2
    
    sites = []
    # Generate fcc points: (i, j, k) where i,j,k are integers, and parity of i+j+k is even? 
    # Actually fcc is: (0,0,0) type and (0.5, 0.5, 0) type etc.
    # Simpler: iterate over integer grid i,j,k.
    # Point is (i, j, k) if i+j+k is even? No, that's BCC.
    # FCC: (i, j, k) where i,j,k are integers.
    # Basis: (0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)
    # Let's generate a supercell.
    
    points = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                # Check if (i,j,k) corresponds to an FCC site
                # In a cubic lattice with basis, sites are at:
                # (i, j, k)
                # (i+0.5, j+0.5, k)
                # (i+0.5, j, k+0.5)
                # (i, j+0.5, k+0.5)
                # It's easier to just generate a grid of points with spacing 0.5 and filter.
                # But that's 8x points.
                
                # Alternative: Just generate a simple cubic grid with spacing 0.72/np.sqrt(2) ~ 0.51?
                # No, let's stick to standard FCC generation.
                
                # For each integer i,j,k, we have 4 points in the cell [i, i+1]x[j, j+1]x[k, k+1]
                # Actually, let's just generate points on the grid with step 0.5 and keep those 
                # where (2x, 2y, 2z) are all even or all odd? No.
                
                # Let's use the explicit basis method.
                for bx, by, bz in [(0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    p = np.array([i + bx, j + by, k + bz])
                    points.append(p)
                    
    if not points:
        return np.zeros((n, 3))
        
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    selected = points[idx[:n]]
    
    # Center
    selected = selected - np.mean(selected, axis=0)
    
    # Scale to approximate correct size? 
    # The relaxation will handle scaling/spacing. 
    # However, if initial spacing is too large/small, it might be slow.
    # Typical SC equilibrium distance is around 1.0 (in reduced units where lattice const=1? No, lattice const is 1 in problem, but SC min is around r~1.0-1.1?)
    # The problem states "reduced units (lattice constant 1)".
    # For SC (12-6), min is around 1.0.
    # My spacing param 0.72 was from seed. Let's assume the generated points are roughly at unit distance.
    # If I used basis 0.5, the nearest neighbor distance is sqrt(0.5^2+0.5^2) = 0.707.
    # This is close to typical cluster sizes.
    
    return selected

def _hcp_cluster(n, spacing=0.72):
    """Generate an n-atom hcp-like cluster."""
    if n <= 0:
        return np.empty((0, 3))
        
    points = []
    # HCP layers: ABAB...
    # Layer A: hexagonal packing
    # Layer B: hexagonal packing shifted
    
    # Determine number of layers needed
    # Approx atoms per layer ~ k^2
    # Total n ~ sum k^2 ~ L^3/3
    # L ~ (3n)^(1/3)
    max_layers = int(np.ceil((3 * n) ** (1.0 / 3.0))) + 2
    
    a = spacing # lattice constant in plane
    c = spacing * np.sqrt(8/3) # ideal c/a ratio
    
    for layer_idx in range(max_layers):
        z = layer_idx * c
        # Alternate A and B
        is_b = (layer_idx % 2 == 1)
        
        # Generate hexagonal grid for this layer
        # Range of x,y to cover enough atoms
        max_radius = max_layers * a
        for i in range(-max_radius, max_radius + 1):
            for j in range(-max_radius, max_radius + 1):
                # Hexagonal coordinates to Cartesian
                # x = a * (i + 0.5 * j)
                # y = a * (sqrt(3)/2 * j)
                x = a * (i + 0.5 * j)
                y = a * (np.sqrt(3) / 2 * j)
                
                if is_b:
                    # Shift B layer to sit in holes
                    # Shift by (a/sqrt(3), a/sqrt(3))? 
                    # In hexagonal packing, B is shifted by (1/3, 1/3) of basis vectors.
                    # Vector 1: (a, 0), Vector 2: (a/2, a*sqrt(3)/2)
                    # Shift = (1/3)V1 + (1/3)V2 = (a/3 + a/6, a*sqrt(3)/6) = (a/2, a*sqrt(3)/6)
                    x += a / 2
                    y += a * np.sqrt(3) / 6
                
                points.append([x, y, z])
                
    if not points:
        return np.zeros((n, 3))
        
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    selected = points[idx[:n]]
    
    # Center
    selected = selected - np.mean(selected, axis=0)
    
    return selected

def _icosahedral_cluster(n):
    """Generate an n-atom icosahedral cluster (Mackay icosahedron)."""
    if n <= 0:
        return np.empty((0, 3))
        
    # Mackay icosahedron shells: 1, 12, 42, 100, ...
    # We will generate points on the icosahedron surface and fill inward.
    # Or just generate the specific shells.
    
    # Vertices of a regular icosahedron
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    vertices = vertices / np.linalg.norm(vertices[0]) # Normalize to radius 1
    
    # Generate points on the surface of icosahedron shells
    # This is complex. A simpler approximation for "icosahedral-like" is to use the 
    # vertices and face centers and edge midpoints of a subdivided icosahedron.
    
    # For now, let's use a simpler "nested icosahedron" or just a distorted sphere 
    # with icosahedral symmetry if n matches.
    # If n doesn't match, we take the closest subset.
    
    # Let's generate a dense set of points on the icosahedron surface using subdivision.
    # Subdivision level 0: 12 vertices
    # Level 1: 42 points (12 vertices + 30 edge midpoints)
    # Level 2: 162 points (42 + 120 face centers)
    
    points = []
    
    # Level 0
    points.extend(vertices.tolist())
    
    # Level 1: Edge midpoints
    # Find unique edges
    edges = set()
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(vertices[i] - vertices[j])
            if d < 1.0: # Adjacent vertices in unit icosahedron are dist ~ 1.05?
                 # Actually min dist in unit icosahedron is 1.051?
                 # Let's just check distance < 1.5
                pass
    
    # Simpler: Just generate a random sphere and project? No, that's not icosahedral.
    # Let's use the known coordinates for small clusters or just use the vertices 
    # and their convex hull interior points?
    
    # Given time constraints, let's approximate "icosahedral" with a distorted sphere 
    # that has 12-fold symmetry axes? 
    # Actually, for n=38, 55, 59, 75, 79, 80, none are perfect Mackay numbers (13, 55, 147).
    # 55 is a Mackay number (1+12+42).
    # For others, we can take a subset of a larger Mackay cluster or just use 
    # a spherical shell with icosahedral distortion.
    
    # Let's generate a dense icosahedron (Level 2: 162 points) and take the first n.
    
    # Generate Level 1 (42 points)
    level1 = []
    # Vertices
    level1.extend(vertices.tolist())
    
    # Edges: connect vertices if distance is min
    min_dist = np.inf
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(vertices[i] - vertices[j])
            if d < min_dist:
                min_dist = d
                
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(vertices[i] - vertices[j])
            if d < min_dist * 1.1:
                mid = (vertices[i] + vertices[j]) / 2
                level1.append(mid)
                
    level1 = np.array(level1)
    # Normalize level1 to radius 1
    if len(level1) > 0:
        norms = np.linalg.norm(level1, axis=1)
        level1 = level1 / norms[:, np.newaxis]
        
    # Generate Level 2 (162 points)
    # Add face centers
    # Faces are triangles of 3 vertices that are mutually adjacent.
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(vertices[i] - vertices[j])
                d2 = np.linalg.norm(vertices[j] - vertices[k])
                d3 = np.linalg.norm(vertices[i] - vertices[k])
                if d1 < min_dist * 1.1 and d2 < min_dist * 1.1 and d3 < min_dist * 1.1:
                    faces.append((i, j, k))
                    
    level2 = []
    level2.extend(level1.tolist())
    for i, j, k in faces:
        center = (vertices[i] + vertices[j] + vertices[k]) / 3
        level2.append(center)
        
    level2 = np.array(level2)
    if len(level2) > 0:
        norms = np.linalg.norm(level2, axis=1)
        level2 = level2 / norms[:, np.newaxis]
        
    # If n > 162, we need more. But max n is 80. So level2 is enough.
    # If n < 12, use vertices.
    
    if n <= len(vertices):
        selected = vertices[:n]
    elif n <= len(level1):
        selected = level1[:n]
    else:
        selected = level2[:n]
        
    # Scale to appropriate size (radius ~ n^(1/3) * 0.5?)
    # Relaxation will fix it, but good initial scale helps.
    # Typical radius for n=80 is ~ 3-4 units.
    # Our radius is 1. Let's scale by n^(1/3) * 0.5
    scale = 0.5 * n ** (1.0/3.0)
    selected = selected * scale
    
    return selected

def _random_compact(n, rng):
    """Generate a random compact cluster."""
    spacing = 0.72
    radius = spacing * n ** (1.0 / 3.0) * 0.75
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check distance
            if all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
                points.append(p)
    return np.array(points)

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 290  # Leave 10s margin
    
    rng = np.random.default_rng(size)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse starting points
    starts = []
    
    # 1. FCC
    try:
        starts.append(_fcc_cluster(size))
    except Exception:
        pass
        
    # 2. HCP
    try:
        starts.append(_hcp_cluster(size))
    except Exception:
        pass
        
    # 3. Icosahedral
    try:
        starts.append(_icosahedral_cluster(size))
    except Exception:
        pass
        
    # 4. Random Compact (multiple)
    for _ in range(4):
        try:
            starts.append(_random_compact(size, rng))
        except Exception:
            pass
            
    # Relax each start
    for i, start in enumerate(starts):
        if time.time() > deadline:
            break
            
        try:
            energy, positions = relax(start, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
            
    # If time remains, do a final polish on the best solution
    if best_positions is not None and time.time() < deadline - 5:
        try:
            # Use a tighter tolerance or more iterations?
            # relax uses L-BFGS-B. We can call it again with the best positions.
            # Or use sutton_chen_energy_gradient to check?
            # Let's just call relax again, it might find a slightly better local min 
            # if the previous one stopped early? Unlikely if it converged.
            # Instead, let's try a few more random starts if time permits.
            while time.time() < deadline - 10:
                start = _random_compact(size, rng)
                energy, positions = relax(start, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass

    if best_positions is None:
        # Fallback: return a simple valid cluster
        best_positions = _random_compact(size, rng)
        
    return best_positions
# EVOLVE-BLOCK-END

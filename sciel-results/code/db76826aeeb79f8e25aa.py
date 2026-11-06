import numpy as np
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
import time

def _get_icosahedron_vertices(radius=1.0):
    """
    Generate the 12 vertices of a regular icosahedron centered at origin.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi],
        [0, -1, phi],
        [0, 1, -phi],
        [0, -1, -phi],
        [1, phi, 0],
        [-1, phi, 0],
        [1, -phi, 0],
        [-1, -phi, 0],
        [phi, 0, 1],
        [phi, 0, -1],
        [-phi, 0, 1],
        [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    return verts * radius

def _generate_mackay_icosahedron(n, r1=1.0, r2=1.8, r3=2.6):
    """
    Generate a cluster structure based on Mackay icosahedral shells.
    Shell 1: 12 atoms (icosahedron vertices)
    Shell 2: 42 atoms (12 vertices + 30 edge midpoints of icosahedron)
    Shell 3: 120 atoms (more complex, approximated)
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Shell 1: 12 atoms
    if n > 1:
        v1 = _get_icosahedron_vertices(r1)
        for v in v1:
            if len(points) < n:
                points.append(v)
    
    # Shell 2: 42 atoms
    if n > 13:
        base_verts = _get_icosahedron_vertices(1.0)
        
        # Find edges of the icosahedron
        min_dist = np.inf
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(base_verts[i] - base_verts[j])
                if d < min_dist:
                    min_dist = d
        
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(base_verts[i] - base_verts[j])
                if np.isclose(d, min_dist):
                    edges.append((i, j))
        
        # 42 points: 12 vertices + 30 edge midpoints
        shell2_pts = []
        for v in base_verts:
            shell2_pts.append(v)
        
        for i, j in edges:
            mid = (base_verts[i] + base_verts[j]) / 2.0
            shell2_pts.append(mid)
        
        shell2_pts = np.array(shell2_pts)
        norms = np.linalg.norm(shell2_pts, axis=1)
        shell2_pts = shell2_pts / norms[:, np.newaxis] * r2
        
        for v in shell2_pts:
            if len(points) < n:
                points.append(v)
    
    # Shell 3: approximate with another scaled icosahedron if needed
    if n > 55:
        v3 = _get_icosahedron_vertices(r3)
        for v in v3:
            if len(points) < n:
                points.append(v)
        
        while len(points) < n:
            v_add = _get_icosahedron_vertices(r3 + 0.5)
            for v in v_add:
                if len(points) < n:
                    points.append(v)
    
    return np.array(points[:n])

def _generate_polytetrahedral(n, r1=1.0, r2=1.8):
    """
    Generate a polytetrahedral structure.
    Dzugutov potential favors polytetrahedral order.
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Start with a central atom
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Add tetrahedral shells
    # A tetrahedron has 4 vertices
    # For polytetrahedral, we can use a close-packed arrangement
    
    # Approximate by using icosahedral shells but with different radii
    # that favor tetrahedral coordination
    
    if n > 1:
        # First shell: 12 atoms (icosahedral)
        v1 = _get_icosahedron_vertices(r1)
        for v in v1:
            if len(points) < n:
                points.append(v)
    
    if n > 13:
        # Second shell: 42 atoms
        base_verts = _get_icosahedron_vertices(1.0)
        min_dist = np.inf
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(base_verts[i] - base_verts[j])
                if d < min_dist:
                    min_dist = d
        
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(base_verts[i] - base_verts[j])
                if np.isclose(d, min_dist):
                    edges.append((i, j))
        
        shell2_pts = []
        for v in base_verts:
            shell2_pts.append(v)
        
        for i, j in edges:
            mid = (base_verts[i] + base_verts[j]) / 2.0
            shell2_pts.append(mid)
        
        shell2_pts = np.array(shell2_pts)
        norms = np.linalg.norm(shell2_pts, axis=1)
        shell2_pts = shell2_pts / norms[:, np.newaxis] * r2
        
        for v in shell2_pts:
            if len(points) < n:
                points.append(v)
    
    if n > 55:
        v3 = _get_icosahedron_vertices(2.6)
        for v in v3:
            if len(points) < n:
                points.append(v)
        
        while len(points) < n:
            v_add = _get_icosahedron_vertices(3.1)
            for v in v_add:
                if len(points) < n:
                    points.append(v)
    
    return np.array(points[:n])

def _generate_random_compact(n, seed):
    """
    Generate a random compact cluster.
    """
    rng = np.random.default_rng(seed)
    radius = 1.5 * n ** (1.0/3.0)
    
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 500:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if points:
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if np.min(dists) < 0.8:
                continue
        points.append(p)
        attempts += 1
        
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
            
    return np.array(points[:n])

def _generate_fcc_cluster(n):
    """
    Generate an FCC-like cluster.
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    points = []
    a = 1.0  # lattice constant
    
    # Generate FCC points in a cube
    max_coord = int(np.ceil((n / 4) ** (1/3))) + 1
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if len(points) >= n:
                    break
                # FCC: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    if len(points) >= n:
                        break
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    points.append(p)
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    
    # Center the cluster
    if points:
        points = np.array(points[:n])
        center = np.mean(points, axis=0)
        points = points - center
    
    return points

def construct(instance):
    """
    Find low-energy Dzugutov cluster structures.
    """
    size = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + 280  # 20s margin before 300s limit
    
    best_energy = float('inf')
    best_positions = None
    
    def try_relax(init_positions, label=""):
        nonlocal best_energy, best_positions
        try:
            energy, positions = relax(init_positions, potential, None)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
                return True
        except Exception:
            pass
        return False
    
    # 1. Mackay icosahedron
    try:
        init_ico = _generate_mackay_icosahedron(size)
        try_relax(init_ico, "mackay")
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_positions if best_positions is not None else _generate_mackay_icosahedron(size)
    
    # 2. Polytetrahedral
    try:
        init_poly = _generate_polytetrahedral(size)
        try_relax(init_poly, "polytetrahedral")
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_positions if best_positions is not None else _generate_mackay_icosahedron(size)
    
    # 3. FCC-like
    try:
        init_fcc = _generate_fcc_cluster(size)
        try_relax(init_fcc, "fcc")
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_positions if best_positions is not None else _generate_mackay_icosahedron(size)
    
    # 4. Random compact
    try:
        init_rand = _generate_random_compact(size, seed=size)
        try_relax(init_rand, "random")
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_positions if best_positions is not None else _generate_mackay_icosahedron(size)
    
    # 5. Perturbed best
    if best_positions is not None:
        try:
            rng = np.random.default_rng(size + 100)
            perturbed = best_positions + rng.normal(0, 0.1, best_positions.shape)
            try_relax(perturbed, "perturbed")
        except Exception:
            pass
    
    if time.time() > deadline:
        return best_positions if best_positions is not None else _generate_mackay_icosahedron(size)
    
    # 6. Another random seed
    try:
        init_rand2 = _generate_random_compact(size, seed=size + 500)
        try_relax(init_rand2, "random2")
    except Exception:
        pass
    
    if time.time() > deadline:
        return best_positions if best_positions is not None else _generate_mackay_icosahedron(size)
    
    # 7. More perturbation if time allows
    if best_positions is not None and time.time() < deadline - 30:
        try:
            rng = np.random.default_rng(size + 200)
            for _ in range(3):
                perturbed = best_positions + rng.normal(0, 0.05, best_positions.shape)
                try_relax(perturbed, "perturbed_extra")
                if time.time() > deadline - 10:
                    break
        except Exception:
            pass
    
    if best_positions is None:
        return _generate_mackay_icosahedron(size)
        
    return best_positions

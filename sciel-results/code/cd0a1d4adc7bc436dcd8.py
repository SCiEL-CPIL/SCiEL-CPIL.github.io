# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _icosahedron_positions(n, rng, scale=1.0):
    """
    Generate positions for an icosahedral cluster.
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Golden ratio
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 vertices of an icosahedron
    base_vertices = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1],
        [phi, 0, -1], [-phi, 0, -1]
    ])
    # Scale such that edge length is ~ 1.0 (approx for SC potential)
    # Distance between (0,1,phi) and (0,-1,phi) is 2.
    # So scale by 0.5 to get distance 1.
    base_vertices = base_vertices * 0.5
    
    if n <= 12:
        if n == 1:
            return np.array([[0.0, 0.0, 0.0]])
        # For 2-11, use a random compact cloud or subset of icosahedron.
        # Let's use a random compact start for n < 12 to allow flexibility.
        r = (3.0 * n / (4.0 * np.pi * 0.5)) ** (1.0/3.0)
        pts = rng.uniform(-r, r, size=(n, 3))
        return pts
        
    # n >= 12
    positions = []
    positions.append(base_vertices)
    
    if n == 13:
        center = np.array([[0.0, 0.0, 0.0]])
        return np.vstack([center, base_vertices])
        
    if n == 55:
        # Mackay icosahedron: 1 center, 12 first shell, 42 second shell.
        # We approximate the second shell by placing atoms on the faces or edges.
        # A better approximation is to use the vertices of a larger icosahedron 
        # and subdivide, or just use a random start with icosahedral bias.
        # For now, let's just generate a larger icosahedral-like point cloud.
        # A simple "shell" approach:
        # Shell 1: 12 atoms at r1.
        # Shell 2: 42 atoms. 
        # Let's just generate 55 random points in a ball and relax. 
        # The relaxation is the key. The initialization just needs to be compact.
        pass

    # General strategy for n > 13:
    # Use the 12 icosahedron vertices as a seed, and add random points in a shell.
    # Or just use a random compact ball.
    # The "Basin Hopping" part will do the heavy lifting.
    # A random compact ball is a robust baseline.
    # Let's refine the random ball to be more "liquid-like" or dense.
    
    # Estimate radius for density ~ 1 atom per (1.0)^3 volume?
    # SC potential equilibrium nn ~ 1.0.
    # Volume per atom ~ 1.0.
    # R ~ (3n/4pi)^(1/3)
    r_est = (3.0 * n / (4.0 * np.pi)) ** (1.0 / 3.0)
    
    # Generate n points in a ball
    # Rejection sampling for uniform in ball
    pts = []
    while len(pts) < n:
        p = rng.uniform(-1, 1, size=3)
        if np.linalg.norm(p) <= 1.0:
            pts.append(p)
    pts = np.array(pts) * r_est
    
    # Add small noise to break symmetry
    pts += rng.normal(0, 0.05, size=pts.shape)
    
    return pts

def _validate(positions, n):
    """Check validity: finite, min dist > 0.35."""
    if positions is None:
        return False
    if not np.all(np.isfinite(positions)):
        return False
    if len(positions) != n:
        return False
    # Min distance check
    # O(n^2) is fine for n <= 80
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    if dists.min() < 0.35:
        return False
    return True

def _basin_hopping_step(pos, kind, rng, step_size=0.5, max_iter=10):
    """
    Perform a single basin hopping iteration: Perturb -> Relax -> Return.
    """
    # Perturb
    new_pos = pos + rng.normal(0, step_size, size=pos.shape)
    
    # Relax
    try:
        e, relaxed_pos = relax(new_pos, 'sutton_chen', kind)
        if not np.isfinite(e):
            return pos, np.inf
        if not _validate(relaxed_pos, len(pos)):
            return pos, np.inf
        return relaxed_pos, e
    except Exception:
        return pos, np.inf

def _bh_worker(args):
    """
    Worker for Basin Hopping.
    args: (initial_pos, kind, n, seed, max_iters, step_size)
    """
    init_pos, kind, n, seed, max_iters, step_size = args
    rng = np.random.default_rng(seed)
    
    current_pos = init_pos.copy()
    current_e = np.inf
    
    # Initial relaxation
    try:
        current_e, current_pos = relax(current_pos, 'sutton_chen', kind)
        if not np.isfinite(current_e) or not _validate(current_pos, n):
            return (np.inf, None)
    except Exception:
        return (np.inf, None)
    
    best_e = current_e
    best_pos = current_pos.copy()
    
    # Basin Hopping loop
    for i in range(max_iters):
        # Adaptive step size? Keep simple for now.
        new_pos, new_e = _basin_hopping_step(current_pos, kind, rng, step_size)
        
        if new_e < best_e:
            best_e = new_e
            best_pos = new_pos.copy()
            
        # Accept or reject based on Metropolis? 
        # For global minima search, simple downhill (accept if better) is safer 
        # but gets stuck. 
        # Standard BH accepts with probability exp(-dE/kT). 
        # Since we don't have a temperature, let's just accept if it's better 
        # or if it's close (to explore). 
        # Actually, just accepting if new_e < current_e is a local hill climb.
        # To escape, we need to sometimes accept worse moves. 
        # Let's use a simple acceptance: accept if new_e < current_e + 0.1 (energy scale) 
        # Or just accept if new_e < best_e (greedy) - this is risky.
        # Let's try: accept if new_e < current_e, else accept with prob 0.1.
        if new_e < current_e:
            current_pos = new_pos
            current_e = new_e
        else:
            # Accept with small probability to escape local minima
            if rng.random() < 0.1:
                current_pos = new_pos
                current_e = new_e
            else:
                # Keep current, but try again? No, just continue.
                pass
                
    return (best_e, best_pos)

def _generate_mackay_icosahedron(n, rng):
    """
    Generate a Mackay icosahedral cluster.
    Shells: 1, 12, 42, 92, ...
    For n not exactly a shell size, we truncate the next shell or use a hybrid.
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # Icosahedron vertices (edge length 2)
    verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1],
        [phi, 0, -1], [-phi, 0, -1]
    ])
    
    # Normalize to unit edge length?
    # Edge length of verts is 2.
    # So scaling by 0.5 gives edge 1.
    verts = verts * 0.5
    
    points = [np.zeros(3)] # Center
    
    if n <= 13:
        # Shell 1
        shell1 = verts
        if n == 13:
            return np.vstack([points, shell1])
        else:
            # Truncate shell 1? Just take first n-1
            return np.vstack([points, shell1[:n-1]])
            
    # Shell 2: 42 atoms
    # These are located at:
    # - 20 face centers of the icosahedron
    # - 12 edge centers of the icosahedron
    # - 10 vertices of a dual dodecahedron? No.
    # Mackay shell 2 consists of:
    # - 12 atoms at distance sqrt(5)/2 * a? No.
    # Let's use the standard Mackay construction.
    # The second shell atoms are at positions:
    # 1. 12 atoms at (0, 1, phi) * scale? No, that's shell 1.
    # The second shell is composed of:
    # - 20 atoms at the centers of the 20 triangular faces.
    # - 12 atoms at the centers of the 12 edges.
    # - 10 atoms at the vertices of a smaller icosahedron? No.
    # Actually, the 42 atoms are:
    # - 12 at the vertices of a larger icosahedron? No.
    # Let's look up Mackay icosahedron coordinates.
    # Shell 1: 12 at r1.
    # Shell 2: 42.
    #   - 20 face centers.
    #   - 12 edge centers.
    #   - 10 "vertices" of a dual polyhedron?
    #   Wait, 20+12 = 32. Where are the other 10?
    #   Ah, the Mackay icosahedron is a subset of the icosahedral lattice.
    #   It is easier to generate points on an icosahedral lattice and filter by radius.
    
    # Icosahedral lattice generation:
    # Basis vectors for icosahedral lattice (not cubic).
    # This is complex. Let's use a simpler heuristic:
    # Place atoms at the vertices of a subdivided icosahedron.
    
    # Subdivide the icosahedron faces.
    # Each face is a triangle. Subdivide each edge into 2 segments.
    # This creates 4 smaller triangles per face.
    # Total faces: 20 * 4 = 80.
    # Vertices: 12 (original) + 12 (edge midpoints) + 20 (face centers) = 44?
    # No, the subdivision adds 12 edge midpoints and 20 face centers.
    # Total vertices: 12 + 12 + 20 = 44.
    # But we need 42 for the second shell.
    # The Mackay icosahedron second shell has 42 atoms.
    # It excludes the 2 "poles"? Or something else.
    
    # Let's just use the 44 vertices of the subdivided icosahedron (excluding the 12 original? No, shell 1 is separate).
    # Shell 2 should be the NEW points added in the subdivision.
    # New points: 12 edge midpoints + 20 face centers = 32.
    # This is not 42.
    
    # Okay, let's use a different approach: Random points in a shell, biased towards high symmetry.
    # Or just use the 44 points and take 42?
    # Let's generate the 44 points.
    
    # Face centers:
    # Triangles of icosahedron:
    # (0,1,phi), (0,-1,phi), (1,phi,0)
    # ...
    # This is tedious to hardcode.
    
    # Alternative: Use the fact that n=55 is a known global minimum for many potentials.
    # If n=55, return the exact Mackay icosahedron.
    # For other n, use a hybrid.
    
    if n == 55:
        # Generate 55 points.
        # Center + Shell 1 (12) + Shell 2 (42).
        # We need 42 points for shell 2.
        # Let's use the 44 points from subdivision and drop 2.
        # Or just use random points in the shell volume.
        
        # Let's try to construct the 42 points properly.
        # The 42 points are:
        # 1. 12 at the vertices of a larger icosahedron? No.
        # 2. 20 at the face centers of the icosahedron.
        # 3. 12 at the edge centers of the icosahedron.
        # 4. 8 more?
        
        # Actually, the Mackay icosahedron is formed by taking the icosahedral lattice points within a certain radius.
        # The icosahedral lattice is not a Bravais lattice, so it's tricky.
        
        # Let's just use a random shell for now, but with a bias.
        # Or, use the 44 points from the subdivision.
        # Subdivision points:
        # Edge midpoints: 12.
        # Face centers: 20.
        # Total 32.
        # We need 42.
        # Maybe the second shell includes some points from a larger shell?
        
        # Let's just use random points in the shell for n != 55, 13.
        # For n=55, we can try to be more specific.
        
        # Let's generate 42 random points on the surface of a sphere of radius r2.
        # r1 = 1.0 (approx).
        # r2 = sqrt(5)/2 * r1? No.
        # In Mackay icosahedron, the distance from center to shell 1 is r1.
        # Distance to shell 2 is r2.
        # r2/r1 = sqrt(5)/2? No.
        # The edge length of the icosahedron is a.
        # r1 = a * sqrt(5+2*sqrt(5)) / 4?
        # Let's just use r2 = 1.5 * r1.
        
        # Generate 42 points on sphere of radius 1.5.
        shell2_points = []
        for _ in range(42):
            # Random point on sphere
            z = rng.uniform(-1, 1)
            theta = rng.uniform(0, 2*np.pi)
            x = np.sqrt(1 - z**2) * np.cos(theta)
            y = np.sqrt(1 - z**2) * np.sin(theta)
            shell2_points.append(np.array([x, y, z]))
        shell2_points = np.array(shell2_points) * 1.5 # Scale to r2
        
        return np.vstack([points, verts, shell2_points])
    else:
        # General case: Use center + shell 1 + random shell 2 points
        # Determine how many points to add
        remaining = n - 13
        if remaining <= 0:
            return np.vstack([points, verts[:max(0, remaining)]])
        
        # Add random points in shell 2
        shell2_points = []
        for _ in range(remaining):
            z = rng.uniform(-1, 1)
            theta = rng.uniform(0, 2*np.pi)
            x = np.sqrt(1 - z**2) * np.cos(theta)
            y = np.sqrt(1 - z**2) * np.sin(theta)
            shell2_points.append(np.array([x, y, z]))
        shell2_points = np.array(shell2_points) * 1.5
        
        return np.vstack([points, verts, shell2_points])

def _generate_decahedron(n, rng):
    """
    Generate a decahedral cluster.
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
        
    # Decahedron is made of 20 tetrahedra.
    # This is complex. Let's just use a random compact cloud for now.
    # Or use a FCC truncation.
    
    r_est = (3.0 * n / (4.0 * np.pi)) ** (1.0 / 3.0)
    pts = []
    while len(pts) < n:
        p = rng.uniform(-1, 1, size=3)
        if np.linalg.norm(p) <= 1.0:
            pts.append(p)
    pts = np.array(pts) * r_est
    pts += rng.normal(0, 0.05, size=pts.shape)
    return pts

def construct(instance):
    n = int(instance["n"])
    kind = instance.get("kind")
    
    start_time = time.time()
    deadline = start_time + 280.0  # 20s safety margin
    
    base_seed = 42 ^ (n * 101)
    
    # Generate diverse starting configurations
    # We will use Basin Hopping for each start.
    # Number of starts: 4 (to fit in 4 threads)
    # Each start gets a budget of time/4.
    
    # Start 1: Mackay Icosahedron (if n is close to 13, 55, 147)
    rng1 = np.random.default_rng(base_seed)
    pos1 = _generate_mackay_icosahedron(n, rng1)
    
    # Start 2: Random compact
    rng2 = np.random.default_rng(base_seed + 1)
    r_est = (3.0 * n / (4.0 * np.pi)) ** (1.0 / 3.0)
    pts2 = []
    while len(pts2) < n:
        p = rng2.uniform(-1, 1, size=3)
        if np.linalg.norm(p) <= 1.0:
            pts2.append(p)
    pos2 = np.array(pts2) * r_est * 1.1 # Slightly larger to avoid overlap
    pos2 += rng2.normal(0, 0.1, size=pos2.shape)
    
    # Start 3: BCC-like
    rng3 = np.random.default_rng(base_seed + 2)
    # Simple BCC grid
    r_est_bcc = (3.0 * n / (2.0 * 4.0 * np.pi)) ** (1.0 / 3.0) * 1.2
    max_idx = int(np.ceil(r_est_bcc)) + 2
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                p1 = np.array([i, j, k])
                if np.linalg.norm(p1) <= r_est_bcc:
                    points.append(p1)
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5])
                if np.linalg.norm(p2) <= r_est_bcc:
                    points.append(p2)
    if len(points) >= n:
        points = np.array(points)
        points -= points.mean(axis=0)
        dists = np.linalg.norm(points, axis=1)
        sorted_idx = np.argsort(dists)
        pos3 = points[sorted_idx[:n]]
    else:
        # Fallback
        pos3 = rng3.uniform(-r_est_bcc, r_est_bcc, size=(n, 3))
    pos3 += rng3.normal(0, 0.1, size=pos3.shape)
    
    # Start 4: FCC-like
    rng4 = np.random.default_rng(base_seed + 3)
    r_est_fcc = (3.0 * n / (4.0 * 4.0 * np.pi)) ** (1.0 / 3.0) * 1.2
    max_idx = int(np.ceil(r_est_fcc)) + 2
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                p1 = np.array([i, j, k])
                if np.linalg.norm(p1) <= r_est_fcc:
                    points.append(p1)
                p2 = np.array([i + 0.5, j + 0.5, k])
                if np.linalg.norm(p2) <= r_est_fcc:
                    points.append(p2)
                p3 = np.array([i + 0.5, j, k + 0.5])
                if np.linalg.norm(p3) <= r_est_fcc:
                    points.append(p3)
                p4 = np.array([i, j + 0.5, k + 0.5])
                if np.linalg.norm(p4) <= r_est_fcc:
                    points.append(p4)
    if len(points) >= n:
        points = np.array(points)
        points -= points.mean(axis=0)
        dists = np.linalg.norm(points, axis=1)
        sorted_idx = np.argsort(dists)
        pos4 = points[sorted_idx[:n]]
    else:
        # Fallback
        pos4 = rng4.uniform(-r_est_fcc, r_est_fcc, size=(n, 3))
    pos4 += rng4.normal(0, 0.1, size=pos4.shape)
    
    best_e = np.inf
    best_p = None
    
    # Prepare tasks for Basin Hopping
    # Each worker will run BH for a fixed number of iterations or time.
    # Let's allocate 60 seconds per worker (4 * 60 = 240s total, leaving 40s buffer).
    # But we don't know how long relax takes.
    # Let's just run a fixed number of BH iterations, say 20-50.
    # If relax is ~0.1s, 50 iters is 5s. That's fast.
    # If relax is ~1s, 50 iters is 50s.
    # Let's use 30 iterations.
    
    tasks = [
        (pos1, kind, n, base_seed, 30, 0.5),
        (pos2, kind, n, base_seed + 1, 30, 0.5),
        (pos3, kind, n, base_seed + 2, 30, 0.5),
        (pos4, kind, n, base_seed + 3, 30, 0.5)
    ]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for t in tasks:
            if time.time() > deadline - 10.0:
                break
            futures.append(executor.submit(_bh_worker, t))
            
        for future in concurrent.futures.as_completed(futures):
            if time.time() > deadline - 5.0:
                break
            try:
                e, p = future.result()
                if e < best_e:
                    best_e = e
                    best_p = p
            except Exception:
                pass
                
    # If we have time left, run more BH iterations on the best solution
    if best_p is not None and time.time() < deadline - 20.0:
        # Run a second round of BH on the best solution found, with larger steps
        # to try to escape the local minimum of the best solution.
        rng_ref = np.random.default_rng(base_seed + 100)
        # Perturb the best solution significantly
        perturbed_best = best_p + rng_ref.normal(0, 1.0, size=best_p.shape)
        
        # Run one more BH worker on this perturbed best
        task_ref = (perturbed_best, kind, n, base_seed + 200, 20, 1.0)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_bh_worker, task_ref)
            try:
                e, p = future.result()
                if e < best_e:
                    best_e = e
                    best_p = p
            except Exception:
                pass

    if best_p is None:
        # Fallback
        best_p = _icosahedron_positions(n, np.random.default_rng(0))
        
    return np.asarray(best_p, dtype=float)

# EVOLVE-BLOCK-END

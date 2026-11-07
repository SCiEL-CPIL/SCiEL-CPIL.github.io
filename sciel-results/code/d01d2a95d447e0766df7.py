# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _icosahedron_positions(n, rng, scale=1.0):
    """
    Generate positions for an icosahedral cluster.
    For small n, this is the icosahedron (13 atoms).
    For larger n, we use Mackay icosahedra (shells).
    If n is not a magic number (13, 55, 147...), we truncate the next shell.
    """
    # 12 vertices of a regular icosahedron
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    vertices /= np.linalg.norm(vertices[0]) # Normalize to unit sphere
    
    # Mackay shells radii (relative to first shell radius 1)
    # Shell 1: 12 atoms (vertices)
    # Shell 2: 42 atoms (faces + edges of icosahedron, scaled by sqrt(3))
    # Shell 3: 92 atoms (scaled by sqrt(5))
    # Total: 1 (center) + 12 + 42 + 92 = 147
    
    positions = [np.zeros(3)] # Center atom
    
    # Shell 1
    shell1 = vertices * 1.0
    positions.extend(shell1)
    
    # Shell 2
    # Generate points on the surface of the icosahedron scaled by sqrt(3)
    # This corresponds to the centers of the triangular faces and midpoints of edges?
    # Actually, the second shell of a Mackay icosahedron consists of 42 atoms.
    # 12 face centers, 30 edge midpoints? No, 20 faces, 30 edges.
    # The standard Mackay icosahedron shell 2 has 42 atoms.
    # Let's generate them by taking the 12 vertices and projecting? 
    # Easier: The second shell atoms are at distance sqrt(3) from center.
    # They correspond to the 12 vertices of a larger icosahedron? No.
    # Let's use a simpler heuristic for non-magic numbers: 
    # Just generate a sphere of points with icosahedral symmetry where possible.
    
    # For robustness with arbitrary n (38, 55, 59, 75, 79, 80):
    # We will generate the 147-atom Mackay icosahedron and select the n closest to center.
    # This ensures high symmetry and compactness.
    
    # Generate Shell 2 and 3 points
    # Shell 2 (42 atoms):
    # 12 atoms at the centers of the 20 faces? No, that's 20.
    # 30 atoms at the midpoints of the 30 edges? 
    # 20 + 30 = 50. Too many.
    # The Mackay icosahedron 55-atom cluster is 1 + 12 + 42.
    # The 42 atoms are: 12 on the vertices of a larger icosahedron? No.
    # Let's use the explicit coordinates for the 55-atom Mackay icosahedron.
    # Or simpler: Use a spherical shell distribution with icosahedral symmetry constraints.
    
    # Given the complexity of exact Mackay shell generation for arbitrary n,
    # and the fact that n is small (< 80), we can generate a dense set of 
    # icosahedrally symmetric points and select the best n.
    
    # Points in Shell 2 (distance sqrt(3)):
    # The 42 points are:
    # - 12 points: (0, 1, phi) * sqrt(3) ? No.
    # Let's stick to the 13-atom core and add atoms on the surface of a sphere 
    # with icosahedral symmetry.
    
    # For n > 13, we add atoms on the faces and edges of the icosahedron.
    # Face centers: 20 points. Distance from center = sqrt(1 + 1/3) = 2/sqrt(3) approx 1.15?
    # Let's calculate face center distance for unit icosahedron.
    # Vertex norm = 1. Face center is average of 3 vertices.
    # v1=(0,1,phi), v2=(1,phi,0), v3=(phi,0,1)
    # fc = (v1+v2+v3)/3. Norm(fc) = ?
    # |v1+v2+v3|^2 = 3 + 3 + 3 + 2*(0*1+1*phi+phi*0 + 0*phi+1*0+phi*1 + phi*0+0*1+1*phi)
    # = 9 + 2*(2phi + 2phi + 2phi) = 9 + 12phi.
    # |fc| = sqrt(9+12phi)/3.
    
    # Instead of complex geometry, let's use a "Jittered Icosahedron" approach.
    # Start with 13 atoms.
    # If n > 13, add atoms on the surface of a sphere with radius R ~ (n/13)^(1/3).
    # Use icosahedral symmetry to place them.
    
    # Actually, the most robust simple strategy for n in [30, 80] is:
    # 1. Start with 13-atom icosahedron.
    # 2. Generate a large number of candidate positions on a sphere of radius R.
    # 3. Select n-13 positions that are furthest from existing atoms (MaxMin distance).
    # 4. This creates a "quasi-icosahedral" shell.
    
    # However, the prompt suggests embedding known minima. 
    # The 55-atom Mackay icosahedron is a known local minimum (often global for Al13 etc, but for SC it's a strong competitor).
    # Let's explicitly generate the 55-atom Mackay icosahedron.
    
    # 55-atom Mackay Icosahedron Coordinates:
    # Center: (0,0,0)
    # Shell 1 (12): Vertices of icosahedron, radius 1.
    # Shell 2 (42): 
    #   - 12 atoms at (0, 1, phi) * sqrt(3) ? No.
    #   - The 42 atoms are:
    #     12 atoms: (0, 1, phi) * (something)
    #     30 atoms: (1, phi, 0) * (something)
    #     Actually, the second shell of a Mackay icosahedron is formed by the 
    #     icosahedron of 12 atoms scaled by sqrt(3) AND the 30 atoms which are 
    #     the midpoints of the edges of the *first* shell? No.
    
    # Let's use a simpler, verified generator for icosahedral shells.
    # Reference: "Mackay icosahedra"
    # Shell 1: 12 atoms.
    # Shell 2: 42 atoms.
    #   12 atoms are at the vertices of an icosahedron scaled by sqrt(3).
    #   30 atoms are at the midpoints of the edges of an icosahedron scaled by sqrt(3)?
    #   Let's check distances.
    #   If we take the 12 vertices of the unit icosahedron and scale by sqrt(3), we get 12 atoms.
    #   The edge midpoints of the unit icosahedron, scaled by sqrt(3), give 30 atoms.
    #   Total 42.
    #   This forms the 55-atom cluster.
    
    # Let's implement this.
    
    # Get edge midpoints of the unit icosahedron
    # Edges connect vertices that are at distance 2 (in unit icosahedron with edge length 2? No, our vertices are normalized to radius 1).
    # Distance between adjacent vertices in unit icosahedron:
    # v1=(0,1,phi), v2=(1,phi,0). d^2 = 1 + (phi-1)^2 + phi^2 = 1 + phi^2 - 2phi + 1 + phi^2 = 2 + 2phi^2 - 2phi.
    # phi^2 = phi + 1. So 2 + 2(phi+1) - 2phi = 4. d=2.
    # So edges have length 2.
    
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(vertices[i] - vertices[j])
            if np.isclose(d, 2.0):
                edges.append((vertices[i] + vertices[j]) / 2.0)
                
    # Scale Shell 1 and Shell 2 by sqrt(3) for the 55-atom cluster?
    # Wait, the 55-atom cluster has shell 1 at radius 1 and shell 2 at radius sqrt(3)?
    # Let's check the ratio. In Mackay icosahedra, the shell radii are 1, sqrt(3), sqrt(5).
    # So Shell 1 (12 atoms) is at radius 1.
    # Shell 2 (42 atoms) is at radius sqrt(3).
    # The 12 atoms of Shell 2 are the vertices of the icosahedron scaled by sqrt(3).
    # The 30 atoms of Shell 2 are the edge midpoints of the icosahedron scaled by sqrt(3).
    
    shell2_vertices = vertices * np.sqrt(3)
    shell2_edges = np.array(edges) * np.sqrt(3)
    
    all_positions = np.vstack([
        np.zeros(3), # Center
        vertices,    # Shell 1 (12)
        shell2_vertices, # Shell 2 Vertices (12)
        shell2_edges     # Shell 2 Edges (30)
    ])
    
    # This gives 1 + 12 + 12 + 30 = 55 atoms.
    
    # If n < 55, we need to remove atoms.
    # If n > 55, we need to add atoms.
    
    # For n > 55, we can add a third shell (Mackay 147-atom cluster).
    # Shell 3 is at radius sqrt(5).
    # It consists of 92 atoms.
    # 20 face centers of the icosahedron scaled by sqrt(5)?
    # 60 ... ?
    # Let's generate the 147-atom cluster if n > 55.
    
    if n > 55:
        # Generate Shell 3
        # Face centers of the unit icosahedron
        faces = []
        # Find faces (triangles)
        # This is complex. Let's approximate Shell 3 by adding points on a sphere of radius sqrt(5)
        # using a Fibonacci sphere or similar, but constrained to icosahedral symmetry if possible.
        # For simplicity and robustness, let's just generate a dense set of points on the sphere
        # of radius sqrt(5) and select the ones that are furthest from existing points.
        
        # But wait, n is at most 80. So we only need up to 25 atoms from Shell 3.
        # We can generate 92 candidate points for Shell 3 and pick the best 25.
        
        # How to generate 92 points for Shell 3?
        # They are:
        # 20 face centers of the icosahedron.
        # 60 points on the edges? No.
        # 12 points on the vertices? No.
        # The 92 points are:
        # 20 face centers of the unit icosahedron, scaled by sqrt(5).
        # 60 points: midpoints of the edges of the *second* shell?
        # 12 points: vertices of the *third* shell?
        
        # Let's just use a generic "Add atoms to surface" strategy for n > 55.
        # Take the 55-atom cluster.
        # Generate many random points on a sphere of radius sqrt(5).
        # Select the (n-55) points that maximize the minimum distance to existing atoms.
        
        pass # Handled below

    # Strategy:
    # 1. If n <= 55: Select the n closest atoms from the 55-atom Mackay icosahedron.
    # 2. If n > 55: Start with 55-atom Mackay icosahedron. Add (n-55) atoms on a sphere of radius sqrt(5) 
    #    by selecting positions that maximize distance to existing atoms.
    
    if n <= 55:
        # Sort by distance to origin
        dists = np.linalg.norm(all_positions, axis=1)
        sorted_idx = np.argsort(dists)
        selected = all_positions[sorted_idx[:n]]
    else:
        selected = all_positions.copy()
        remaining = n - len(selected)
        
        # Generate candidate points on sphere of radius sqrt(5)
        # Use a Fibonacci sphere for uniform distribution
        N_cand = 200
        cand_pts = []
        for i in range(N_cand):
            z = 1 - (i / (N_cand - 1)) * 2
            radius = np.sqrt(1 - z*z)
            theta = np.pi * (1 + np.sqrt(5)) * i # Golden angle
            x = np.cos(theta) * radius
            y = np.sin(theta) * radius
            cand_pts.append([x, y, z])
        cand_pts = np.array(cand_pts) * np.sqrt(5)
        
        # Select 'remaining' points from cand_pts that are furthest from 'selected'
        new_pts = []
        for _ in range(remaining):
            min_dists = []
            for p in cand_pts:
                d = np.min(np.linalg.norm(selected - p, axis=1))
                min_dists.append(d)
            min_dists = np.array(min_dists)
            best_idx = np.argmax(min_dists)
            new_pts.append(cand_pts[best_idx])
            # Remove selected candidate to avoid duplicates
            cand_pts = np.delete(cand_pts, best_idx, axis=0)
            selected = np.vstack([selected, cand_pts[best_idx]]) # Wait, I deleted it. 
            # Correction:
            # I should add to a list and vstack at the end.
            pass
            
        # Redo the selection logic cleanly
        selected = all_positions.copy()
        new_pts = []
        cand_pts = cand_pts.copy()
        for _ in range(remaining):
            min_dists = []
            for p in cand_pts:
                d = np.min(np.linalg.norm(selected - p, axis=1))
                min_dists.append(d)
            min_dists = np.array(min_dists)
            best_idx = np.argmax(min_dists)
            new_pts.append(cand_pts[best_idx])
            cand_pts = np.delete(cand_pts, best_idx, axis=0)
            
        selected = np.vstack([selected, np.array(new_pts)])
        
    # Apply noise to break symmetry and allow relaxation to find the specific minimum
    if noise > 0:
        selected += rng.normal(0, noise, size=selected.shape)
        
    return selected

def _relax_worker(args):
    x, kind, n, seed_val = args
    try:
        e, p = relax(x, 'sutton_chen', kind)
        if not np.all(np.isfinite(p)) or not np.isfinite(e):
            return (np.inf, None)
        if n > 1:
            d = np.sqrt(np.sum((p[:, None, :] - p[None, :, :]) ** 2, axis=-1))
            np.fill_diagonal(d, np.inf)
            if d.min() < 0.35:
                return (np.inf, None)
        return (e, p)
    except Exception:
        return (np.inf, None)

def construct(instance):
    n = int(instance["n"])
    kind = instance.get("kind")
    
    start_time = time.time()
    deadline = start_time + 290.0
    
    base_seed = 42 ^ (n * 101)
    
    # Generate diverse icosahedral starts
    # 1. Low noise icosahedron
    rng1 = np.random.default_rng(base_seed)
    pos1 = _icosahedron_positions(n, rng1, noise=0.02)
    
    # 2. Medium noise icosahedron
    rng2 = np.random.default_rng(base_seed + 1)
    pos2 = _icosahedron_positions(n, rng2, noise=0.1)
    
    # 3. High noise icosahedron (to escape local minima near the icosahedron)
    rng3 = np.random.default_rng(base_seed + 2)
    pos3 = _icosahedron_positions(n, rng3, noise=0.25)
    
    # 4. Random packing (fallback/diversity)
    rng4 = np.random.default_rng(base_seed + 3)
    # Simple random packing
    r_est = 0.7 * n ** (1.0 / 3.0) * 1.5
    pts = []
    attempts = 0
    while len(pts) < n and attempts < 5000:
        attempts += 1
        p = rng4.uniform(-r_est, r_est, 3)
        if np.linalg.norm(p) > r_est:
            continue
        if pts:
            dists = np.linalg.norm(np.array(pts) - p, axis=1)
            if dists.min() < 0.5:
                continue
        pts.append(p)
    if len(pts) < n:
        pts = [list(rng4.uniform(-r_est, r_est, 3)) for _ in range(n - len(pts))]
        pts = pts[:n]
    pos4 = np.array(pts)
    
    tasks = [
        (pos1, kind, n, base_seed),
        (pos2, kind, n, base_seed + 1),
        (pos3, kind, n, base_seed + 2),
        (pos4, kind, n, base_seed + 3),
    ]
    
    best_e = np.inf
    best_p = None
    
    # Run parallel relaxations
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_relax_worker, t) for t in tasks]
        for future in concurrent.futures.as_completed(futures):
            if time.time() > deadline - 10.0:
                break
            try:
                e, p = future.result()
                if e < best_e:
                    best_e = e
                    best_p = p
            except Exception:
                pass

    # Basin Hopping on the best result if time permits
    if best_p is not None and time.time() < deadline - 30.0:
        # Perform a few basin hopping steps
        current_e = best_e
        current_p = best_p
        T = 0.5 # Initial temperature
        for step in range(20): # Max steps
            if time.time() > deadline - 10.0:
                break
            
            # Perturb
            rng_bh = np.random.default_rng(base_seed + 100 + step)
            step_size = 0.2 * (1 - step/20.0) # Annealing step size
            perturbed = current_p + rng_bh.normal(0, step_size, size=current_p.shape)
            
            # Relax
            try:
                e_new, p_new = relax(perturbed, 'sutton_chen', kind)
                if not np.all(np.isfinite(p_new)) or not np.isfinite(e_new):
                    continue
                if n > 1:
                    d = np.sqrt(np.sum((p_new[:, None, :] - p_new[None, :, :]) ** 2, axis=-1))
                    np.fill_diagonal(d, np.inf)
                    if d.min() < 0.35:
                        continue
            except:
                continue
                
            # Metropolis acceptance
            delta_e = e_new - current_e
            if delta_e < 0 or rng_bh.random() < np.exp(-delta_e / T):
                current_e = e_new
                current_p = p_new
                if current_e < best_e:
                    best_e = current_e
                    best_p = current_p
                    
            T *= 0.95 # Cool down

    if best_p is None:
        # Fallback
        best_p = np.zeros((n, 3))
        for i in range(n):
            x = (i % 10) * 0.5
            y = ((i // 10) % 10) * 0.5
            z = (i // 100) * 0.5
            best_p[i] = [x, y, z]
            
    return np.asarray(best_p, dtype=float)

# EVOLVE-BLOCK-END

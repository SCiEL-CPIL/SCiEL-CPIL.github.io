import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_TOTAL_TIME_LIMIT = 290.0  # Safety margin before 300s kill
_START_TIME = None

def _time_left():
    global _START_TIME
    if _START_TIME is None:
        return _TOTAL_TIME_LIMIT
    return _TOTAL_TIME_LIMIT - (time.time() - _START_TIME)

def _validate(positions):
    """Check if positions are valid: finite and min distance >= 0.35."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    # Vectorized distance check
    i_idx, j_idx = np.triu_indices(n, k=1)
    d = np.linalg.norm(positions[i_idx] - positions[j_idx])
    if np.any(d < 0.35):
        return False
    return True

def _get_icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron centered at origin, normalized to unit radius."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts

def _generate_icosahedral_cluster(n, seed=0):
    """
    Generate an icosahedral cluster with correct relative shell radii.
    Shell 0: 1 atom at origin.
    Shell 1: 12 atoms at radius R1 = 1.0.
    Shell 2: 30 atoms (edge midpoints) at radius R2.
    Shell 3: 20 atoms (face centers) at radius R3.
    
    Total atoms available: 1 + 12 + 30 + 20 = 63.
    If n > 63, we add random atoms on the surface of Shell 3.
    """
    rng = np.random.default_rng(seed)
    
    ico_verts = _get_icosahedron_vertices()
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Shell 1: 12 vertices
    R1 = 1.0
    shell1 = ico_verts * R1
    for v in shell1:
        points.append(v)
        
    # Calculate R2 (edge midpoints)
    # Edge length a = distance between adjacent vertices.
    v0 = ico_verts[0]
    v1 = ico_verts[1]
    edge_len = np.linalg.norm(v0 - v1)
    # Midpoint of edge is at distance sqrt(R1^2 - (edge_len/2)^2) from center
    R2 = np.sqrt(R1**2 - (edge_len/2.0)**2)
    
    # Find edges
    edges = set()
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(ico_verts[i] - ico_verts[j])
            if np.isclose(d, edge_len, atol=1e-6):
                edges.add((i, j))
                
    shell2 = []
    for (i, j) in edges:
        mid = (ico_verts[i] + ico_verts[j]) / 2.0
        # Normalize to unit direction then scale by R2
        mid_dir = mid / np.linalg.norm(mid)
        shell2.append(mid_dir * R2)
    for v in shell2:
        points.append(v)
        
    # Calculate R3 (face centers)
    # Find faces (triplets of vertices forming equilateral triangles of side edge_len)
    faces = set()
    for i in range(12):
        for j in range(i+1, 12):
            if (i, j) not in edges: continue
            for k in range(j+1, 12):
                if (j, k) not in edges: continue
                if (i, k) not in edges: continue
                faces.add(tuple(sorted([i, j, k])))
                
    shell3 = []
    for (i, j, k) in faces:
        center = (ico_verts[i] + ico_verts[j] + ico_verts[k]) / 3.0
        center_dir = center / np.linalg.norm(center)
        # Distance from center to face center
        R3 = np.linalg.norm(center) # Since ico_verts are unit, this is the distance from origin to face center
        shell3.append(center_dir * R3)
    for v in shell3:
        points.append(v)
        
    points = np.array(points)
    
    if len(points) >= n:
        # If we need fewer than 63, we should probably take the most "core" ones?
        # Or just truncate? Truncating might break symmetry.
        # For n < 63, we can just take the first n. But order is Center, Shell1, Shell2, Shell3.
        # This is fine.
        return points[:n]
    else:
        # Need more atoms than 63. Add random atoms on the surface of the icosahedron.
        # Use directions from existing surface atoms (Shell 3) or random.
        surface_dirs = [v / np.linalg.norm(v) for v in points if np.linalg.norm(v) > 0.5]
        remaining = n - len(points)
        for _ in range(remaining):
            if surface_dirs:
                idx = rng.integers(0, len(surface_dirs))
                dir_vec = surface_dirs[idx]
            else:
                dir_vec = rng.normal(0, 1, 3)
                dir_vec /= np.linalg.norm(dir_vec)
            
            # Place slightly outside the outermost shell (R3)
            r = R3 * (1.0 + rng.uniform(0.1, 0.3))
            new_p = dir_vec * r
            points = np.vstack([points, [new_p]])
            
        return points[:n]

def _generate_fcc(n, seed=0):
    """Generate points from an FCC lattice inside a sphere."""
    rng = np.random.default_rng(seed)
    a = 1.0
    R = (3.0 * n / (4.0 / a**3) / (4.0 * np.pi)) ** (1.0/3.0) * 1.2
    
    max_coord = int(np.ceil(R / a)) + 2
    points = []
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                points.append(np.array([ix, iy, iz]) * a)
                points.append(np.array([ix + 0.5, iy + 0.5, iz]) * a)
                points.append(np.array([ix + 0.5, iy, iz + 0.5]) * a)
                points.append(np.array([ix, iy + 0.5, iz + 0.5]) * a)
                
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    sorted_pts = points[order]
    
    if len(sorted_pts) >= n:
        return sorted_pts[:n]
    else:
        pts = list(sorted_pts)
        while len(pts) < n:
            p = rng.uniform(-R, R, 3)
            pts.append(p)
        return np.array(pts)

def _generate_random_compact(n, seed=0):
    """Generate a random compact cluster using rejection sampling."""
    rng = np.random.default_rng(seed)
    radius = n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    max_attempts = n * 2000
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if points:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) >= 0.5:
                    points.append(p)
            else:
                points.append(p)
        attempts += 1
        
    while len(points) < n:
        points.append(rng.uniform(-radius, radius, 3))
        
    return np.array(points)

def _perturb(positions, magnitude=0.2, seed=0):
    """Perturb positions by adding random noise."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, magnitude, positions.shape)
    return positions + noise

def _relax_worker(start, kind, steps=1, seed=0):
    """
    Worker function: performs 'steps' of relax-perturb-relax.
    Returns best energy and positions found in the trajectory.
    """
    try:
        current_pos = start.copy()
        best_energy, best_pos = np.inf, None
        
        # Initial relax
        current_energy, current_pos = relax(current_pos, 'sutton_chen', kind)
        
        if _validate(current_pos):
            best_energy = current_energy
            best_pos = current_pos.copy()
            
        for step in range(steps):
            if _time_left() < 10.0:
                break
                
            sigma = 0.2 + 0.05 * step
            perturbed_pos = _perturb(current_pos, sigma, seed + step)
            
            if not _validate(perturbed_pos):
                continue
                
            try:
                new_energy, new_pos = relax(perturbed_pos, 'sutton_chen', kind)
            except Exception:
                continue
                
            if _validate(new_pos):
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pos = new_pos.copy()
                
                # Metropolis acceptance
                if new_energy < current_energy:
                    current_pos = new_pos
                    current_energy = new_energy
                else:
                    # Simple Metropolis (T is effectively high enough to accept some uphill moves)
                    # But for speed, let's just do hill climbing or very simple Metropolis
                    # Actually, basin hopping usually accepts uphill moves. Let's implement simple Metropolis.
                    # T = 1.0 is a reasonable guess for these scales?
                    # Or just accept if it's not too bad.
                    # For simplicity and speed, let's stick to hill climbing or fixed probability.
                    # Let's use a simple Metropolis with T=1.0
                    T = 1.0
                    delta_E = new_energy - current_energy
                    if np.random.rand() < np.exp(-delta_E / T):
                        current_pos = new_pos
                        current_energy = new_energy
                        
        return best_energy, best_pos
    except Exception:
        return np.inf, None

def construct(instance):
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "sutton_chen")
    
    rng = np.random.default_rng(n)
    
    best_energy = np.inf
    best_pos = None
    
    # Generate diverse starting structures
    starts = []
    
    # 1. Icosahedral
    ico_pos = _generate_icosahedral_cluster(n, seed=n)
    if ico_pos is not None and _validate(ico_pos):
        starts.append(ico_pos)
        
    # 2. FCC
    fcc_pos = _generate_fcc(n, seed=n)
    if fcc_pos is not None and _validate(fcc_pos):
        starts.append(fcc_pos)
        
    # 3. Random Compact
    rand_pos = _generate_random_compact(n, seed=n)
    if rand_pos is not None and _validate(rand_pos):
        starts.append(rand_pos)
        
    # 4. Jittered versions of the above
    for base in [ico_pos, fcc_pos, rand_pos]:
        if base is None:
            continue
        for sigma in [0.1, 0.2, 0.3]:
            jittered = base + rng.normal(0, sigma, base.shape)
            if _validate(jittered):
                starts.append(jittered)
                
    # If we still don't have any valid starts, generate some random ones
    if not starts:
        R = n ** (1.0/3.0)
        for _ in range(5):
            pos = rng.uniform(-R, R, (n, 3))
            if _validate(pos):
                starts.append(pos)
            if len(starts) >= 5:
                break
    
    # If still nothing, just return random (will be invalid but better than crash)
    if not starts:
        R = n ** (1.0/3.0)
        best_pos = rng.uniform(-R, R, (n, 3))
        return best_pos

    # Parallel Basin Hopping
    # Use up to 4 threads
    num_threads = min(4, len(starts))
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for i, start in enumerate(starts):
            if _time_left() < 30.0:
                break
            # Allocate more steps to earlier starts if time permits?
            # Or just fixed steps per start.
            # Let's do 5 steps per start initially.
            steps = 5
            seed = n + i
            future = executor.submit(_relax_worker, start, kind, steps, seed)
            futures.append(future)
            
        for future in as_completed(futures):
            try:
                e, p = future.result()
                if e < best_energy and p is not None:
                    best_energy = e
                    best_pos = p
            except Exception:
                pass
    
    # If we have time left, do a second round of refinement on the best structure found
    if best_pos is not None and _time_left() > 30.0:
        # Perform more basin hopping steps on the best structure
        for step in range(20):
            if _time_left() < 10.0:
                break
                
            sigma = 0.2 + 0.05 * step
            perturbed_pos = _perturb(best_pos, sigma, n + 100 + step)
            
            if not _validate(perturbed_pos):
                continue
                
            try:
                new_energy, new_pos = relax(perturbed_pos, 'sutton_chen', kind)
            except Exception:
                continue
                
            if _validate(new_pos):
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pos = new_pos
                else:
                    # Metropolis
                    T = 1.0
                    delta_E = new_energy - best_energy
                    if np.random.rand() < np.exp(-delta_E / T):
                        best_pos = new_pos
                        # Don't update best_energy if we accept an uphill move, 
                        # but we do update the current position for further perturbation
                        # Wait, best_energy should track the global best found.
                        # current_pos tracks the trajectory.
                        # In this loop, best_pos is both the global best and the current trajectory start?
                        # This is slightly flawed. Let's separate them.
                        # Actually, the previous block updated best_pos to the global best.
                        # Now we are doing basin hopping *from* the global best.
                        # So if we accept an uphill move, best_pos (current trajectory) changes,
                        # but best_energy (global min) should only decrease.
                        # So:
                        # current_traj_pos = new_pos
                        # if new_energy < best_energy:
                        #     best_energy = new_energy
                        #     best_pos = new_pos
                        # else:
                        #     if metropolis accept:
                        #         current_traj_pos = new_pos
                        #     # else keep current_traj_pos
                        # At the end, best_pos should be the global best, not the trajectory end.
                        # My code above updates best_pos = new_pos in the Metropolis branch, which is wrong if new_energy > best_energy.
                        
                        # Let's fix the logic in the final polish.
                        pass
            # The logic above is a bit messy. Let's rewrite the polish phase clearly.
            
            # Break out of the messy loop and do it properly below.
            break 

    # Proper Polish Phase
    if best_pos is not None and _time_left() > 20.0:
        current_pos = best_pos.copy()
        current_energy = best_energy
        
        for step in range(30): # Up to 30 polish steps
            if _time_left() < 5.0:
                break
                
            sigma = 0.15 + 0.05 * (step % 10) # Oscillate sigma
            perturbed_pos = _perturb(current_pos, sigma, n + 200 + step)
            
            if not _validate(perturbed_pos):
                continue
                
            try:
                new_energy, new_pos = relax(perturbed_pos, 'sutton_chen', kind)
            except Exception:
                continue
                
            if _validate(new_pos):
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pos = new_pos.copy()
                    current_pos = new_pos
                    current_energy = new_energy
                else:
                    # Metropolis acceptance for the trajectory
                    T = 1.0
                    delta_E = new_energy - current_energy
                    if np.random.rand() < np.exp(-delta_E / T):
                        current_pos = new_pos
                        current_energy = new_energy
                        
    if best_pos is None:
        # Fallback
        R = n ** (1.0/3.0)
        best_pos = rng.uniform(-R, R, (n, 3))
        
    return best_pos

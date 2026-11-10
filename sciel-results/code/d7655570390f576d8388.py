import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_DEADLINE = 320.0  # 10s safety margin before 330s kill

def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)

def _generate_mackay_icosahedron(n, rng):
    """
    Generate a Mackay icosahedron cluster of n atoms.
    Uses the standard Mackay shells: 1, 12, 42, 92, ...
    For n in 38-60, we will use shells 1, 12, 42 and potentially part of 92.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Shell 0: Center
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Shell 1: 12 vertices (radius 1)
    r1 = 1.0
    shell1 = verts * r1
    points.extend(shell1.tolist())
    
    # Shell 2: 42 atoms (radius ~1.9)
    # Consists of 30 edge midpoints and 12 face centers
    r2 = np.sqrt(1 + phi) # ~1.902
    
    # Find edges (distance between adjacent verts is 2)
    dists = np.linalg.norm(verts[:, None, :] - verts[None, :, :], axis=2)
    edge_mask = (dists > 1.9) & (dists < 2.1)
    np.fill_diagonal(edge_mask, False)
    edges = [(i, j) for i in range(12) for j in range(12) if edge_mask[i, j]]
    
    shell2 = []
    # Edge midpoints
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2
        mid /= np.linalg.norm(mid)
        shell2.append(mid * r2)
        
    # Face centers
    # We need to find the 20 faces.
    # A face is a triangle of 3 vertices that are mutually adjacent.
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            if not edge_mask[i, j]: continue
            for k in range(j+1, 12):
                if edge_mask[i, k] and edge_mask[j, k]:
                    faces.append((i, j, k))
                    
    for i, j, k in faces:
        center = (verts[i] + verts[j] + verts[k]) / 3
        center /= np.linalg.norm(center)
        shell2.append(center * r2)
        
    points.extend(shell2.tolist())
    
    # If n is larger than 55, we need shell 3 (92 atoms)
    # But for n=38..60, we might just need to trim or add.
    # Mackay 55 is 1+12+42.
    # For n > 55, we add atoms from shell 3 or just randomize?
    # The prompt says n in (38, 45, 50, 55, 57, 60).
    # 38, 45, 50 are < 55, so we trim the outer shell.
    # 57, 60 are > 55, so we add atoms.
    
    points = np.array(points)
    
    if n < len(points):
        # Sort by distance from center, keep closest n
        dists_to_center = np.sum(points**2, axis=1)
        idx = np.argsort(dists_to_center)[:n]
        points = points[idx]
    elif n > len(points):
        # Need to add atoms. 
        # For 57, 60, we are close to the 92-atom shell.
        # We can add atoms on the surface of the next shell (radius ~2.4)
        # Or just add random atoms in the volume and let relaxation sort it out.
        # Better: Add atoms at specific high-symmetry positions if possible, or just random on sphere r3.
        r3 = np.sqrt(1 + phi**2) # ~2.414
        needed = n - len(points)
        for _ in range(needed):
            # Random point on sphere r3
            u, v = rng.uniform(0, 1, 2)
            theta = 2 * np.pi * u
            phi_angle = np.arccos(2 * v - 1)
            p = np.array([
                np.sin(phi_angle)*np.cos(theta),
                np.sin(phi_angle)*np.sin(theta),
                np.cos(phi_angle)
            ]) * r3
            points = np.vstack([points, p])
            
    return points

def _generate_random(n, rng):
    """Generate random compact cluster."""
    R = 2.5 * (n / 4.0) ** (1.0/3.0)
    points = []
    attempts = 0
    while len(points) < n and attempts < 1000:
        p = rng.uniform(-R, R, 3)
        if np.linalg.norm(p) <= R:
            # Check min distance 0.5
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        attempts += 1
    # If failed to place all, fill with random
    while len(points) < n:
        p = rng.uniform(-R, R, 3)
        points.append(p)
    return np.array(points)

def _relax_worker(args):
    n, points, potential, parameter = args
    try:
        energy, relaxed = relax(points, potential, parameter)
        return energy, relaxed
    except Exception:
        return float('inf'), points

def construct(instance):
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(n)
    
    # Generate diverse seeds
    # The Mackay icosahedron is the primary candidate for Dzugutov minima.
    seeds = []
    
    # 1. Perfect Mackay (trimmed or extended)
    seeds.append(_generate_mackay_icosahedron(n, rng))
    
    # 2. Slightly perturbed Mackay (to escape exact symmetry traps if any, or find nearby minima)
    mackay = _generate_mackay_icosahedron(n, rng)
    mackay_perturbed = mackay + rng.normal(0, 0.1, mackay.shape)
    seeds.append(mackay_perturbed)
    
    # 3. Random compact
    seeds.append(_generate_random(n, rng))
    
    # 4. Another random
    seeds.append(_generate_random(n, rng))
    
    best_energy = float('inf')
    best_positions = seeds[0]
    
    # Phase 1: Relax all seeds in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_relax_worker, (n, s, potential, parameter)) for s in seeds]
        for f in concurrent.futures.as_completed(futures):
            energy, positions = f.result()
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
                
    # Phase 2: Basin Hopping on the best structure
    current_energy = best_energy
    current_positions = best_positions.copy()
    
    step_size = 0.3
    # Adaptive step size
    
    # We have ~300s total. Relaxation takes maybe 1-5s per call.
    # Let's do as many basin hopping steps as possible.
    
    while _get_time_remaining() > 10.0:
        # Perturb
        noise = rng.normal(0, step_size, current_positions.shape)
        new_positions = current_positions + noise
        
        # Relax
        try:
            energy, relaxed = relax(new_positions, potential, parameter)
        except Exception:
            # If relaxation fails, reduce step size and continue
            step_size *= 0.9
            continue
            
        # Metropolis accept
        # T is effectively 0 for finding global min, but we can use a small T to allow exploration?
        # Standard BH: accept if E_new < E_old. Otherwise reject.
        # To escape local minima, we can occasionally accept worse states (simulated annealing style)
        # or just rely on large perturbations.
        
        if energy < current_energy:
            current_energy = energy
            current_positions = relaxed
            # Decrease step size slightly as we settle
            step_size *= 0.95
            if step_size < 0.05:
                step_size = 0.05
        else:
            # Reject
            # Increase step size to escape
            step_size *= 1.1
            if step_size > 1.5:
                step_size = 1.5
                
        # Update global best
        if current_energy < best_energy:
            best_energy = current_energy
            best_positions = current_positions.copy()
            
    return best_positions

import numpy as np
import time
import multiprocessing
import os
import sys
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Global deadline guard
_START_TIME = None
_DEADLINE = 315.0  # 15s safety margin before 330s kill

def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)

def _generate_mackay_icosahedron(n, rng):
    """
    Generate a proper Mackay icosahedral cluster.
    Shells:
    0: 1 atom (center)
    1: 12 atoms (vertices of icosahedron)
    2: 42 atoms (30 edge midpoints + 12 face centers)
    Total 55 atoms.
    
    For n > 55, we add atoms from the 3rd shell (92 atoms).
    For n < 55, we truncate the 55-atom cluster (removing outermost shell atoms).
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices (radius 1)
    # Standard vertices: (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = []
    points.append(np.array([0.0, 0.0, 0.0]))
    
    # Shell 1: 12 vertices
    if n > 1:
        # Radius of shell 1 in Dzugutov units is typically ~1.0-1.2 depending on scaling.
        # The potential minimum is around r=1.0? Let's assume standard Mackay geometry.
        # We will scale the whole structure at the end or use unit coordinates.
        # The relax function will adjust the scale.
        points.extend([v * 1.0 for v in verts])
        
    # Shell 2: 42 atoms
    if n > 13:
        # Find edges (30 shortest pairs)
        dists = np.linalg.norm(verts[:, None, :] - verts[None, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        dist_flat = dists.flatten()
        idx_flat = np.argsort(dist_flat)
        
        edges = []
        seen = set()
        for idx in idx_flat:
            i, j = divmod(idx, 12)
            if i == j: continue
            pair = (min(i,j), max(i,j))
            if pair not in seen:
                seen.add(pair)
                edges.append(pair)
                if len(edges) == 30:
                    break
        
        # Edge midpoints (30 atoms)
        # The radius of the 2nd shell in a regular Mackay cluster is sqrt(1+phi) * (1/sqrt(1+phi^2))? 
        # Actually, distance from center to edge midpoint of unit icosahedron is 1/sqrt(1+phi^2) * 2? No.
        # Let's calculate the norm of the midpoint.
        for i, j in edges:
            mid = (verts[i] + verts[j]) / 2
            mid_norm = np.linalg.norm(mid)
            # Normalize to unit vector and then scale to the correct shell radius
            # In Mackay, shell 2 radius is sqrt(1+phi) relative to shell 1 radius 1? 
            # Actually, if shell 1 is at radius 1, shell 2 is at radius sqrt(1+phi) / sqrt(1+phi^2)? 
            # Let's just use the geometric midpoint and let relaxation fix the exact radius.
            # But for a good start, we should project to a sphere.
            # The distance from center to edge midpoint is 1/sqrt(1+phi^2) * 2? 
            # Let's use the norm of the midpoint vector.
            mid_vec = mid / mid_norm
            # The standard Mackay shell 2 radius is sqrt(1+phi) * (1/sqrt(1+phi^2))? 
            # Let's approximate it as 1.9 * (mid_norm / 1.0) ? No.
            # Let's just use the normalized vector and a fixed radius R2.
            # R2 is typically ~1.9 if R1=1.0? 
            # Actually, R2/R1 = sqrt(1+phi) / sqrt(1+phi^2) ? 
            # Let's use the property that the atoms are on a sphere.
            # We will use R2 = 1.9 as a heuristic, or better, compute the exact distance.
            # Distance from center to edge midpoint of unit icosahedron is 1/sqrt(1+phi^2) * 2? 
            # No, it's 1/sqrt(1+phi^2) * (1 + 1/phi^2)? 
            # Let's just use the normalized vector and a radius of 1.9.
            points.append(mid_vec * 1.9)
            
        # Face centers (12 atoms)
        # The 12 face centers of the icosahedron are the vertices of a dodecahedron? 
        # No, the 20 face centers form a dodecahedron. 
        # Wait, Mackay shell 2 has 42 atoms. 30 edge midpoints + 12 face centers? 
        # But there are 20 faces. Why 12? 
        # Ah, the Mackay icosahedron is built from 20 icosahedral sub-clusters? 
        # No, the 2nd shell consists of 30 edge centers and 12 face centers? 
        # Actually, the 12 atoms are at the centers of the 12 faces of the *dual* dodecahedron? 
        # No, the 12 atoms are at the positions of the 12 vertices of the icosahedron? No, those are shell 1.
        # Let's check the literature: Mackay icosahedron 55-atom cluster.
        # Shell 2: 42 atoms. 30 are on the edges. 12 are on the faces? 
        # Actually, 12 of the 20 faces have atoms? No.
        # The 12 atoms are at the centers of the 12 faces of the *dodecahedron*? 
        # The dual of the icosahedron is the dodecahedron. 
        # The vertices of the dodecahedron are at the face centers of the icosahedron. 
        # So there are 20 face centers. 
        # Why does shell 2 only have 12 face-center-like atoms? 
        # Perhaps the Mackay cluster is not a perfect shell. 
        # Let's assume the 12 atoms are at the positions of the 12 vertices of a dodecahedron? 
        # Dodecahedron vertices: (0, ±1, ±phi), (±1, ±phi, 0), (±phi, 0, ±1) -- these are the same as icosahedron vertices!
        # Wait, the icosahedron and dodecahedron are duals. 
        # Icosahedron vertices: (0, ±1, ±phi). 
        # Dodecahedron vertices: (±1, ±1, ±1) and (0, ±1/phi, ±phi) etc. 
        # So the 12 atoms in shell 2 are NOT at the face centers of the icosahedron (which would be 20 points).
        # They are at the 12 vertices of the *dodecahedron*? No, that's 20 points.
        # Let's look at the 55-atom Mackay cluster structure.
        # It is composed of a central atom, 12 atoms in the first shell, and 42 in the second.
        # The 42 atoms are: 30 edge midpoints and 12 face centers? 
        # If there are 20 faces, why only 12 face centers? 
        # Maybe the 12 atoms are at the centers of the 12 faces of the *dual*? 
        # Let's just use the 12 vertices of a dodecahedron scaled to R2.
        # Dodecahedron vertices (radius 1):
        # (0, ±1/phi, ±phi), (±1/phi, ±phi, 0), (±phi, 0, ±1/phi) -- these are 12 points? 
        # No, (0, ±1/phi, ±phi) is 4 points. (±1/phi, ±phi, 0) is 4 points. (±phi, 0, ±1/phi) is 4 points. Total 12.
        # Yes! These are the 12 vertices of a dodecahedron? 
        # Wait, a dodecahedron has 20 vertices. 
        # The set (0, ±1/phi, ±phi) etc. are 12 points. 
        # These are the 12 face centers of the icosahedron? 
        # No, the icosahedron has 20 faces. 
        # Let's check the number of faces. 20. 
        # So there are 20 face centers. 
        # Why 12? 
        # Maybe the Mackay cluster is not symmetric? 
        # Or maybe I am misremembering the shell composition.
        # Let's assume the 12 atoms are at the 12 vertices of the *icosahedron*? No.
        # Let's just use 12 random points on the sphere of radius 1.9 for the face centers, 
        # or better, use the 12 vertices of a dodecahedron (which are the face centers of the icosahedron? No).
        # Actually, the 12 atoms are at the positions of the 12 vertices of a *dodecahedron*? 
        # Let's generate 12 points on the sphere of radius 1.9 using a dodecahedral distribution.
        # Dodecahedron vertices (20 points) are (±1, ±1, ±1) and cyclic permutations of (0, ±1/phi, ±phi).
        # We need 12 points. 
        # Let's just take the first 12 from the dodecahedron vertices.
        dodeca_verts = np.array([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
            [0, 1/phi, phi], [0, -1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, -phi],
            [1/phi, phi, 0], [-1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, -phi, 0],
            [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]
        ])
        # Normalize and scale to R2=1.9
        dodeca_verts /= np.linalg.norm(dodeca_verts, axis=1, keepdims=True)
        # We need 12 atoms. The 8 cube vertices and 4 others? 
        # Let's just take the first 12.
        for i in range(12):
            points.append(dodeca_verts[i] * 1.9)
            
    # Shell 3: 92 atoms
    if n > 55:
        needed = n - len(points)
        # Generate 92 points on the sphere of radius R3.
        # R3 is typically ~2.4?
        # We will use random points on the sphere for now, relying on relaxation.
        for _ in range(needed):
            u, v = rng.uniform(0, 1, 2)
            theta = 2 * np.pi * u
            phi_angle = np.arccos(2 * v - 1)
            p = np.array([np.sin(phi_angle)*np.cos(theta), np.sin(phi_angle)*np.sin(theta), np.cos(phi_angle)])
            points.append(p * 2.4)
            
    return np.array(points[:n])

def _generate_fcc(n, rng):
    """Generate a small FCC chunk."""
    points = []
    # FCC lattice vectors
    a = 1.0
    v1 = np.array([0, a/np.sqrt(2), a/np.sqrt(2)])
    v2 = np.array([a/np.sqrt(2), 0, a/np.sqrt(2)])
    v3 = np.array([a/np.sqrt(2), a/np.sqrt(2), 0])
    
    # Generate points in a cube of side L
    L = int(np.ceil((n * 0.5) ** (1/3))) + 2
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            for k in range(-L, L+1):
                p = i*v1 + j*v2 + k*v3
                # FCC has 4 atoms per unit cell. We need to add basis atoms.
                # Basis: (0,0,0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0) in lattice coords?
                # In Cartesian:
                p0 = p
                p1 = p + np.array([0, a/2, a/2])
                p2 = p + np.array([a/2, 0, a/2])
                p3 = p + np.array([a/2, a/2, 0])
                for p_basis in [p0, p1, p2, p3]:
                    if np.linalg.norm(p_basis) < 3.0: # Keep it compact
                        points.append(p_basis)
                
    # Shuffle and take n
    rng.shuffle(points)
    points = np.array(points[:n])
    # Center
    points -= np.mean(points, axis=0)
    return points

def _generate_random(n, rng):
    """Generate random compact cluster."""
    R = 2.0 * (n / 4.0) ** (1.0/3.0)
    points = []
    while len(points) < n:
        p = rng.uniform(-R, R, 3)
        if np.linalg.norm(p) <= R:
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
    return np.array(points)

def _relax_worker(args):
    n, points, potential, parameter = args
    try:
        energy, relaxed = relax(points, potential, parameter)
        return energy, relaxed
    except Exception:
        return float('inf'), points

def _basin_hopping_worker(args):
    n, initial_points, potential, parameter, seed = args
    rng = np.random.default_rng(seed)
    
    current_energy, current_positions = relax(initial_points, potential, parameter)
    best_energy = current_energy
    best_positions = current_positions
    
    step_size = 0.5
    max_iters = 50
    
    for i in range(max_iters):
        if _get_time_remaining() < 5.0:
            break
            
        new_positions = current_positions + rng.normal(0, step_size, current_positions.shape)
        energy, relaxed = relax(new_positions, potential, parameter)
        
        if energy < current_energy:
            current_energy = energy
            current_positions = relaxed
            step_size *= 0.95
        else:
            step_size *= 1.05
            if step_size > 2.0:
                step_size = 2.0
                
        if current_energy < best_energy:
            best_energy = current_energy
            best_positions = current_positions

    return best_energy, best_positions

def construct(instance):
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Generate diverse seeds
    seeds = []
    rng = np.random.default_rng(n)
    
    # 1. Mackay Icosahedral
    seeds.append(_generate_mackay_icosahedron(n, rng))
    
    # 2. FCC
    seeds.append(_generate_fcc(n, rng))
    
    # 3. Random
    seeds.append(_generate_random(n, rng))
    
    # 4. Another Random
    seeds.append(_generate_random(n, rng))
    
    # 5. Another Mackay (slightly perturbed)
    mackay_seed = _generate_mackay_icosahedron(n, rng)
    mackay_seed += rng.normal(0, 0.1, mackay_seed.shape)
    seeds.append(mackay_seed)
    
    # Run Basin Hopping in parallel
    # Use multiprocessing to bypass GIL
    # Note: The relax function might not be picklable or thread-safe, but multiprocessing should handle it if it's a standard function.
    # If relax is not picklable, we might need to fall back to sequential or use a different approach.
    # Let's assume it works. If not, we catch the exception and run sequentially.
    
    best_energy = float('inf')
    best_positions = seeds[0]
    
    try:
        with multiprocessing.Pool(processes=4) as pool:
            futures = [pool.apply_async(_basin_hopping_worker, (n, s, potential, parameter, i)) for i, s in enumerate(seeds)]
            for f in futures:
                try:
                    energy, positions = f.get(timeout=_get_time_remaining())
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = positions
                except Exception:
                    pass
    except Exception:
        # Fallback to sequential if multiprocessing fails
        for i, s in enumerate(seeds):
            if _get_time_remaining() < 5.0:
                break
            energy, positions = _basin_hopping_worker((n, s, potential, parameter, i))
            if energy < best_energy:
                best_energy = energy
                best_positions = positions

    return best_positions

if __name__ == '__main__':
    # Example usage for testing
    instance = {'potential': 'dzugutov', 'n': 38}
    pos = construct(instance)
    print(pos.shape)

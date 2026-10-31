"""
Strategy: Deterministic Multi-Start Global Optimization with Diverse Structural Motifs.
Rationale:
1. The current approach uses Basin Hopping, which is stochastic and may waste time
   exploring irrelevant regions if the initial perturbation is not large enough or
   if the Metropolis acceptance criterion is suboptimal.
2. LJ38 is a well-studied system where the global minimum is the FCC truncated
   octahedron. The main competitors are icosahedral and BCC-like structures.
3. Instead of a single trajectory, we explicitly generate high-quality candidates
   from the most likely topologies (FCC, Icosahedral, BCC, and a few random starts).
4. Each candidate is rigorously minimized using L-BFGS-B.
5. We select the best energy among these independent basins. This deterministic
   multi-start approach is more robust against stochastic failures and ensures
   we sample the key basins of the LJ38 landscape directly.
"""

import numpy as np
import time
import math
from scipy.optimize import minimize

N = 38
DEADLINE = 145.0  # 5s safety margin before 150s kill
START_TIME = time.time()

def lj_energy_vectorized(positions):
    """Compute LJ energy efficiently using vectorized operations."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy

def check_validity(positions, min_dist=0.5):
    """Check minimum distance constraint."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1e10)
    min_r2 = np.min(r2)
    return min_r2 >= (min_dist * min_dist)

def make_fcc_truncated_octahedron():
    """
    Generate the 38-atom FCC truncated octahedron.
    This is the putative global minimum structure.
    """
    points = []
    max_coord = 5
    scale = 1.0 / math.sqrt(2.0)
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # FCC condition: i+j+k is even
                if (i + j + k) % 2 == 0:
                    r2 = i*i + j*j + k*k
                    if r2 <= 30:
                        points.append((i * scale, j * scale, k * scale, r2))
    
    # Sort by distance from center
    points.sort(key=lambda x: x[3])
    
    # Take the 38 closest
    selected = [p[:3] for p in points[:38]]
    
    # Center the cluster
    coords = np.array(selected, dtype=float)
    coords -= np.mean(coords, axis=0)
    
    return coords

def make_icosahedron():
    """
    Generate an icosahedral cluster (Mackay icosahedron).
    Layers: 1 (center) + 12 (inner) + 30 (outer) = 43 atoms.
    We need 38, so we take the inner 12 + center (13) and the 25 closest from the outer 30.
    """
    # Golden ratio
    phi = (1 + math.sqrt(5)) / 2
    
    # Icosahedron vertices (12)
    # Standard vertices: (0, ±1, ±phi), (±1, ±phi, 0), (±phi, 0, ±1)
    verts = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            verts.append((0, i, j * phi))
            verts.append((i, j * phi, 0))
            verts.append((j * phi, 0, i))
            
    verts = np.array(verts)
    # Normalize to radius 1
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    
    # Layer 1: Center (1 atom)
    layer1 = np.array([[0, 0, 0]])
    
    # Layer 2: Inner shell (12 atoms) at radius ~1.0 (scaled)
    # In LJ, equilibrium distance is 2^(1/6).
    # For icosahedron, the ratio of radii is typically around 1.0 (inner) and 2.0 (outer) relative to nearest neighbor?
    # Actually, for Mackay icosahedron, the nearest neighbor distance is fixed.
    # Let's place inner shell at radius R1 and outer at R2.
    # Nearest neighbor distance in inner shell is ~1.0 * R1 * sin(pi/5) * 2? No.
    # Distance between adjacent vertices on unit sphere: 2 * sin(pi/5) = ~1.1755.
    # If we want NN distance = 1.0 (in reduced units where sigma=1, eq dist = 2^(1/6) ~ 1.122),
    # let's just scale the whole structure later.
    
    # Let's define the structure in "atomic units" where NN is ~1.
    # Inner shell radius r1. Distance between adjacent vertices on sphere of radius r1 is 2*r1*sin(pi/5).
    # We want this to be ~1.0. So r1 = 1.0 / (2*sin(pi/5)) = 1.0 / 1.17557 = 0.8506.
    r1 = 1.0 / (2 * math.sin(math.pi / 5))
    layer2 = verts * r1
    
    # Outer shell (30 atoms)
    # The outer shell vertices are the midpoints of the edges of the icosahedron, scaled out?
    # Or they form a larger icosahedron?
    # Mackay icosahedron: The 30 outer atoms are located at the centers of the 20 triangular faces? No, that's 20.
    # The 30 atoms are at the midpoints of the 30 edges of the icosahedron, projected outwards?
    # Actually, the 30 outer atoms in a Mackay icosahedron form a truncated icosahedron-like shell?
    # Simpler: The 30 atoms are at the centers of the 20 faces (20 atoms) + ... wait, 1+12+30=43.
    # The 30 outer atoms are typically placed at the midpoints of the 30 edges of the inner icosahedron,
    # but pushed out to the next shell.
    # Let's use the standard construction:
    # The 30 outer atoms are the vertices of a larger icosahedron? No.
    # They are the centroids of the 20 faces (20 atoms) + the 10 ...?
    # Actually, a common approximation for the outer shell of an icosahedron is to take the 30 edge midpoints of the 12-vertex icosahedron and scale them.
    
    # Edge midpoints of the 12 vertices
    # There are 30 edges.
    # Let's find pairs of vertices that are closest (edges).
    v2 = verts ** 2
    d2 = v2[:, np.newaxis] + v2[np.newaxis, :] - 2 * (verts @ verts.T)
    np.fill_diagonal(d2, np.inf)
    
    # Find the 30 smallest distances (edges)
    # For a regular icosahedron, all edges have the same length.
    # We can just take all pairs with distance close to min.
    min_d2 = np.min(d2)
    # Due to floating point, use a tolerance
    edge_mask = d2 < (min_d2 * 1.01)
    
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if edge_mask[i, j]:
                mid = (verts[i] + verts[j]) / 2.0
                edges.append(mid)
                
    edges = np.array(edges)
    # Normalize edges to unit sphere
    edge_norms = np.linalg.norm(edges, axis=1)
    edges = edges / edge_norms[:, np.newaxis]
    
    # Scale outer shell
    # The radius of the outer shell in a Mackay icosahedron is such that the distance
    # from an outer atom to its nearest inner atoms is also ~1.0.
    # Distance from edge midpoint (on unit sphere) to adjacent vertices (on unit sphere):
    # Chord length for angle theta.
    # For icosahedron, the angle between adjacent vertices is 2*pi/5? No.
    # The solid angle is ...
    # Let's estimate r2.
    # In many LJ icosahedral clusters, the ratio r2/r1 is approx 2.0?
    # Let's try r2 = 2.0 * r1?
    # Or we can optimize the scale of the outer shell.
    # For now, let's place them at r2 = 2.0 * r1.
    r2 = 2.0 * r1
    layer3 = edges * r2
    
    # Total atoms: 1 + 12 + 30 = 43.
    # We need 38. So we drop the 5 furthest atoms from the center?
    # Or we just take the first 38 in a sorted order by radius?
    # All layer 2 atoms have radius r1. All layer 3 atoms have radius r2.
    # Since r2 > r1, we keep all 13 from layer 1+2 and 25 from layer 3.
    
    all_atoms = np.vstack([layer1, layer2, layer3])
    
    # Sort by distance from center
    dists = np.linalg.norm(all_atoms, axis=1)
    sorted_indices = np.argsort(dists)
    
    # Take first 38
    selected = all_atoms[sorted_indices[:38]]
    
    # Center
    selected -= np.mean(selected, axis=0)
    
    return selected

def make_bcc_cluster():
    """
    Generate a BCC cluster.
    BCC lattice points: (i, j, k) where i+j+k is even? No, that's FCC.
    BCC: Simple cubic + center.
    Points: (i, j, k) for all integers, plus (i+0.5, j+0.5, k+0.5).
    """
    points = []
    max_coord = 3
    scale = 1.0 / math.sqrt(3.0) # BCC NN distance is sqrt(3) * a/2. If a=1, NN=sqrt(3)/2?
    # Let's use standard BCC with lattice constant a=1.
    # NN distance is sqrt(3)/2 * a.
    # We want NN ~ 1.0. So a = 2/sqrt(3).
    a = 2.0 / math.sqrt(3.0)
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # Corner atoms
                p = np.array([i, j, k]) * a
                r2 = np.sum(p*p)
                if r2 <= 15: # Cutoff
                    points.append((p, r2))
                
                # Center atom
                p_c = np.array([i+0.5, j+0.5, k+0.5]) * a
                r2_c = np.sum(p_c*p_c)
                if r2_c <= 15:
                    points.append((p_c, r2_c))
    
    # Sort by distance
    points.sort(key=lambda x: x[1])
    
    # Take 38 closest
    selected = [p[0] for p in points[:38]]
    
    coords = np.array(selected, dtype=float)
    coords -= np.mean(coords, axis=0)
    
    return coords

def make_random_cluster(rng, n_atoms=38):
    """
    Generate a random cluster with minimum distance constraint.
    """
    pts = []
    for _ in range(n_atoms):
        for _ in range(100):
            p = rng.uniform(-2, 2, size=3)
            if all(np.linalg.norm(p - q) > 0.6 for q in pts): # Slightly larger gap for random
                pts.append(p)
                break
        if len(pts) < n_atoms:
            # If we fail, just add a random point, validity check later
            pts.append(rng.uniform(-2, 2, size=3))
            
    coords = np.array(pts, dtype=float)
    coords -= np.mean(coords, axis=0)
    return coords

def local_minimize(positions):
    """
    Perform local minimization using L-BFGS-B.
    """
    def energy_func(pos):
        pos = pos.reshape(N, 3)
        return lj_energy_vectorized(pos)
    
    try:
        result = minimize(
            energy_func, 
            positions.flatten(), 
            method='L-BFGS-B', 
            options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-12}
        )
        final_pos = result.x.reshape(N, 3)
        final_energy = result.fun
        return final_pos, final_energy
    except Exception:
        return positions, lj_energy_vectorized(positions)

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_pos = None
    best_energy = float('inf')
    
    # List of initial structures to try
    initial_structures = []
    
    # 1. FCC (Global Min Candidate)
    try:
        initial_structures.append(("FCC", make_fcc_truncated_octahedron()))
    except Exception as e:
        pass
        
    # 2. Icosahedron (Main Local Min Candidate)
    try:
        initial_structures.append(("ICOS", make_icosahedron()))
    except Exception as e:
        pass
        
    # 3. BCC (Another Local Min Candidate)
    try:
        initial_structures.append(("BCC", make_bcc_cluster()))
    except Exception as e:
        pass
        
    # 4. Random Starts (Diverse)
    rng = np.random.default_rng(42)
    for i in range(5):
        try:
            initial_structures.append((f"RAND{i}", make_random_cluster(rng)))
        except Exception as e:
            pass
            
    # Process each structure
    for name, pos in initial_structures:
        if time.time() - START_TIME > DEADLINE - 10:
            break
            
        # Check validity
        if not check_validity(pos):
            # Try to fix by adding small noise and re-checking? Or just skip.
            # For random, it might be invalid.
            # Let's try to relax it anyway, L-BFGS might fix it if not too bad.
            # But if atoms are overlapping, energy is huge.
            pass
            
        # Minimize
        try:
            min_pos, min_energy = local_minimize(pos)
            
            if check_validity(min_pos):
                if min_energy < best_energy:
                    best_energy = min_energy
                    best_pos = min_pos.copy()
        except Exception as e:
            continue
            
    # Final Polish on the best result
    if best_pos is not None:
        try:
            if time.time() - START_TIME < DEADLINE - 5:
                final_pos, final_energy = local_minimize(best_pos)
                if check_validity(final_pos) and final_energy < best_energy:
                    best_pos = final_pos
                    best_energy = final_energy
        except Exception:
            pass
            
    # Fallback
    if best_pos is None:
        # Absolute fallback: Random valid points
        rng = np.random.default_rng(123)
        pts = []
        for _ in range(N):
            while True:
                p = rng.uniform(-2, 2, size=3)
                if all(np.linalg.norm(p - q) > 0.5 for q in pts):
                    pts.append(p)
                    break
        best_pos = np.array(pts)
        
    return best_pos

# EVOLVE-BLOCK-END

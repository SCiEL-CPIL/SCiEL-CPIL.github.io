import numpy as np
from scipy.optimize import minimize
import time

N = 55

def get_icosahedron_vertices():
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)
    ]
    verts = np.array(verts, dtype=float)
    # Normalize to unit length
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    return verts

def get_icosahedron_edge_midpoints(verts):
    # Find edges: pairs of vertices with minimum distance
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i, j] = np.linalg.norm(verts[i] - verts[j])
    
    # The 5th smallest distance in the sorted unique distances (excluding 0) is the edge length
    # Actually, just find the 30 pairs with the smallest non-zero distance
    pairs = []
    min_dist = np.inf
    for i in range(12):
        for j in range(i+1, 12):
            d = dists[i, j]
            if d > 1e-6: # non-zero
                pairs.append((d, i, j))
    pairs.sort(key=lambda x: x[0])
    
    # The first 30 pairs are the edges
    edge_mids = []
    for d, i, j in pairs[:30]:
        mid = (verts[i] + verts[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            mid = mid / norm
            edge_mids.append(mid)
            
    return np.array(edge_mids)

def lj_energy(positions):
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i+1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 1e-6:
                return 1e10
            r2 = r * r
            r6 = r2 * r2 * r2
            r12 = r6 * r6
            energy += 4.0 * (r12 / r12 - r6 / r6) # 4*(r^-12 - r^-6)
    return energy

def construct():
    start_time = time.time()
    
    verts = get_icosahedron_vertices()
    edge_mids = get_icosahedron_edge_midpoints(verts)
    
    # Initial guess for radii
    # r1: inner shell
    # rv2: outer vertices
    # re2: outer edge midpoints
    # Typical values: r1 ~ 1.0, rv2 ~ 2.0, re2 ~ 1.8
    x0 = np.array([1.0, 2.0, 1.8])
    
    def energy_func(params):
        r1, rv2, re2 = params
        if r1 < 0.5 or rv2 < 0.5 or re2 < 0.5:
            return 1e10
            
        positions = []
        positions.append(np.array([0.0, 0.0, 0.0]))
        
        for v in verts:
            positions.append(v * r1)
        
        for v in verts:
            positions.append(v * rv2)
            
        for e in edge_mids:
            positions.append(e * re2)
            
        positions = np.array(positions)
        
        # Check minimum distance constraint
        # This is expensive, but we can skip it in the inner loop if we trust the bounds
        # or do a quick check
        # For now, rely on bounds and the fact that LJ goes to +inf if too close
        
        return lj_energy(positions)
    
    # Optimize the 3 parameters
    res = minimize(energy_func, x0, method='L-BFGS-B', bounds=[(0.5, 5.0)] * 3, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
    
    optimal_params = res.x
    r1, rv2, re2 = optimal_params
    
    # Construct the positions
    positions = []
    positions.append(np.array([0.0, 0.0, 0.0]))
    
    for v in verts:
        positions.append(v * r1)
        
    for v in verts:
        positions.append(v * rv2)
        
    for e in edge_mids:
        positions.append(e * re2)
        
    positions = np.array(positions)
    
    # Full relaxation to polish
    # Use L-BFGS-B on all 165 coordinates
    # Flatten positions
    x_flat = positions.flatten()
    
    def energy_flat(x):
        pos = x.reshape((55, 3))
        return lj_energy(pos)
    
    # Check time
    if time.time() - start_time < 100:
        res2 = minimize(energy_flat, x_flat, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        positions = res2.x.reshape((55, 3))
        
    # Final check on minimum distance
    # If any distance is < 0.5, we might need to adjust, but LJ repulsion should prevent this
    # unless the optimizer fails.
    
    return positions

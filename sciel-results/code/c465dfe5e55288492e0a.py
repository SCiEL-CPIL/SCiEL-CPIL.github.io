import numpy as np
import time
import scipy.optimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

def _fcc_cluster(n):
    """Generate a roughly FCC-like cluster of n atoms."""
    # Simple approach: generate points in a cubic grid, filter by distance, take first n
    # Or better: use a known small FCC fragment generation
    # For simplicity and robustness, we'll use a dense random packing with a bias towards lattice-like spacing
    # Actually, let's try to generate a specific shape.
    # A simpler robust initialization for BH is just random but compact.
    # The knowledge says structural templates help.
    # Let's generate points on a 3D grid and select n closest to center.
    side = int(np.ceil(n ** (1/3)) + 2)
    grid = np.array(np.meshgrid(np.arange(-side, side+1), np.arange(-side, side+1), np.arange(-side, side+1)))
    points = np.vstack(grid.T).astype(float)
    points = points * 1.0 # spacing 1.0
    # Sort by distance to origin
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _icosahedron(n):
    """Generate an icosahedral cluster."""
    # 1 center, 12 first shell, 30 second shell, 20 third shell (total 63)
    # For n > 63, add more shells or use random
    if n <= 1:
        return np.array([[0,0,0]])
    if n <= 13:
        # 12 vertices of icosahedron
        t = (1 + np.sqrt(5)) / 2
        verts = [
            [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
            [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
            [t, 0, 1], [t, 0, -1], [-t, 0, 1], [-t, 0, -1]
        ]
        verts = np.array(verts)
        # Normalize to unit radius, then scale to spacing 1.0 (approx)
        # Distance from center to vertex is sqrt(1+t^2)
        # We want nearest neighbor distance ~ 1.0
        # In icosahedron, edge length is 2.
        # So scale factor is 0.5.
        verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * 0.5 * np.sqrt(1+t*t) # This is getting complicated.
        # Let's just use random for simplicity if n is not small, or a simple heuristic.
        pass
    
    # For general n, a simple "random compact" is often sufficient for BH to find the basin.
    # Let's stick to a robust random compact initialization to avoid bugs in specific geometry generators.
    rng = np.random.default_rng(42)
    radius = n ** (1/3) * 0.8
    pts = []
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p-q) > 0.8 for q in pts):
            pts.append(p)
    return np.array(pts)

def construct(instance):
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 45.0 # 45s per instance to be safe (300s / 6 = 50s, leave margin)
    
    best_energy = np.inf
    best_pos = None
    
    # Try a few different initializations
    inits = []
    
    # 1. Random compact
    rng = np.random.default_rng(n)
    radius = n ** (1/3) * 0.9
    pts = []
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p-q) > 0.9 for q in pts):
            pts.append(p)
    inits.append(np.array(pts))
    
    # 2. FCC-like grid
    side = int(np.ceil(n ** (1/3)) + 2)
    grid = np.array(np.meshgrid(np.arange(-side, side+1), np.arange(-side, side+1), np.arange(-side, side+1)))
    points = np.vstack(grid.T).astype(float)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    inits.append(points[idx])
    
    # 3. Another random with different seed
    rng2 = np.random.default_rng(n + 100)
    pts2 = []
    while len(pts2) < n:
        p = rng2.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p-q) > 0.9 for q in pts2):
            pts2.append(p)
    inits.append(np.array(pts2))

    for init in inits:
        if time.time() > deadline:
            break
            
        # Relax initial guess
        try:
            energy, pos = relax(init, "morse", rho)
        except Exception:
            continue
            
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
            
        # Basin Hopping / Simulated Annealing
        T = 0.5 # Initial temperature
        cooling = 0.99
        step_size = 0.2
        n_steps = 200 # Max steps per init
        
        for i in range(n_steps):
            if time.time() > deadline:
                break
                
            # Perturb
            noise = np.random.normal(0, step_size, pos.shape)
            pos_new = pos + noise
            
            # Relax
            try:
                energy_new, pos_new_relaxed = relax(pos_new, "morse", rho)
            except Exception:
                continue
                
            # Metropolis acceptance
            if energy_new < best_energy:
                best_energy = energy_new
                best_pos = pos_new_relaxed.copy()
                
            delta_e = energy_new - energy
            if delta_e < 0 or np.random.rand() < np.exp(-delta_e / T):
                energy = energy_new
                pos = pos_new_relaxed.copy()
                
            T *= cooling
            if T < 0.01:
                break
                
    return best_pos

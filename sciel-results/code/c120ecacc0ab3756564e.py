import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
DEADLINE = 145.0  # Safety margin from 150s

def _log_energy(pts):
    """
    Compute logarithmic energy.
    pts: (N, 3) array of points on the unit sphere.
    Returns: energy
    """
    n = len(pts)
    if n < 2:
        return 0.0
    
    # Compute pairwise distances
    dot = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1.0)
    
    # Check for very small distances
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    if np.any(d_sq < 1e-12):
        return float('inf')
        
    # Energy: sum_{i<j} log(1/d_ij) = -0.25 * sum_{i<j} log(dist_sq_ij)
    log_d_sq = np.log(np.maximum(dist_sq, 1e-300))
    np.fill_diagonal(log_d_sq, 0.0)
    energy = -0.25 * np.sum(log_d_sq)
    return float(energy)

def _retract(x_flat):
    """
    Project flattened vector to sphere.
    """
    pts = x_flat.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    # Handle zero norms safely
    norms = np.where(norms < 1e-15, 1.0, norms)
    pts = pts / norms
    return pts

def _subdivided_icosahedron(n_points):
    """
    Generate points on the sphere using a subdivided icosahedron.
    """
    phi = (1 + math.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 10], [0, 1, 7], [0, 7, 3],
        [0, 3, 4], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
        [7, 3, 9], [3, 4, 8], [4, 2, 6], [2, 6, 8], [6, 8, 9],
        [2, 5, 9], [2, 9, 8], [11, 4, 2], [10, 2, 6], [7, 6, 9]
    ]
    
    def _icosa_subdivide(vertices, faces, levels):
        verts_list = [v.copy() for v in vertices]
        fs = [f.copy() for f in faces]
        
        for _ in range(levels):
            new_faces = []
            mid_cache = {}
            
            def get_midpoint(v1, v2):
                idx = (min(v1, v2), max(v1, v2))
                if idx in mid_cache:
                    return mid_cache[idx]
                mid = (verts_list[v1] + verts_list[v2]) / 2.0
                mid = mid / np.linalg.norm(mid)
                verts_list.append(mid)
                mid_cache[idx] = len(verts_list) - 1
                return mid_cache[idx]
                
            for f in fs:
                v1, v2, v3 = f
                m12 = get_midpoint(v1, v2)
                m23 = get_midpoint(v2, v3)
                m31 = get_midpoint(v3, v1)
                
                new_faces.append([v1, m12, m31])
                new_faces.append([v2, m23, m12])
                new_faces.append([v3, m31, m23])
                new_faces.append([m12, m23, m31])
                
            fs = new_faces
            
        return np.array(verts_list), fs

    # Level 2 gives 162 points. We want 150.
    verts, _ = _icosa_subdivide(verts, faces, 2)
    
    if len(verts) > n_points:
        # Remove points that are closest to their nearest neighbor
        min_dists = np.zeros(len(verts))
        for i in range(len(verts)):
            dists = np.linalg.norm(verts - verts[i], axis=1)
            np.fill_diagonal(dists, np.inf)
            min_dists[i] = np.min(dists)
        
        order = np.argsort(min_dists)
        indices_to_keep = order[len(verts)-n_points:]
        verts = verts[indices_to_keep]
        
    return verts

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + math.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_uniform_sphere(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    theta = np.random.uniform(0, 2 * math.pi, n)
    phi = np.arccos(2 * np.random.uniform(0, 1, n) - 1)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])

def construct():
    start_time = time.time()
    
    best_energy = float('inf')
    best_pts = None
    
    # 1. Generate Initializations
    inits = []
    
    # Geodesic Dome (Subdivided Icosahedron)
    try:
        geo_pts = _subdivided_icosahedron(N)
        inits.append(geo_pts)
    except Exception:
        pass
        
    # Fibonacci Sphere
    inits.append(_fibonacci_sphere(N))
    
    # Random Uniform (multiple seeds)
    for seed in [42, 123, 999, 2024]:
        inits.append(_random_uniform_sphere(N, seed=seed))
        
    # 2. Optimization Loop
    # Strategy: 
    # A. Quick L-BFGS-B to get close to a local minimum (fast, gradient-based)
    # B. Nelder-Mead (derivative-free) to polish and potentially escape shallow minima
    #    NM is robust to the non-smoothness of the log potential near collisions.
    
    for i, init_pts in enumerate(inits):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break
            
        try:
            x0 = init_pts.flatten()
            
            # Phase A: L-BFGS-B
            # We use a moderate iteration count to get "good enough"
            res_lbfgs = minimize(
                lambda x: _log_energy(_retract(x)), 
                x0, 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            
            current_pts = _retract(res_lbfgs.x)
            current_energy = _log_energy(current_pts)
            
            if current_energy < best_energy and np.isfinite(current_energy):
                best_energy = current_energy
                best_pts = current_pts.copy()
                
            # Phase B: Nelder-Mead Polish
            # NM requires the function to be defined on the flat space.
            # We define it as E(retract(x)).
            # This effectively performs a local search in the ambient space,
            # but the retraction ensures we stay on the sphere.
            
            # NM can be slow, so we limit iterations and use a time check inside.
            # scipy's NM doesn't have a direct time callback, so we rely on maxiter.
            # Given N=150, 450 dimensions, NM is very expensive per iteration.
            # We must be very careful with maxiter.
            
            # Instead of full NM on 450D, which is too slow, 
            # we use a "Hybrid" approach:
            # Use COBYLA or SLSQP? No, they are constrained.
            # Let's stick to L-BFGS-B but with a "jitter" strategy?
            
            # Actually, for 450D, NM is likely too slow to be useful within 150s if run many times.
            # Let's try a different structural change:
            # **Simulated Annealing** on the sphere?
            # Or **Basin Hopping** with L-BFGS-B as the local minimizer?
            
            # Basin Hopping is a global optimization method that uses a local minimizer
            # and moves to new random configurations, accepting them based on energy.
            # This is a structural change from pure local descent.
            
            # Let's implement a simple Basin Hopping loop for the best candidate found so far
            # or for each initialization if time permits.
            
        except Exception:
            continue
            
    # If we have time, run Basin Hopping on the best result found
    if best_pts is not None and time.time() - start_time < DEADLINE - 5:
        
        def _local_minimize(x_flat):
            """
            Local minimizer for Basin Hopping.
            Uses L-BFGS-B.
            """
            res = minimize(
                lambda x: _log_energy(_retract(x)), 
                x_flat, 
                method='L-BFGS-B', 
                options={'maxiter': 50, 'ftol': 1e-12}
            )
            pts = _retract(res.x)
            e = _log_energy(pts)
            return e, pts, res.x
        
        # Simple Basin Hopping implementation
        # x_cur is the current best flattened point
        x_cur = best_pts.flatten()
        e_cur = best_energy
        
        # Parameters
        n_steps = 20 # Number of BH steps
        step_scale = 0.1 # Initial step size in ambient space
        
        # We need a random perturbation function
        def _perturb(x_flat, scale):
            noise = np.random.normal(0, scale, size=x_flat.shape)
            return x_flat + noise
            
        # Annealing schedule
        T = 1.0
        T_min = 1e-4
        decay = 0.8
        
        for step in range(n_steps):
            if time.time() - start_time > DEADLINE - 2:
                break
                
            # Perturb
            x_new = _perturb(x_cur, step_scale)
            
            # Local Minimize
            try:
                e_new, pts_new, x_new_min = _local_minimize(x_new)
            except Exception:
                continue
                
            if not np.isfinite(e_new):
                continue
                
            # Metropolis Acceptance
            delta_e = e_new - e_cur
            if delta_e < 0 or np.random.rand() < math.exp(-delta_e / T):
                # Accept
                x_cur = x_new_min
                e_cur = e_new
                best_pts = pts_new
                best_energy = e_new
                
            # Anneal
            T *= decay
            # Optionally decay step size
            step_scale *= 0.95

    # Fallback
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)

    # Final normalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    
    return best_pts

# To ensure the code runs even if scipy optimize has issues, 
# we wrap the main logic in construct. 
# The above code is the implementation.


# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import powell, minimize
from scipy.spatial.distance import pdist

def _score(points):
    """Compute R = (d_min/d_max)^2 from 120 pairwise distances."""
    d = pdist(points)
    d_min = d.min()
    d_max = d.max()
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _objective_log_ratio(params):
    """
    Objective function for minimization: log(d_max) - log(d_min).
    Minimizing this is equivalent to maximizing d_min / d_max.
    """
    pts = params.reshape(16, 2)
    d = pdist(pts)
    d_min = d.min()
    d_max = d.max()
    
    if d_min <= 0 or d_max <= 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 1e10
        
    return np.log(d_max) - np.log(d_min)

def _generate_initializations(rng):
    """Generate a list of diverse initial configurations."""
    inits = []
    n = 16
    
    # 1. Regular 16-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    p1 = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(p1.flatten())
    
    # 2. Two concentric 8-gons
    angles1 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    angles2 = np.linspace(0, 2*np.pi, 8, endpoint=False) + np.pi/8
    r1 = 1.0
    r2 = 0.5
    p2 = np.column_stack([
        np.concatenate([r1*np.cos(angles1), r2*np.cos(angles2)]),
        np.concatenate([r1*np.sin(angles1), r2*np.sin(angles2)])
    ])
    inits.append(p2.flatten())
    
    # 3. Three rings (6, 5, 5)
    a1 = np.linspace(0, 2*np.pi, 6, endpoint=False)
    a2 = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
    a3 = np.linspace(0, 2*np.pi, 5, endpoint=False)
    r1, r2, r3 = 1.0, 0.6, 0.3
    p3 = np.column_stack([
        np.concatenate([r1*np.cos(a1), r2*np.cos(a2), r3*np.cos(a3)]),
        np.concatenate([r1*np.sin(a1), r2*np.sin(a2), r3*np.sin(a3)])
    ])
    inits.append(p3.flatten())
    
    # 4. Grid 4x4
    xs = np.linspace(-1, 1, 4)
    ys = np.linspace(-1, 1, 4)
    X, Y = np.meshgrid(xs, ys)
    p4 = np.column_stack([X.flatten(), Y.flatten()])
    inits.append(p4.flatten())
    
    # 5. Random points in unit disk
    for _ in range(5):
        theta = rng.uniform(0, 2*np.pi, n)
        r = np.sqrt(rng.uniform(0, 1, n))
        p_rand = np.column_stack([r*np.cos(theta), r*np.sin(theta)])
        inits.append(p_rand.flatten())
        
    # 6. Perturbed 16-gon
    p6 = p1 + rng.normal(0, 0.1, (n, 2))
    inits.append(p6.flatten())
    
    # 7. Hexagonal packing style (subset)
    # Approximate hexagonal lattice points
    pts_hex = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i * 1.0 + (j % 2) * 0.5
            y = j * np.sqrt(3) / 2
            pts_hex.append([x, y])
    # Select 16 closest to center
    pts_hex = np.array(pts_hex)
    centers = np.linalg.norm(pts_hex, axis=1)
    idx = np.argsort(centers)[:16]
    p7 = pts_hex[idx]
    # Normalize
    p7 = p7 / np.max(np.linalg.norm(p7, axis=1))
    inits.append(p7.flatten())

    return inits

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    deadline = time.time() + 590.0  # 10s safety margin
    rng = np.random.RandomState(42)
    
    best_points = None
    best_score = 0.0
    
    inits = _generate_initializations(rng)
    
    # Add some more random initializations to fill up the list
    while len(inits) < 15:
        theta = rng.uniform(0, 2*np.pi, n)
        r = np.sqrt(rng.uniform(0.5, 1.0, n))
        p_rand = np.column_stack([r*np.cos(theta), r*np.sin(theta)])
        inits.append(p_rand.flatten())
        
    # Shuffle initializations to ensure diversity in time allocation
    rng.shuffle(inits)
    
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 30:
            break
            
        try:
            # Powell's method
            # maxiter limits the number of iterations
            # We don't have a direct time limit per call, so we rely on maxiter
            # and check time between calls.
            res = powell(
                _objective_log_ratio, 
                x0, 
                maxiter=1000, 
                disp=False,
                tol=1e-8
            )
            
            if time.time() > deadline - 10:
                # Just score the current best and break if time is tight
                pts = res.x.reshape(16, 2)
                s = _score(pts)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
                break
                
            pts = res.x.reshape(16, 2)
            s = _score(pts)
            
            if s > best_score:
                best_score = s
                best_points = pts.copy()
                
        except Exception:
            continue
            
    # Final Polish: Nelder-Mead on the best result found
    if best_points is not None and time.time() < deadline - 5:
        try:
            res_nm = minimize(
                _objective_log_ratio, 
                best_points.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            pts_nm = res_nm.x.reshape(16, 2)
            s_nm = _score(pts_nm)
            if s_nm > best_score:
                best_score = s_nm
                best_points = pts_nm.copy()
        except Exception:
            pass

    # Fallback
    if best_points is None or best_score <= 0:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    # Ensure shape and finiteness
    best_points = np.asarray(best_points, dtype=np.float64).reshape(16, 2)
    if not np.all(np.isfinite(best_points)):
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_points

# EVOLVE-BLOCK-END

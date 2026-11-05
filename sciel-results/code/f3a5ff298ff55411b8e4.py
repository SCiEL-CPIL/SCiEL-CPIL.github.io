import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s buffer for safety

    n = 8
    
    # Precompute indices for i < j
    # Not strictly necessary if using np.triu mask, but good for clarity
    # Mask is (n, n)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    def score_vec(v):
        """
        Minimize negative log of product of distances (scaled to diameter 2).
        """
        pts = v.reshape(n, 2)
        
        # Compute squared distances
        # d2[i, j] = ||p_i - p_j||^2
        d2 = np.sum(pts**2)[:, None] + np.sum(pts**2)[None, :] - 2 * (pts @ pts.T)
        d2 = np.maximum(d2, 0)
        
        # Diameter squared
        D2 = np.max(d2)
        if D2 < 1e-12:
            return 1e10
            
        # Distances
        d = np.sqrt(d2)
        
        # Extract upper triangle distances
        d_upper = d[mask]
        
        # Check for zero distances
        if np.any(d_upper < 1e-9):
            return 1e10
            
        # Log product
        log_prod = np.sum(np.log(d_upper))
        
        return -log_prod

    # Initial guess: Regular Octagon
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    init_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    init_vec = init_pts.flatten()
    
    best_score = score_vec(init_vec)
    best_vec = init_vec
    
    # 1. Differential Evolution
    # Bounds: [-2, 2] is safe for diameter 2 shapes
    bounds = [(-2, 2)] * (2 * n)
    
    try:
        de_result = differential_evolution(
            score_vec, 
            bounds, 
            maxiter=300, 
            popsize=15, 
            seed=42,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.7
        )
        if de_result.fun < best_score:
            best_score = de_result.fun
            best_vec = de_result.x
    except Exception:
        pass

    # 2. Refine with L-BFGS-B
    if time.time() < deadline:
        try:
            res = minimize(
                score_vec, 
                best_vec, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_vec = res.x
        except Exception:
            pass

    # 3. Local Refinements from Perturbed Regular Octagons
    # Generate diverse perturbations
    np.random.seed(123)
    
    perturbations = []
    
    # Radial perturbations
    for k in range(n):
        pts = init_pts.copy()
        # Move vertex k radially
        angle = angles[k]
        pts[k] += 0.2 * np.array([np.cos(angle), np.sin(angle)])
        perturbations.append(pts.flatten())
        
    # Pair radial perturbations
    for k in range(n):
        pts = init_pts.copy()
        angle1 = angles[k]
        angle2 = angles[(k+4)%n]
        pts[k] += 0.2 * np.array([np.cos(angle1), np.sin(angle1)])
        pts[(k+4)%n] -= 0.2 * np.array([np.cos(angle2), np.sin(angle2)])
        perturbations.append(pts.flatten())
        
    # Random perturbations
    for _ in range(20):
        pts = init_pts + np.random.normal(0, 0.1, (n, 2))
        perturbations.append(pts.flatten())
        
    for vec in perturbations:
        if time.time() > deadline - 2:
            break
        try:
            res = minimize(
                score_vec, 
                vec, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-10}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_vec = res.x
        except Exception:
            pass

    # Final Polish
    if time.time() < deadline - 1:
        try:
            res = minimize(
                score_vec, 
                best_vec, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_vec = res.x
        except Exception:
            pass

    best_pts = best_vec.reshape(n, 2)
    return best_pts.tolist()

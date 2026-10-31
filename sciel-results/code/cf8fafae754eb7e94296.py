"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as optimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin against 120s limit
    
    # Precompute all triples
    triples = list(combinations(range(N), 3))
    idx_i = np.array([t[0] for t in triples])
    idx_j = np.array([t[1] for t in triples])
    idx_k = np.array([t[2] for t in triples])
    
    def score(x_flat):
        """
        Returns the minimum triangle area divided by convex hull area.
        Higher is better.
        """
        x = x_flat.reshape(-1, 2)
        
        # Check for degeneracy
        try:
            hull = ConvexHull(x)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 0.0
        except Exception:
            return 0.0
            
        # Vectorized triangle area calculation
        # v1 = x[j] - x[i], v2 = x[k] - x[i]
        # area = 0.5 * |v1[0]*v2[1] - v1[1]*v2[0]|
        x_i = x[idx_i]
        x_j = x[idx_j]
        x_k = x[idx_k]
        
        v1 = x_j - x_i
        v2 = x_k - x_i
        
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        return min_area / hull_area

    def objective(x_flat):
        """
        Objective for minimization: negative score.
        """
        return -score(x_flat)
    
    # Strategy:
    # 1. Start with a symmetric initialization: 1 center, 6 inner, 6 outer (staggered).
    #    This structure is known to perform well for point packing problems.
    # 2. Use Powell's method which is robust for non-smooth objectives like min-area.
    # 3. Also try regular polygon as a fallback/alternative.
    # 4. Polish with Nelder-Mead for high precision.
    
    best_x = None
    best_score = -np.inf
    
    # Initialization 1: Symmetric "1-6-6" structure
    # 1 center, 6 inner (radius 0.4, angles 0, 60, ...), 6 outer (radius 1.0, angles 30, 90, ...)
    pts_sym = np.zeros((N, 2))
    pts_sym[0] = [0.0, 0.0]
    r_inner = 0.4
    for i in range(6):
        angle = i * np.pi / 3
        pts_sym[i+1] = [r_inner * np.cos(angle), r_inner * np.sin(angle)]
    r_outer = 1.0
    for i in range(6):
        angle = (i * np.pi / 3) + np.pi / 6
        pts_sym[i+7] = [r_outer * np.cos(angle), r_outer * np.sin(angle)]
    
    # Initialization 2: Regular polygon
    angles_reg = np.linspace(0, 2*np.pi, N, endpoint=False)
    pts_reg = np.column_stack([np.cos(angles_reg), np.sin(angles_reg)])
    
    candidates = [pts_sym.flatten(), pts_reg.flatten()]
    
    # Run Powell optimization on each candidate
    for init in candidates:
        if time.time() > deadline - 10:
            break
            
        try:
            res = optimize.minimize(
                objective, 
                init, 
                method='Powell',
                options={'maxiter': 1000, 'maxfev': 5000}
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_x = res.x.copy()
        except Exception:
            pass
        
        # Also try Nelder-Mead on the same start for comparison
        if time.time() > deadline - 10:
            break
        try:
            res_nm = optimize.minimize(
                objective, 
                init, 
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            s_nm = -res_nm.fun
            if s_nm > best_score:
                best_score = s_nm
                best_x = res_nm.x.copy()
        except Exception:
            pass
    
    # Try a few random restarts around the best solution found
    if best_x is not None:
        num_restarts = 3
        for i in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Perturb the best solution
            noise_scale = 0.1 / (i + 1)
            x_perturbed = best_x + np.random.randn(N*2) * noise_scale
            
            try:
                res = optimize.minimize(
                    objective, 
                    x_perturbed, 
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
                )
                s = -res.fun
                if s > best_score:
                    best_score = s
                    best_x = res.x.copy()
            except Exception:
                pass
    
    # Final polish with tighter tolerances
    if time.time() < deadline - 5 and best_x is not None:
        try:
            res = optimize.minimize(
                objective, 
                best_x, 
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_x = res.x.copy()
        except Exception:
            pass
            
    if best_x is None:
        # Fallback to regular polygon
        best_x = pts_reg.flatten()
        
    return best_x.reshape(-1, 2)

# EVOLVE-BLOCK-END

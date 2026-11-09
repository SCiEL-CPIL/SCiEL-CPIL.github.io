"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from scipy.optimize import minimize, differential_evolution

def _score(points):
    """Compute Delta/n^n where Delta is product of distances over ordered pairs,
    after scaling to diameter 2."""
    pts = np.array(points, dtype=np.float64)
    n = len(pts)
    # Compute all pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    # Diameter = max distance
    dmax = np.max(dist)
    if dmax < 1e-15:
        return -np.inf
    # Scale so diameter = 2
    dist = dist / dmax * 2.0
    # Product over ordered pairs (i != j)
    # Delta = product of dist[i,j] for i != j
    mask = ~np.eye(n, dtype=bool)
    log_delta = np.sum(np.log(dist[mask]))
    score = log_delta / (n ** n)
    return score

def _neg_score(points):
    """Negative score for minimization."""
    s = _score(points)
    if s == -np.inf:
        return 1e10
    return -s

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s limit
    
    # Regular octagon as baseline
    reg_oct = [[math.cos(2 * math.pi * k / 8), math.sin(2 * math.pi * k / 8)] for k in range(8)]
    best_score = _score(reg_oct)
    best_points = reg_oct
    
    # We search in a 16-dimensional space (8 points in 2D)
    # Parameterization: use angles and radii, or just raw coordinates
    # Let's use raw coordinates for simplicity, but constrain to reasonable range
    
    n = 8
    dim = 2 * n  # 16 dimensions
    
    # Initial guess: regular octagon flattened
    x0 = np.array([p[0] for p in reg_oct] + [p[1] for p in reg_oct])
    
    # Define objective
    def objective(x):
        pts = np.array([[x[2*i], x[2*i+1]] for i in range(n)])
        return _neg_score(pts)
    
    # Try differential evolution for global search
    # Bounds: coordinates in [-10, 10]
    bounds = [(-10.0, 10.0)] * dim
    
    # Use a subset of DE for speed
    try:
        result_de = differential_evolution(
            objective, bounds,
            maxiter=50, popsize=15,
            seed=42, tol=1e-6,
            polish=False  # We'll polish separately
        )
        elapsed = time.time() - start_time
        if elapsed < DEADLINE:
            pts_de = np.array([[result_de.x[2*i], result_de.x[2*i+1]] for i in range(n)])
            score_de = _score(pts_de)
            if score_de > best_score:
                best_score = score_de
                best_points = pts_de.tolist()
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed >= DEADLINE:
        return best_points
    
    # Local polish with L-BFGS-B
    def polish(x0, method='L-BFGS-B'):
        try:
            res = minimize(
                objective, x0, method=method,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            return res.x, res.fun
        except Exception:
            return x0, objective(x0)
    
    # Polish from current best
    x_best = np.array([p[0] for p in best_points] + [p[1] for p in best_points])
    x_polished, fun_polished = polish(x_best)
    pts_polished = np.array([[x_polished[2*i], x_polished[2*i+1]] for i in range(n)])
    score_polished = _score(pts_polished)
    if score_polished > best_score:
        best_score = score_polished
        best_points = pts_polished.tolist()
    
    elapsed = time.time() - start_time
    if elapsed >= DEADLINE:
        return best_points
    
    # Try Nelder-Mead for further refinement (handles non-smoothness better)
    x_nm = np.array([p[0] for p in best_points] + [p[1] for p in best_points])
    def nm_obj(x):
        return _neg_score([[x[2*i], x[2*i+1]] for i in range(n)])
    
    try:
        res_nm = minimize(nm_obj, x_nm, method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
        elapsed = time.time() - start_time
        if elapsed < DEADLINE:
            pts_nm = np.array([[res_nm.x[2*i], res_nm.x[2*i+1]] for i in range(n)])
            score_nm = _score(pts_nm)
            if score_nm > best_score:
                best_score = score_nm
                best_points = pts_nm.tolist()
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed >= DEADLINE:
        return best_points
    
    # Try multiple random restarts with local optimization
    np.random.seed(123)
    for restart in range(10):
        elapsed = time.time() - start_time
        if elapsed >= DEADLINE:
            break
        
        # Random initialization near the regular octagon
        angles = 2 * math.pi * np.arange(n) / n + np.random.normal(0, 0.1, n)
        radii = 1.0 + np.random.normal(0, 0.1, n)
        x_rand = np.zeros(dim)
        for i in range(n):
            x_rand[2*i] = radii[i] * math.cos(angles[i])
            x_rand[2*i+1] = radii[i] * math.sin(angles[i])
        
        x_p, fun_p = polish(x_rand)
        pts_p = np.array([[x_p[2*i], x_p[2*i+1]] for i in range(n)])
        score_p = _score(pts_p)
        if score_p > best_score:
            best_score = score_p
            best_points = pts_p.tolist()
        
        # Nelder-Mead from this point
        try:
            res_nm2 = minimize(nm_obj, x_p, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            elapsed = time.time() - start_time
            if elapsed < DEADLINE:
                pts_nm2 = np.array([[res_nm2.x[2*i], res_nm2.x[2*i+1]] for i in range(n)])
                score_nm2 = _score(pts_nm2)
                if score_nm2 > best_score:
                    best_score = score_nm2
                    best_points = pts_nm2.tolist()
        except Exception:
            pass
    
    # Final polish
    x_final = np.array([p[0] for p in best_points] + [p[1] for p in best_points])
    x_final_polished, _ = polish(x_final)
    pts_final = np.array([[x_final_polished[2*i], x_final_polished[2*i+1]] for i in range(n)])
    score_final = _score(pts_final)
    if score_final > best_score:
        best_score = score_final
        best_points = pts_final.tolist()
    
    # Ensure distinct points and valid coordinates
    result = []
    for p in best_points:
        x, y = float(p[0]), float(p[1])
        if abs(x) >= 1e6 or abs(y) >= 1e6:
            x = max(-1e5, min(1e5, x))
            y = max(-1e5, min(1e5, y))
        result.append([x, y])
    
    # Check distinctness
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if abs(result[i][0] - result[j][0]) < 1e-10 and abs(result[i][1] - result[j][1]) < 1e-10:
                result[j][0] += 1e-8
    
    return result
# EVOLVE-BLOCK-END

import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _log_score_vec(P):
    """Compute log of score = log(product of all ordered-pair distances) - 28*log(diameter).
    P is (8,2) array. Returns log score (higher is better)."""
    n = 8
    # Compute all pairwise squared distances
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    # Zero out diagonal
    np.fill_diagonal(d2, 1.0)
    # Log of product of distances = sum of log(d) over ordered pairs (i != j)
    # That's 2 * sum over i<j of log(d_ij)
    # log(d_ij) = 0.5 * log(d2_ij)
    # So sum over ordered pairs = 2 * sum_{i<j} 0.5*log(d2_ij) = sum_{i<j} log(d2_ij)
    iu = np.triu_indices(n, 1)
    log_prod_ordered = np.sum(np.log(d2[iu]))  # this equals log of product over ordered pairs
    D2 = np.max(d2)
    if D2 <= 0:
        return -np.inf
    # Score = Delta / n^n where Delta = product of distances over ordered pairs after scaling to diameter 2
    # After scaling to diameter 2, each distance d becomes d/D * 2
    # Product scales by (2/D)^{n(n-1)} = (2/D)^56
    # So log_score = log_prod_ordered + 56*(log(2) - log(D))
    # = log_prod_ordered + 56*log(2) - 56*log(D)
    # But wait: the problem says "product over ordered pairs of distances after scaling the set to diameter 2"
    # Ordered pairs: n*(n-1) = 8*7 = 56 pairs
    # Each distance d_ij -> d_ij / D * 2
    # Product -> prod(d_ij) * (2/D)^56
    # log -> log_prod + 56*log(2) - 56*log(D)
    # Score = that product / n^n = product / 8^8
    # log_score = log_prod + 56*log(2) - 56*log(D) - 8*log(8)
    # For regular octagon on unit circle: D=2, all distances known
    # Let's just compute: log_score = log_prod_ordered + 56*(log(2) - log(sqrt(D2))) - 8*log(8)
    log_D = 0.5 * np.log(D2)
    log_score = log_prod_ordered + 56.0 * (np.log(2.0) - log_D) - 8.0 * np.log(8.0)
    return log_score

def _score(P):
    """Return the actual score (not log)."""
    ls = _log_score_vec(P)
    if ls == -np.inf:
        return 0.0
    return np.exp(ls)

def _two_squares_points(r1, r2, theta1, theta2, cx, cy):
    """Generate 8 points: 4 on circle of radius r1 rotated by theta1, 4 on circle of radius r2 rotated by theta2, centered at (cx, cy)."""
    angles1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2]) + theta1
    angles2 = np.array([0, np.pi/2, np.pi, 3*np.pi/2]) + theta2
    P1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    P2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
    P = np.vstack([P1, P2])
    P[:, 0] += cx
    P[:, 1] += cy
    return P

def _obj_manifold(params):
    """Objective for 5-parameter manifold: minimize negative log score."""
    r1, r2, theta1, theta2, cx, cy = params
    if r1 < 0 or r2 < 0:
        return 1e10
    P = _two_squares_points(r1, r2, theta1, theta2, cx, cy)
    ls = _log_score_vec(P)
    if ls == -np.inf:
        return 1e10
    return -ls

def _obj_full(P_flat):
    """Objective for full 16D: minimize negative log score."""
    P = P_flat.reshape(8, 2)
    ls = _log_score_vec(P)
    if ls == -np.inf:
        return 1e10
    return -ls

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 150s limit, keep 10s margin
    
    np.random.seed(42)
    
    best_log_score = -np.inf
    best_P = None
    
    # Phase 1: Differential evolution on 5-parameter manifold
    # Bounds: r in [0.1, 5], theta in [0, 2*pi], center in [-3, 3]
    bounds = [(0.1, 5.0), (0.1, 5.0), (0.0, 2*np.pi), (0.0, 2*np.pi), (-3.0, 3.0), (-3.0, 3.0)]
    
    try:
        de_res = differential_evolution(
            _obj_manifold, bounds, 
            maxiter=200, popsize=20, tol=1e-8, 
            mutation=(0.5, 1.5), recombination=0.8,
            seed=42, polish=True, init='latinhypercube'
        )
        if de_res.fun < 1e9:
            ls = -de_res.fun
            if ls > best_log_score:
                best_log_score = ls
                P_man = _two_squares_points(*de_res.x)
                best_P = P_man.copy()
    except Exception:
        pass
    
    # Phase 2: Full 16D local polish from best manifold result
    if best_P is not None and time.time() < deadline - 20:
        x0 = best_P.flatten()
        try:
            res = minimize(_obj_full, x0, method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < 1e9:
                ls = -res.fun
                if ls > best_log_score:
                    best_log_score = ls
                    best_P = res.x.reshape(8, 2).copy()
        except Exception:
            pass
        
        # Powell method as second polish
        if time.time() < deadline - 15:
            try:
                res2 = minimize(_obj_full, best_P.flatten(), method='Powell',
                                options={'maxiter': 5000, 'xtol': 1e-10, 'ftol': 1e-10})
                if res2.fun < 1e9:
                    ls = -res2.fun
                    if ls > best_log_score:
                        best_log_score = ls
                        best_P = res2.x.reshape(8, 2).copy()
            except Exception:
                pass
    
    # Phase 3: Additional random restarts on full 16D if time permits
    if time.time() < deadline - 20:
        for restart in range(10):
            if time.time() > deadline - 15:
                break
            # Random starting point near the current best
            if best_P is not None:
                noise = np.random.normal(0, 0.05, size=16)
                x0 = best_P.flatten() + noise
            else:
                x0 = np.random.uniform(-1, 1, size=16)
            
            try:
                res = minimize(_obj_full, x0, method='Nelder-Mead',
                               options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
                if res.fun < 1e9:
                    ls = -res.fun
                    if ls > best_log_score:
                        best_log_score = ls
                        best_P = res.x.reshape(8, 2).copy()
            except Exception:
                pass
    
    # Final: if no valid result, return regular octagon
    if best_P is None:
        angles = [2 * np.pi * k / 8 for k in range(8)]
        best_P = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    # Ensure distinctness: check minimum distance
    P = best_P
    min_d = np.inf
    for i in range(8):
        for j in range(i+1, 8):
            d = np.linalg.norm(P[i] - P[j])
            if d < min_d:
                min_d = d
    if min_d < 1e-8:
        # Nudge closest pair
        for i in range(8):
            for j in range(i+1, 8):
                d = np.linalg.norm(P[i] - P[j])
                if d < 1e-8:
                    P[j] += np.array([1e-6, 0])
    
    # Scale to ensure coordinates are within bounds (|coord| < 1e6)
    max_coord = np.max(np.abs(P))
    if max_coord > 1e5:
        P = P / max_coord * 1e5
    
    return P.tolist()

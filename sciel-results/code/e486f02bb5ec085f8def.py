import numpy as np
from scipy.optimize import minimize
import itertools
import time

N = 11
S3 = np.sqrt(3.0)
H = S3 / 2.0  # height of unit equilateral triangle

# Precompute all triples
TRIPLES = list(itertools.combinations(range(N), 3))
NTRIPLES = len(TRIPLES)

def _area_from_points(P):
    """Compute areas of all triangles from N points (N x 2 array)."""
    areas = np.empty(NTRIPLES)
    for idx, (i, j, k) in enumerate(TRIPLES):
        pi, pj, pk = P[i], P[j], P[k]
        cross = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
        areas[idx] = 0.5 * abs(cross)
    return areas

def _min_area(P):
    """Exact minimum triangle area."""
    return _area_from_points(P).min()

def _soft_min_area(P, temperature=0.01):
    """Smooth surrogate: soft-min of triangle areas via log-sum-exp."""
    areas = _area_from_points(P)
    # soft min = -t * log(sum(exp(-a_i/t)))
    scaled = -areas / temperature
    max_s = scaled.max()
    log_sum = max_s + np.log(np.exp(scaled - max_s).sum())
    return -temperature * log_sum

def _soft_min_area_grad(P, temperature=0.01):
    """Gradient of soft-min area w.r.t. point positions."""
    areas = _area_from_points(P)
    scaled = -areas / temperature
    max_s = scaled.max()
    exps = np.exp(scaled - max_s)
    weights = exps / exps.sum()  # softmax weights
    
    grad = np.zeros((N, 2))
    for idx, (i, j, k) in enumerate(TRIPLES):
        pi, pj, pk = P[i], P[j], P[k]
        # d(area)/d(pi), d(area)/d(pj), d(area)/d(pk)
        # area = 0.5 * |cross|, for non-degenerate, d(area)/dp = 0.5 * sign(cross) * perp
        cross = (pj[0] - pi[0]) * (pk[1] - pi[1]) - (pj[1] - pi[1]) * (pk[0] - pi[0])
        sign = 1.0 if cross >= 0 else -1.0
        # d(cross)/dpi = -(pj-pk), d(cross)/dpj = (pi-pk), d(cross)/dpk = (pj-pi)
        dpi = -0.5 * sign * np.array([pj[0] - pk[0], pj[1] - pk[1]])
        dpj = 0.5 * sign * np.array([pi[0] - pk[0], pi[1] - pk[1]])
        dpk = 0.5 * sign * np.array([pj[0] - pi[0], pj[1] - pi[1]])
        
        # Chain rule: d(soft_min)/dp = sum_k weights_k * d(-area_k/t)/dp * ... 
        # Actually soft_min = -t * log(sum(exp(-a_k/t)))
        # d(soft_min)/dp = -t * (1/sum(exp)) * sum(exp(-a_k/t) * (-1/t) * da_k/dp)
        # = (sum(exp(-a_k/t) * da_k/dp)) / sum(exp(-a_k/t))
        # = sum(weights_k * da_k/dp)
        w = weights[idx]
        grad[i] += w * dpi
        grad[j] += w * dpj
        grad[k] += w * dpk
    return grad

def _inside_triangle(P, tol=1e-9):
    """Check if all points are inside or on the equilateral triangle."""
    for x, y in P:
        if y < -tol:
            return False
        if y > S3 * x + tol:
            return False
        if y > -S3 * (x - 1.0) + tol:
            return False
    return True

def _project_to_triangle(P):
    """Project points to be inside the triangle."""
    P = P.copy()
    for i in range(N):
        x, y = P[i]
        # Project to closest point in triangle
        # Simple: clip y >= 0, y <= min(sqrt(3)*x, -sqrt(3)*(x-1))
        y_max = min(S3 * x, -S3 * (x - 1.0))
        if y < 0:
            y = 0
        if y > y_max:
            y = y_max
        P[i] = [x, y]
    return P

def _random_point_in_triangle(rng):
    """Generate random point in triangle."""
    u = rng.random()
    v = rng.random()
    if u + v > 1.0:
        u = 1.0 - u
        v = 1.0 - v
    x = u + 0.5 * v
    y = H * v
    return [x, y]

def construct():
    deadline = time.time() + 140.0  # 10s safety margin
    rng = np.random.RandomState(42)
    
    best_P = None
    best_val = 0.0
    
    # Phase 1: Generate diverse starting points
    starts = []
    
    # Grid-based starts
    for gx in np.linspace(0.1, 0.9, 3):
        for gy in np.linspace(0.1, 0.8, 3):
            P = np.zeros((N, 2))
            for i in range(N):
                # Distribute points roughly
                xi = 0.1 + 0.8 * ((i * 7 + int(gx*10) + int(gy*10)) % N) / N
                yi = 0.1 + 0.7 * ((i * 13 + int(gx*10) * 3 + int(gy*10) * 5) % N) / N
                # Ensure inside
                y_max = min(S3 * xi, -S3 * (xi - 1.0))
                yi = min(yi, y_max * 0.95)
                yi = max(yi, 0.01)
                P[i] = [xi, yi]
            starts.append(P)
    
    # Random starts
    for _ in range(20):
        P = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        starts.append(P)
    
    # Phase 2: For each start, do soft-min optimization then Nelder-Mead polish
    for start_idx, P0 in enumerate(starts):
        if time.time() > deadline - 30:
            break
        
        # Step 1: Optimize soft-min with L-BFGS-B
        try:
            def obj(x):
                P = x.reshape(N, 2)
                return _soft_min_area(P, temperature=0.005)
            
            def obj_grad(x):
                P = x.reshape(N, 2)
                return _soft_min_area_grad(P, temperature=0.005).ravel()
            
            res = minimize(obj, P0.ravel(), jac=obj_grad, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            P_soft = res.x.reshape(N, 2)
            P_soft = _project_to_triangle(P_soft)
            
            if not _inside_triangle(P_soft, tol=1e-6):
                P_soft = _project_to_triangle(P_soft)
            
            val_soft = _min_area(P_soft)
            
            # Step 2: Nelder-Mead polish on exact min-area
            def exact_obj(x):
                P = x.reshape(N, 2)
                return -_min_area(P)
            
            res2 = minimize(exact_obj, P_soft.ravel(), method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
            P_nm = res2.x.reshape(N, 2)
            P_nm = _project_to_triangle(P_nm)
            
            if _inside_triangle(P_nm, tol=1e-6):
                val_nm = _min_area(P_nm)
                if val_nm > best_val:
                    best_val = val_nm
                    best_P = P_nm.copy()
            else:
                if val_soft > best_val:
                    best_val = val_soft
                    best_P = P_soft.copy()
                    
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish on the best solution
    if best_P is not None and time.time() < deadline - 5:
        # Multiple Nelder-Mead passes with tighter tolerances
        for polish_round in range(3):
            if time.time() > deadline - 2:
                break
            try:
                def exact_obj(x):
                    P = x.reshape(N, 2)
                    return -_min_area(P)
                
                res3 = minimize(exact_obj, best_P.ravel(), method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
                P_polished = res3.x.reshape(N, 2)
                P_polished = _project_to_triangle(P_polished)
                
                if _inside_triangle(P_polished, tol=1e-8):
                    val_polished = _min_area(P_polished)
                    if val_polished > best_val:
                        best_val = val_polished
                        best_P = P_polished.copy()
            except Exception:
                break
    
    if best_P is None:
        # Fallback: random points
        best_P = np.array([_random_point_in_triangle(rng) for _ in range(N)])
    
    # Final safety check
    if not _inside_triangle(best_P, tol=1e-6):
        best_P = _project_to_triangle(best_P)
    
    return best_P

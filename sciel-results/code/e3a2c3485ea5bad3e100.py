import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11
H = np.sqrt(3) / 2

# Pre-compute indices for combinations of 3 from 11
_tri_indices = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])

# Triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, H])

def _all_areas_cart(pts):
    """Vectorized computation of all 165 triangle areas from cartesian points."""
    pi = pts[_tri_indices[:, 0]]
    pj = pts[_tri_indices[:, 1]]
    pk = pts[_tri_indices[:, 2]]
    
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    
    return 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)

def _min_area_cart(pts):
    """Exact minimum area from cartesian points."""
    return float(np.min(_all_areas_cart(pts)))

def _clamp_to_triangle(p):
    """Project a point onto the interior of the equilateral triangle."""
    x, y = p
    c = 2 * y / np.sqrt(3)
    b = x - 0.5 * c
    a = 1 - b - c
    
    eps = 1e-12
    a = max(eps, min(1 - 2*eps, a))
    b = max(eps, min(1 - 2*eps, b))
    c = max(eps, min(1 - 2*eps, c))
    
    s = a + b + c
    a /= s; b /= s; c /= s
    
    return a * V0 + b * V1 + c * V2

def _clamp_all(pts):
    """Clamp all points to be inside the triangle."""
    return np.array([_clamp_to_triangle(p) for p in pts])

def _soft_min_area_cart(pts, beta=100.0):
    """
    Smooth approximation of min area for gradient-based optimization.
    """
    areas = _all_areas_cart(pts)
    m = np.min(areas)
    # min(a) = m - 1/beta * log(sum(exp(-beta * (a - m))))
    shifted = -beta * (areas - m)
    shifted = np.clip(shifted, -500, 500)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return m - log_sum / beta

def _objective_smooth_cart(flat_pts, beta=100.0):
    """Objective to minimize: negative soft-min area."""
    pts = flat_pts.reshape(-1, 2)
    pts = _clamp_all(pts)
    return -_soft_min_area_cart(pts, beta)

def _objective_hard_cart(flat_pts):
    """Objective to minimize: negative exact min area."""
    pts = flat_pts.reshape(-1, 2)
    pts = _clamp_all(pts)
    return -_min_area_cart(pts)

def _init_structured():
    """
    Structured initialization leveraging symmetry and key geometric points.
    """
    pts = []
    pts.append(V0.copy())
    pts.append(V1.copy())
    pts.append(V2.copy())
    
    pts.append((V0 + V1) / 2)
    pts.append((V1 + V2) / 2)
    pts.append((V2 + V0) / 2)
    
    pts.append((V0 + V1 + V2) / 3)
    
    cx, cy = 1/3, H/3
    r = 0.12
    for angle in [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]:
        px = cx + r * np.cos(angle)
        py = cy + r * np.sin(angle)
        pts.append([px, py])
        
    return np.array(pts)

def _init_random(seed):
    """Random initialization inside triangle."""
    rng = np.random.default_rng(seed)
    pts = []
    for _ in range(N):
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        p = r1 * V1 + r2 * V2
        pts.append(p)
    return np.array(pts)

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We aim to finish by 145s.
    deadline = start_time + 145.0
    
    def time_left():
        return deadline - time.time()
    
    best_pts = None
    best_score = -1.0
    
    # Phase 1: Global Search using Differential Evolution
    # DE is robust for non-convex problems and can escape local minima better than simple multi-start local search.
    # We optimize the smooth surrogate objective to allow DE to navigate the landscape effectively.
    
    if time_left() > 60.0:
        try:
            # Bounds for cartesian coordinates
            # x in [0, 1], y in [0, H]
            # DE handles bounds natively, so we don't need to clamp inside the objective for DE,
            # but we must ensure the objective handles points slightly outside if DE pushes them there?
            # Actually, DE respects bounds strictly. So we can just use the raw objective without clamping 
            # IF we trust DE to stay in bounds. However, our _objective_smooth_cart clamps.
            # Let's define a specific objective for DE that doesn't clamp to avoid masking boundary effects?
            # No, clamping is safe. If DE proposes (1.1, 0), it's clamped to boundary. Gradient is 0 there.
            # This is fine.
            
            bounds = [(0, 1), (0, H)] * N
            
            # Use a moderate beta for DE to balance smoothness and accuracy
            beta_de = 50.0
            
            # DE parameters
            # popsize=15 means 15*22=330 individuals? No, popsize is multiplier for N variables.
            # N=22 variables. popsize=15 -> 330 individuals. 
            # maxiter=50 -> 50 generations.
            # Total evals ~ 330 * 50 = 16500. 
            # Each eval is ~165 area calcs + clamp. Fast.
            
            de_result = differential_evolution(
                _objective_smooth_cart,
                bounds=bounds,
                args=(beta_de,),
                seed=42,
                maxiter=50,
                tol=1e-6,
                popsize=15,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False,
                init='sobol',
                updating='immediate'
            )
            
            if de_result.success or -de_result.fun > best_score:
                de_pts = _clamp_all(de_result.x.reshape(-1, 2))
                score = _min_area_cart(de_pts)
                if score > best_score:
                    best_score = score
                    best_pts = de_pts.copy()
        except Exception:
            pass
    
    # If DE failed or didn't find anything, fallback to structured init
    if best_pts is None:
        best_pts = _init_structured()
        best_score = _min_area_cart(best_pts)
    
    # Phase 2: Local Refinement using L-BFGS-B with Annealing Beta
    # Take the best solution from DE (or structured) and refine it with gradient-based methods.
    # We use a schedule of increasing beta to sharpen the approximation of the min function.
    
    current_pts = best_pts.copy()
    betas = [50.0, 100.0, 200.0, 500.0]
    
    for beta in betas:
        if time_left() < 10.0:
            break
        
        try:
            x0 = current_pts.flatten()
            bounds = [(0, 1), (0, H)] * N
            
            res = minimize(
                _objective_smooth_cart, 
                x0, 
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            opt_pts = _clamp_all(res.x.reshape(-1, 2))
            score = _min_area_cart(opt_pts)
            
            if score > best_score:
                best_score = score
                best_pts = opt_pts.copy()
                current_pts = opt_pts.copy()
            else:
                # If it didn't improve, keep the previous best as current for next beta?
                # Or use the optimizer result anyway to see if next beta helps?
                # Usually, if it didn't improve, the optimizer got stuck or the landscape changed.
                # Let's stick with the best known.
                pass
                
        except Exception:
            pass
    
    # Phase 3: Nelder-Mead Polish on Hard Objective
    # NM is derivative-free and handles the non-differentiability of min() well.
    
    if time_left() > 10.0:
        try:
            res = minimize(
                _objective_hard_cart,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            nm_pts = _clamp_all(res.x.reshape(-1, 2))
            nm_score = _min_area_cart(nm_pts)
            
            if nm_score > best_score:
                best_score = nm_score
                best_pts = nm_pts.copy()
        except Exception:
            pass
        
        # Try NM on a few perturbed versions to escape local minima
        rng = np.random.default_rng(123)
        for _ in range(5):
            if time_left() < 5.0:
                break
            
            perturbed = best_pts + rng.normal(0, 0.02, size=(N, 2))
            perturbed = _clamp_all(perturbed)
            
            try:
                res = minimize(
                    _objective_hard_cart,
                    perturbed.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                nm_pts = _clamp_all(res.x.reshape(-1, 2))
                nm_score = _min_area_cart(nm_pts)
                
                if nm_score > best_score:
                    best_score = nm_score
                    best_pts = nm_pts.copy()
            except Exception:
                pass

    return best_pts

if __name__ == "__main__":
    pts = construct()
    print(f"Best score: {_min_area_cart(pts)}")

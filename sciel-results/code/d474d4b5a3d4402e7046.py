"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull

try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False

def _compute_score(points):
    """
    Compute the score: min triangle area / convex hull area.
    points: np.array of shape (N, 2)
    """
    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area <= 1e-10:
        return 0.0
        
    # Compute all triangle areas vectorized
    # Indices for all unique triples i < j < k
    i_idx, j_idx, k_idx = np.indices((N, N, N))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    
    p1 = points[i_flat]
    p2 = points[j_flat]
    p3 = points[k_flat]
    
    # Area = 0.5 * |cross(p2-p1, p3-p1)|
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _generate_points(params):
    """
    Generate points from parameters [a1, b1, a2, b2].
    Angles: theta_k = 2*pi*k/13 + a1*cos(2*pi*k/13 + b1) + a2*cos(2*2*pi*k/13 + b2)
    Points on unit circle.
    """
    k = np.arange(N)
    base_angles = 2 * np.pi * k / N
    
    a1, b1, a2, b2 = params
    
    # Fourier perturbation
    h1 = a1 * np.cos(base_angles + b1)
    h2 = a2 * np.cos(2 * base_angles + b2)
    
    theta = base_angles + h1 + h2
    
    points = np.column_stack([np.cos(theta), np.sin(theta)])
    return points

def _objective(params):
    """
    Negative score for minimization.
    """
    points = _generate_points(params)
    score = _compute_score(points)
    return -score

def construct():
    """
    Construct 13 points to maximize the minimum triangle area over convex hull area.
    """
    best_score = -np.inf
    best_points = None
    
    # Regular polygon score
    reg_points = _generate_points(np.array([0.0, 0.0, 0.0, 0.0]))
    reg_score = _compute_score(reg_points)
    best_score = reg_score
    best_points = reg_points
    
    if HAS_CMA:
        # CMA-ES setup
        x0 = np.array([0.0, 0.0, 0.0, 0.0])
        sigma0 = 0.1
        
        # Bounds: [a1_low, a1_high, b1_low, b1_high, a2_low, a2_high, b2_low, b2_high]
        bounds = [-0.2, 0.2, 0.0, 2*np.pi, -0.1, 0.1, 0.0, 2*np.pi]
        
        try:
            es = cma.CMAEvolutionStrategy(
                x0, 
                sigma0, 
                bounds=bounds,
                seed=42,
                verbose=-9,
                maxiter=200
            )
            
            while not es.stop():
                solutions = es.ask()
                fvals = []
                for sol in solutions:
                    fvals.append(_objective(sol))
                es.tell(solutions, fvals)
            
            best_params = es.result.xbest
            best_score_cma = -es.result.fbest
            
            if best_score_cma > best_score:
                best_score = best_score_cma
                best_points = _generate_points(best_params)
                
        except Exception:
            pass # Fallback if CMA fails
            
    else:
        # Fallback to Nelder-Mead with multiple restarts
        from scipy.optimize import minimize
        rng = np.random.RandomState(42)
        
        for trial in range(20):
            if trial == 0:
                x0 = np.array([0.0, 0.0, 0.0, 0.0])
            else:
                x0 = np.array([
                    rng.uniform(-0.1, 0.1),
                    rng.uniform(0, 2*np.pi),
                    rng.uniform(-0.1, 0.1),
                    rng.uniform(0, 2*np.pi)
                ])
            
            try:
                res = minimize(
                    _objective,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_points = _generate_points(res.x)
            except Exception:
                continue

    return best_points
# EVOLVE-BLOCK-END

import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

N = 14
START_TIME = time.time()
DEADLINE = START_TIME + 115.0

def _get_triple_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=int)

TRIPLE_INDICES = _get_triple_indices(N)
I_IDX = TRIPLE_INDICES[:, 0]
J_IDX = TRIPLE_INDICES[:, 1]
K_IDX = TRIPLE_INDICES[:, 2]

def _score_general(points):
    """Compute min triangle area / convex hull area for general points."""
    if len(points) < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    p_i = points[I_IDX]
    p_j = points[J_IDX]
    p_k = points[K_IDX]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def _get_points_7_7(r, theta):
    """Generate points for two concentric heptagons."""
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    x_out = np.cos(angles_outer)
    y_out = np.sin(angles_outer)
    
    angles_inner = angles_outer + theta
    x_in = r * np.cos(angles_inner)
    y_in = r * np.sin(angles_inner)
    
    points = np.vstack([
        np.column_stack((x_out, y_out)),
        np.column_stack((x_in, y_in))
    ])
    return points

def _score_7_7(params):
    """Fast score for the 7+7 specific structure."""
    r, theta = params
    if r <= 0 or r >= 1:
        return 0.0
        
    points = _get_points_7_7(r, theta)
    
    # Analytic hull area for regular heptagon with R=1
    # A = 0.5 * n * R^2 * sin(2*pi/n)
    hull_area = 0.5 * 7 * np.sin(2*np.pi/7)
    
    p_i = points[I_IDX]
    p_j = points[J_IDX]
    p_k = points[K_IDX]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    return min_area / hull_area

def _neg_score_7_7(params):
    return -_score_7_7(params)

def _neg_score_general(flat_points):
    points = flat_points.reshape((N, 2))
    return -_score_general(points)

def construct():
    global START_TIME, DEADLINE
    START_TIME = time.time()
    DEADLINE = START_TIME + 115.0
    
    best_score = 0.0
    best_points = None
    
    # --- Phase 1: Optimized 7+7 Structure ---
    # This structure is highly symmetric and likely optimal or near-optimal.
    # Search space: r in (0, 1), theta in [0, 2*pi/7) due to symmetry.
    
    bounds = [(0.1, 0.95), (0, 2*np.pi/7)]
    
    try:
        # Differential Evolution for global search in 2D
        de_start = time.time()
        res_de = differential_evolution(
            _neg_score_7_7,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-6,
            polish=False # We will polish manually to save time and control it
        )
        
        if res_de.success or res_de.fun < 0:
            best_params_7_7 = res_de.x
            best_score_7_7 = -res_de.fun
        else:
            best_params_7_7 = np.array([0.5, 0.0])
            best_score_7_7 = _score_7_7(best_params_7_7)
            
    except Exception:
        best_params_7_7 = np.array([0.5, 0.0])
        best_score_7_7 = _score_7_7(best_params_7_7)
        
    # Local refinement of the 7+7 parameters
    if time.time() < DEADLINE - 10:
        try:
            res_nm = minimize(
                _neg_score_7_7,
                best_params_7_7,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 2000}
            )
            if -res_nm.fun > best_score_7_7:
                best_score_7_7 = -res_nm.fun
                best_params_7_7 = res_nm.x
        except Exception:
            pass

    if best_score_7_7 > best_score:
        best_score = best_score_7_7
        best_points = _get_points_7_7(best_params_7_7[0], best_params_7_7[1])

    # --- Phase 2: General Optimization from Best 7+7 Point ---
    # Allow the symmetry to break. The 7+7 configuration is a very good starting point.
    # We use Nelder-Mead on the full 28D vector.
    
    if best_points is not None and time.time() < DEADLINE - 20:
        try:
            # Start from the best 7+7 points
            x0 = best_points.flatten()
            
            # Nelder-Mead is robust for this size
            res_general = minimize(
                _neg_score_general,
                x0,
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 500, 'maxfev': 1000}
            )
            
            opt_points = res_general.x.reshape((N, 2))
            opt_score = _score_general(opt_points)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points
        except Exception:
            pass
            
    # --- Phase 3: Alternative Structures (if time remains) ---
    # Try 1 center + 13 outer ring? Or 2 inner + 12 outer?
    # Given the time spent on 7+7, we might not have much left.
    # Let's try a quick check of a 1+13 structure.
    
    if time.time() < DEADLINE - 15:
        def _score_1_13(params):
            r, theta = params
            if r <= 0 or r >= 1:
                return 0.0
            # 13 outer points
            angles_outer = np.linspace(0, 2*np.pi, 13, endpoint=False)
            x_out = np.cos(angles_outer)
            y_out = np.sin(angles_outer)
            
            # 1 inner point
            x_in = r * np.cos(theta)
            y_in = r * np.sin(theta)
            
            points = np.vstack([
                np.column_stack((x_out, y_out)),
                np.array([[x_in, y_in]])
            ])
            
            # Hull is outer 13-gon
            hull_area = 0.5 * 13 * np.sin(2*np.pi/13)
            
            p_i = points[I_IDX]
            p_j = points[J_IDX]
            p_k = points[K_IDX]
            
            v1 = p_j - p_i
            v2 = p_k - p_i
            
            cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            areas = 0.5 * np.abs(cross)
            
            min_area = np.min(areas)
            return min_area / hull_area

        try:
            # Quick DE for 1+13
            bounds_1_13 = [(0.0, 0.9), (0, 2*np.pi/13)]
            res_de_1_13 = differential_evolution(
                lambda p: -_score_1_13(p),
                bounds_1_13,
                seed=42,
                maxiter=50,
                popsize=10,
                tol=1e-5,
                polish=False
            )
            if res_de_1_13.success:
                s = -res_de_1_13.fun
                if s > best_score:
                    best_score = s
                    best_points = _get_points_7_7(res_de_1_13.x[0], res_de_1_13.x[1]) # Incorrect, need specific getter
                    # Fix: create points properly
                    r, th = res_de_1_13.x
                    angles_outer = np.linspace(0, 2*np.pi, 13, endpoint=False)
                    x_out = np.cos(angles_outer)
                    y_out = np.sin(angles_outer)
                    x_in = r * np.cos(th)
                    y_in = r * np.sin(th)
                    best_points = np.vstack([
                        np.column_stack((x_out, y_out)),
                        np.array([[x_in, y_in]])
                    ])
        except Exception:
            pass

    # Final fallback
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points

import numpy as np
from scipy.optimize import minimize
import time

# Try to import CMA, fall back to differential_evolution if not available
try:
    import cma
    HAS_CMA = True
except ImportError:
    from scipy.optimize import differential_evolution
    HAS_CMA = False

N = 11
h = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, h])

def project_to_triangle(p):
    """
    p: array of shape (N, 2)
    Returns projected points inside the triangle (0,0), (1,0), (0.5, h)
    """
    p = p.copy()
    # Clip to bounding box first
    p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
    p[:, 1] = np.clip(p[:, 1], 0.0, h)
    
    # Iterate a few times to handle corners properly
    for _ in range(3):
        # Edge AB: y >= 0
        mask_ab = p[:, 1] < 0.0
        if np.any(mask_ab):
            p[mask_ab, 1] = 0.0
            
        # Edge AC: y <= sqrt(3) x
        mask_ac = p[:, 1] > np.sqrt(3) * p[:, 0]
        if np.any(mask_ac):
            px = p[mask_ac, 0]
            py = p[mask_ac, 1]
            dot = px + py * np.sqrt(3)
            px_proj = dot / 4.0
            py_proj = dot * np.sqrt(3) / 4.0
            px_proj = np.clip(px_proj, 0.0, 0.5)
            py_proj = px_proj * np.sqrt(3)
            p[mask_ac, 0] = px_proj
            p[mask_ac, 1] = py_proj
            
        # Edge BC: y <= sqrt(3)(1-x)
        mask_bc = p[:, 1] > np.sqrt(3) * (1.0 - p[:, 0])
        if np.any(mask_bc):
            px = p[mask_bc, 0]
            py = p[mask_bc, 1]
            d = (np.sqrt(3) * px + py - np.sqrt(3)) / 4.0
            px_proj = px - d * np.sqrt(3)
            py_proj = py - d * 1.0
            px_proj = np.clip(px_proj, 0.5, 1.0)
            py_proj = np.sqrt(3) * (1.0 - px_proj)
            p[mask_bc, 0] = px_proj
            p[mask_bc, 1] = py_proj
            
    return p

def min_triangle_area_vectorized(points):
    """
    points: (N, 2)
    Returns the minimum area of any triangle formed by 3 points.
    Fully vectorized for speed.
    """
    n = len(points)
    # Create indices for all pairs (i, j) where i < j
    i_idx, j_idx = np.triu_indices(n, k=1)
    # Calculate vectors for all pairs
    # shape: (M, 2) where M = n*(n-1)/2
    v_ij = points[j_idx] - points[i_idx]
    
    # We need to find min area for all triples (i, j, k) with i < j < k
    # This is tricky to vectorize completely without huge memory if N is large, 
    # but for N=11, M=55. 
    # However, the standard vectorization for "min over all triples" requires 
    # iterating over one index or using a large 3D array.
    # For N=11, 3D array of size 11x11x11 is small (1331).
    
    # Let's use the 3D broadcast approach which is very fast in NumPy for small N
    # p1[:, None, None, :] -> (N, 1, 1, 2)
    # p2[None, :, None, :] -> (1, N, 1, 2)
    # p3[None, None, :, :] -> (1, 1, N, 2)
    
    p1 = points[:, None, None, :]
    p2 = points[None, :, None, :]
    p3 = points[None, None, :, :]
    
    # v12 = p2 - p1, v13 = p3 - p1
    v12 = p2 - p1
    v13 = p3 - p1
    
    # Cross product z-component: v12_x * v13_y - v12_y * v13_x
    # Shape: (N, N, N)
    cross = v12[..., 0] * v13[..., 1] - v12[..., 1] * v13[..., 0]
    areas = 0.5 * np.abs(cross)
    
    # We only want i < j < k.
    # Create a mask
    idx = np.arange(n)
    mask = (idx[:, None, None] < idx[None, :, None]) & (idx[None, :, None] < idx[None, None, :])
    
    # Set invalid areas to inf
    areas = np.where(mask, areas, np.inf)
    
    return np.min(areas)

def objective_flat(x):
    """
    x: flattened array of shape (2*N,)
    Returns negative min area (to be minimized).
    """
    points = x.reshape(N, 2)
    points = project_to_triangle(points)
    return -min_triangle_area_vectorized(points)

def generate_initial_guesses():
    """
    Generate diverse starting points for the optimizer.
    """
    guesses = []
    rng = np.random.default_rng(42)
    
    # 1. Symmetric rows (5, 3, 2, 1)
    y_levels = [0.08*h, 0.35*h, 0.62*h, 0.88*h]
    counts = [5, 3, 2, 1]
    pts = []
    for y, c in zip(y_levels, counts):
        x_min = y / np.sqrt(3)
        x_max = 1.0 - y / np.sqrt(3)
        if c > 1:
            margin = 0.03 * (x_max - x_min)
            xs = np.linspace(x_min + margin, x_max - margin, c)
        else:
            xs = np.array([(x_min + x_max) / 2])
        for x in xs:
            pts.append([x, y])
    guesses.append(np.array(pts))
    
    # 2. 4-4-3 layout
    y_levels_2 = [0.1*h, 0.4*h, 0.7*h]
    counts_2 = [4, 4, 3]
    pts = []
    for y, c in zip(y_levels_2, counts_2):
        x_min = y / np.sqrt(3)
        x_max = 1.0 - y / np.sqrt(3)
        if c > 1:
            margin = 0.02 * (x_max - x_min)
            xs = np.linspace(x_min + margin, x_max - margin, c)
        else:
            xs = np.array([(x_min + x_max) / 2])
        for x in xs:
            pts.append([x, y])
    guesses.append(np.array(pts))
    
    # 3. Random
    guesses.append(rng.random((N, 2)) * np.array([1.0, h]))
    guesses.append(rng.random((N, 2)) * np.array([1.0, h]))
    
    # Project all to ensure validity
    return [project_to_triangle(g) for g in guesses]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s buffer for 150s limit
    
    best_score = -1
    best_points = None
    
    guesses = generate_initial_guesses()
    
    # Strategy 1: CMA-ES if available, otherwise DE
    # CMA-ES is generally better for this type of problem than DE.
    
    if HAS_CMA:
        # CMA-ES parameters
        # sigma0: initial step size
        # bounds: box constraints
        # maxfevals: max function evaluations
        # We have ~140s. Vectorized objective is ~1ms. So ~100k evals possible.
        # But CMA-ES is efficient. Let's set a generous maxfevals.
        
        for i, start_points in enumerate(guesses):
            if time.time() > deadline - 20:
                break
                
            x0 = start_points.flatten()
            
            # CMA-ES options
            opts = {
                'bounds': [(0.0, 1.0), (0.0, h)] * N,
                'maxfevals': 20000,  # ~20-30s per run
                'verbose': -9,
                'seed': 123 + i,
                'CMA_stopsamefit': 50, # Stop if no improvement for 50 iterations
                'CMA_adaptation_of_maxfevals': False
            }
            
            try:
                es = cma.CMAEvolutionStrategy(x0, 0.05, opts)
                # Run until stop or timeout
                while not es.stop() and time.time() < deadline - 25:
                    es.ask()
                    # Evaluate
                    fvals = [objective_flat(x) for x in es.pop]
                    es.tell(es.pop, fvals)
                    
                res_x = es.result.xbest
                curr_points = project_to_triangle(res_x.reshape(N, 2))
                curr_area = min_triangle_area_vectorized(curr_points)
                if curr_area > best_score:
                    best_score = curr_area
                    best_points = curr_points
                    
            except Exception:
                pass
    else:
        # Fallback to Differential Evolution if cma is not installed
        # DE is slower per eval but we can still try
        for i, start_points in enumerate(guesses[:2]): # Limit restarts
            if time.time() > deadline - 30:
                break
            
            bounds = [(0.0, 1.0), (0.0, h)] * N
            try:
                res_de = differential_evolution(
                    objective_flat, 
                    bounds, 
                    seed=123 + i, 
                    maxiter=100, # Reduced since DE is slower
                    tol=1e-6,
                    polish=False,
                    init='sobol'
                )
                curr_points = project_to_triangle(res_de.x.reshape(N, 2))
                curr_area = min_triangle_area_vectorized(curr_points)
                if curr_area > best_score:
                    best_score = curr_area
                    best_points = curr_points
            except Exception:
                pass

    # Strategy 2: Local Polish with Nelder-Mead
    # NM is good for final fine-tuning if we have a good starting point.
    if best_points is not None and time.time() < deadline - 5.0:
        x0 = best_points.flatten()
        try:
            res = minimize(objective_flat, x0, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 
                                   'adaptive': True})
            curr_points = project_to_triangle(res.x.reshape(N, 2))
            curr_area = min_triangle_area_vectorized(curr_points)
            if curr_area > best_score:
                best_score = curr_area
                best_points = curr_points
        except Exception:
            pass
            
    # Fallback
    if best_points is None:
        best_points = guesses[0]
        
    return best_points

# EVOLVE-BLOCK-END

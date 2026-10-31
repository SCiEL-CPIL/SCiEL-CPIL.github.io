import numpy as np
import time
from scipy.optimize import minimize

N = 11

# Vertices of the equilateral triangle
# (0,0), (1,0), (0.5, sqrt(3)/2)
# Area = sqrt(3)/4

def bary_to_cartesian(u, v):
    """
    Convert barycentric coordinates (u, v, w=1-u-v) to Cartesian.
    Vertices: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    P = u*A + v*B + w*C = v*B + w*C
    x = v*1 + w*0.5 = v + 0.5*(1-u-v) = 0.5 + 0.5v - 0.5u
    y = v*0 + w*sqrt(3)/2 = sqrt(3)/2 * (1-u-v)
    """
    w = 1.0 - u - v
    x = 0.5 * v + 0.5 * w
    y = (3 ** 0.5) / 2.0 * w
    return x, y

def params_to_bary(params):
    """
    params: array of shape (N, 2) containing (a, b) for each point.
    Returns u, v arrays of shape (N,).
    """
    a = params[:, 0]
    b = params[:, 1]
    # Softmax-like normalization to ensure u, v, w > 0 and sum to 1
    # u = exp(a) / (exp(a) + exp(b) + 1)
    # v = exp(b) / (exp(a) + exp(b) + 1)
    # w = 1 / (exp(a) + exp(b) + 1)
    
    # To avoid overflow, shift by max
    max_val = np.maximum(a, b)
    max_val = np.maximum(max_val, 0)
    
    ea = np.exp(a - max_val)
    eb = np.exp(b - max_val)
    e0 = np.exp(-max_val)
    
    denom = ea + eb + e0
    u = ea / denom
    v = eb / denom
    return u, v

def get_min_area_cartesian(points):
    """
    points: array of shape (N, 2)
    Returns the minimum area of any triangle formed by 3 points.
    """
    n = len(points)
    min_area = float('inf')
    # Precompute for speed? N=11 is small, 165 triangles.
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Area = 0.5 * |det(p_j - p_i, p_k - p_i)|
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
                if area < min_area:
                    min_area = area
    return min_area

def objective(params):
    """
    params: flattened array of shape (2*N,)
    Returns negative soft-min area to be minimized.
    """
    # Unflatten
    p = params.reshape(N, 2)
    u, v = params_to_bary(p)
    
    # Convert to Cartesian
    x, y = bary_to_cartesian(u, v)
    points = np.column_stack((x, y))
    
    # Calculate all triangle areas
    n = N
    areas = np.zeros((n, n, n)) # Not needed, just loop
    min_area = float('inf')
    
    # For soft min, we need all areas
    # Let's compute them in a vectorized way if possible, or just loop
    # Loop is fine for N=11
    area_list = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                area = 0.5 * abs((points[j,0]-points[i,0])*(points[k,1]-points[i,1]) - (points[j,1]-points[i,1])*(points[k,0]-points[i,0]))
                area_list.append(area)
                
    areas = np.array(area_list)
    
    # Soft min approximation: -1/beta * log(sum(exp(-beta * area)))
    # We want to maximize min_area, so minimize -min_area.
    # Using log-sum-exp for stability
    beta = 50.0
    # max_area = np.max(areas)
    # shifted = -beta * (areas - max_area)
    # log_sum = np.log(np.sum(np.exp(shifted))) + beta * max_area
    # soft_min = -log_sum / beta
    
    # Simpler: just use the minimum directly? It's non-differentiable.
    # Let's use the log-sum-exp trick properly.
    # f(x) = -1/beta * log(sum(exp(-beta * x_i)))
    # This is a smooth approximation of -min(x_i).
    
    # To avoid under/overflow:
    max_a = np.max(areas)
    exp_terms = np.exp(-beta * (areas - max_a))
    log_sum = np.log(np.sum(exp_terms)) - beta * (-max_a) # Wait.
    # sum(exp(-beta * a_i)) = exp(-beta * max_a) * sum(exp(-beta * (a_i - max_a)))
    # log(sum(exp(-beta * a_i))) = -beta * max_a + log(sum(exp(-beta * (a_i - max_a))))
    # soft_min = -1/beta * ( -beta * max_a + log_sum_shifted )
    # soft_min = max_a - 1/beta * log_sum_shifted
    
    log_sum_shifted = np.log(np.sum(exp_terms))
    soft_min = max_a - (1.0 / beta) * log_sum_shifted
    
    # We want to maximize soft_min, so minimize -soft_min
    return -soft_min

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_score = -1.0
    best_params = None
    
    # Initializations
    # 1. Random
    # 2. Hexagonal lattice
    
    initializations = []
    
    # Random initializations
    for i in range(5):
        a = np.random.uniform(-5, 5, N)
        b = np.random.uniform(-5, 5, N)
        initializations.append(np.column_stack((a, b)).flatten())
        
    # Structured initialization: Triangular lattice
    # Place points in a grid pattern within the triangle
    # This is tricky with barycentric, so let's just try a few random structured ones
    # Or simply use the center and vertices? No, vertices are (0,0), (1,0), (0,5, sqrt3/2)
    # Barycentric for vertices:
    # (0,0) -> u=1, v=0 -> a=inf, b=-inf? 
    # Let's stick to random but maybe biased towards center?
    
    # Add a "center" initialization
    a_center = np.zeros(N)
    b_center = np.zeros(N)
    initializations.append(np.column_stack((a_center, b_center)).flatten())
    
    # Add a "boundary" initialization
    # Points near edges
    for i in range(2):
        a = np.random.uniform(-2, 2, N)
        b = np.random.uniform(-2, 2, N)
        initializations.append(np.column_stack((a, b)).flatten())

    for init in initializations:
        if time.time() > deadline:
            break
            
        try:
            # Optimize
            result = minimize(objective, init, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
            
            if result.success:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_params = result.x
                    
        except Exception:
            pass
            
    # If we have time, try to refine the best solution with higher beta or more iterations
    if best_params is not None and time.time() < deadline - 10:
        # We can't easily change beta inside the objective without closure.
        # We'll just run a second round of optimization from the best point with more iterations.
        try:
            result2 = minimize(objective, best_params, method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-12})
            if result2.success:
                score2 = -result2.fun
                if score2 > best_score:
                    best_score = score2
                    best_params = result2.x
        except Exception:
            pass

    if best_params is None:
        # Fallback to a random point
        u = np.random.uniform(0, 1, N)
        v = np.random.uniform(0, 1-u, N)
        x, y = bary_to_cartesian(u, v)
        return np.column_stack((x, y))
        
    # Convert best params to Cartesian
    p = best_params.reshape(N, 2)
    u, v = params_to_bary(p)
    x, y = bary_to_cartesian(u, v)
    points = np.column_stack((x, y))
    
    return points

# EVOLVE-BLOCK-START
import random
import numpy as np
import time
from scipy.optimize import minimize

N = 11

def bary_to_cartesian(u, v):
    w = 1.0 - u - v
    x = 0.5 * v + 0.5 * w
    y = (3 ** 0.5) / 2.0 * w
    return x, y

def params_to_bary(params):
    a = params[:, 0]
    b = params[:, 1]
    max_val = np.maximum(a, b)
    max_val = np.maximum(max_val, 0)
    ea = np.exp(a - max_val)
    eb = np.exp(b - max_val)
    e0 = np.exp(-max_val)
    denom = ea + eb + e0
    u = ea / denom
    v = eb / denom
    return u, v

def objective(params, beta=50.0):
    p = params.reshape(N, 2)
    u, v = params_to_bary(p)
    x, y = bary_to_cartesian(u, v)
    points = np.column_stack((x, y))
    
    n = N
    area_list = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                area = 0.5 * abs((points[j,0]-points[i,0])*(points[k,1]-points[i,1]) - (points[j,1]-points[i,1])*(points[k,0]-points[i,0]))
                area_list.append(area)
                
    areas = np.array(area_list)
    max_a = np.max(areas)
    exp_terms = np.exp(-beta * (areas - max_a))
    log_sum_shifted = np.log(np.sum(exp_terms))
    soft_min = max_a - (1.0 / beta) * log_sum_shifted
    return -soft_min

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = -1.0
    best_params = None
    
    # Fixed seed for reproducibility in case of ties, but we want diversity
    rng = np.random.default_rng(42)
    
    initializations = []
    
    # 1. Random initializations
    for i in range(10):
        a = rng.uniform(-5, 5, N)
        b = rng.uniform(-5, 5, N)
        initializations.append(np.column_stack((a, b)).flatten())
        
    # 2. Center initialization
    a_center = np.zeros(N)
    b_center = np.zeros(N)
    initializations.append(np.column_stack((a_center, b_center)).flatten())
    
    # 3. "Spread out" initialization
    # Try to place points roughly in a grid
    # Barycentric coordinates for a 3x3 grid inside the triangle
    # (1/3, 1/3), (2/3, 1/3), (1/3, 2/3) etc?
    # Let's just use random but with different scales
    for i in range(5):
        a = rng.uniform(-2, 2, N)
        b = rng.uniform(-2, 2, N)
        initializations.append(np.column_stack((a, b)).flatten())

    for init in initializations:
        if time.time() > deadline:
            break
            
        try:
            # Use a moderate beta for the first pass
            res = minimize(lambda x: objective(x, beta=20.0), init, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-8})
            
            if res.success:
                # Evaluate with higher beta to get a better estimate of the true min
                # Or just use the result from beta=20 as a starting point for a refinement
                score_est = -res.fun
                
                # Refinement with higher beta
                if time.time() < deadline - 5:
                    res2 = minimize(lambda x: objective(x, beta=100.0), res.x, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
                    if res2.success:
                        score = -res2.fun
                        if score > best_score:
                            best_score = score
                            best_params = res2.x
                    else:
                        score = score_est
                        if score > best_score:
                            best_score = score
                            best_params = res.x
                else:
                    score = score_est
                    if score > best_score:
                        best_score = score
                        best_params = res.x
                        
        except Exception:
            pass

    if best_params is None:
        # Fallback
        u = np.random.uniform(0, 1, N)
        v = np.random.uniform(0, 1-u, N)
        x, y = bary_to_cartesian(u, v)
        return np.column_stack((x, y))
        
    p = best_params.reshape(N, 2)
    u, v = params_to_bary(p)
    x, y = bary_to_cartesian(u, v)
    points = np.column_stack((x, y))
    
    return points
# EVOLVE-BLOCK-END

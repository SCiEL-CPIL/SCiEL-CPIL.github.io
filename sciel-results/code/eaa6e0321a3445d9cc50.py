from scipy.optimize import minimize
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def compute_covering_radius_and_facets(points):
    """
    Computes the covering radius and returns the radii and indices for each facet.
    points: (N, 3) array of unit vectors.
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        return np.pi, np.array([]), np.array([])
    
    if len(simplices) == 0:
        return np.pi, np.array([]), np.array([])
        
    # Extract vertices for all facets
    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]
    
    # Compute normals
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    
    valid = norms > 1e-10
    if not np.all(valid):
        norms[~valid] = 1.0
        n[~valid] = [0, 0, 1]
        
    u = n / norms[:, np.newaxis]
    
    # Orient normals outwards
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    # Circumradius
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    max_radius = np.max(radii)
    return max_radius, radii, simplices

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit

    best_points = None
    best_radius = np.pi

    # Initializations
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Create a grid of test points Y for the soft-max objective
    M = 3000
    j = np.arange(M) + 0.5
    zy = 1.0 - 2.0 * j / M
    phiy = np.pi * (1.0 + np.sqrt(5.0)) * j
    ry = np.sqrt(1.0 - zy * zy)
    Y = np.column_stack([ry * np.cos(phiy), ry * np.sin(phiy), zy])

    # Main Optimization Loop: Soft-Max Gradient Descent
    points = init_fib.copy()
    
    # Parameters
    sigma = 0.15
    rho = 0.15
    lr = 0.005
    decay = 0.98
    
    while time.time() - start_time < time_limit * 0.7:
        C = Y @ points.T
        C = np.clip(C, -1.0, 1.0)
        D = np.arccos(C)
        
        D_min = np.min(D, axis=1, keepdims=True)
        exp_term = np.exp(-(D - D_min) / sigma)
        sum_exp = np.sum(exp_term, axis=1, keepdims=True)
        W = exp_term / sum_exp
        
        D_soft = D_min.squeeze(1) - sigma * np.log(sum_exp.squeeze(1))
        
        D_soft_max = np.max(D_soft)
        exp_term_y = np.exp((D_soft - D_soft_max) / rho)
        sum_exp_y = np.sum(exp_term_y)
        V = exp_term_y / sum_exp_y
        
        C_safe = np.clip(C, -1.0 + 1e-6, 1.0 - 1e-6)
        denom = np.sqrt(1.0 - C_safe * C_safe)
        
        Y_exp = Y[:, np.newaxis, :]
        W_exp = W[:, :, np.newaxis]
        denom_exp = denom[:, :, np.newaxis]
        
        G = W_exp * (-Y_exp / denom_exp)
        
        V_exp = V[:, np.newaxis, np.newaxis]
        
        grad = np.sum(V_exp * G, axis=0)
        
        points -= lr * grad
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        sigma *= decay
        rho *= decay
        
        if sigma < 0.01:
            sigma = 0.01
        if rho < 0.01:
            rho = 0.01

    # Polish Phase: L-BFGS-B on Smooth Objective
    sigma_final = 0.01
    rho_final = 0.01
    
    def objective_and_grad(x_flat):
        P = x_flat.reshape(N, 3)
        norms = np.linalg.norm(P, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        P_norm = P / norms
        
        C = Y @ P_norm.T
        C = np.clip(C, -1.0, 1.0)
        D = np.arccos(C)
        
        D_min = np.min(D, axis=1, keepdims=True)
        exp_term = np.exp(-(D - D_min) / sigma_final)
        sum_exp = np.sum(exp_term, axis=1, keepdims=True)
        W = exp_term / sum_exp
        
        D_soft = D_min.squeeze(1) - sigma_final * np.log(sum_exp.squeeze(1))
        
        D_soft_max = np.max(D_soft)
        exp_term_y = np.exp((D_soft - D_soft_max) / rho_final)
        sum_exp_y = np.sum(exp_term_y)
        V = exp_term_y / sum_exp_y
        
        C_safe = np.clip(C, -1.0 + 1e-6, 1.0 - 1e-6)
        denom = np.sqrt(1.0 - C_safe * C_safe)
        
        Y_exp = Y[:, np.newaxis, :]
        W_exp = W[:, :, np.newaxis]
        denom_exp = denom[:, :, np.newaxis]
        
        G = W_exp * (-Y_exp / denom_exp)
        V_exp = V[:, np.newaxis, np.newaxis]
        
        grad_P = np.sum(V_exp * G, axis=0)
        
        obj = rho_final * np.log(sum_exp_y)
        
        radial = np.sum(grad_P * P_norm, axis=1, keepdims=True)
        tangential = grad_P - radial * P_norm
        grad_x = tangential / norms
        
        return obj, grad_x.flatten()

    x0 = points.flatten()
    
    stop_flag = [False]
    def callback(xk):
        if time.time() - start_time > time_limit - 5.0:
            stop_flag[0] = True
        return stop_flag[0]
        
    try:
        res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, callback=callback, options={'maxiter': 10000, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.success or res.status == 1:
            new_points = res.x.reshape(N, 3)
            new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
            
            max_radius, _, _ = compute_covering_radius_and_facets(new_points)
            if max_radius < best_radius:
                best_radius = max_radius
                best_points = new_points.copy()
            
            curr_radius, _, _ = compute_covering_radius_and_facets(points)
            if curr_radius < best_radius:
                best_radius = curr_radius
                best_points = points.copy()
    except Exception:
        pass

    curr_radius, _, _ = compute_covering_radius_and_facets(points)
    if curr_radius < best_radius:
        best_radius = curr_radius
        best_points = points.copy()

    if best_points is None:
        best_points = points

    return best_points

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius(pts):
    """Compute the covering radius in degrees for a set of points on the unit sphere."""
    pts = _normalize(pts)
    n_test = 20000
    i_test = np.arange(n_test) + 0.5
    z_test = 1.0 - 2.0 * i_test / n_test
    phi_test = np.pi * (1.0 + np.sqrt(5.0)) * i_test
    r_test = np.sqrt(1.0 - z_test * z_test)
    test_pts = np.column_stack([r_test * np.cos(phi_test), r_test * np.sin(phi_test), z_test])
    
    dots = test_pts @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    min_dists = np.min(dists, axis=1)
    return np.degrees(np.max(min_dists))

def construct():
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)

    # 2. Pre-processing: Lloyd's Algorithm (Voronoi Centroid Update)
    # This helps escape poor local minima and separates points
    M_lloyd = 20000
    i_test = np.arange(M_lloyd) + 0.5
    z_test = 1.0 - 2.0 * i_test / M_lloyd
    phi_test = np.pi * (1.0 + np.sqrt(5.0)) * i_test
    r_test = np.sqrt(1.0 - z_test * z_test)
    test_pts_lloyd = np.column_stack([r_test * np.cos(phi_test), r_test * np.sin(phi_test), z_test])
    
    for _ in range(100):
        dots = test_pts_lloyd @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)
        nearest = np.argmin(dists, axis=1)
        
        new_pts = np.zeros_like(pts)
        for j in range(N):
            mask = (nearest == j)
            count = np.sum(mask)
            if count > 0:
                centroid = np.mean(test_pts_lloyd[mask], axis=0)
                norm_c = np.linalg.norm(centroid)
                if norm_c > 1e-10:
                    new_pts[j] = centroid / norm_c
                else:
                    new_pts[j] = pts[j]
            else:
                new_pts[j] = pts[j]
        pts = _normalize(new_pts)

    # 3. Optimization: L-BFGS-B with analytic gradient of smooth covering radius
    M_opt = 10000
    i_test_opt = np.arange(M_opt) + 0.5
    z_test_opt = 1.0 - 2.0 * i_test_opt / M_opt
    phi_test_opt = np.pi * (1.0 + np.sqrt(5.0)) * i_test_opt
    r_test_opt = np.sqrt(1.0 - z_test_opt * z_test_opt)
    test_pts_opt = np.column_stack([r_test_opt * np.cos(phi_test_opt), r_test_opt * np.sin(phi_test_opt), z_test_opt])
    
    beta = 8.0
    gamma = 8.0
    
    def obj_and_grad(flat):
        P = _normalize(flat.reshape(N, 3))
        U = P @ test_pts_opt.T # N x M
        U = np.clip(U, -0.999999, 0.999999)
        D = np.arccos(U) # N x M
        
        # Soft-min calculation
        E_neg = np.exp(-gamma * D)
        Z = np.sum(E_neg, axis=0) # M
        # Avoid log(0)
        Z[Z < 1e-300] = 1e-300
        V = E_neg / Z # N x M
        D_soft = -1.0/gamma * np.log(Z) # M
        
        # Soft-max calculation
        E_pos = np.exp(beta * D_soft)
        Y = np.sum(E_pos)
        # Avoid log(0)
        if Y < 1e-300:
            Y = 1e-300
        W = E_pos / Y # M
        obj = 1.0/beta * np.log(Y)
        
        # Gradient calculation
        # S_kj = W_j * V_kj / sqrt(1 - U_kj^2)
        denom = np.sqrt(1.0 - U**2)
        denom[denom < 1e-10] = 1e-10
        A = V * W[None, :] # N x M
        S = A / denom # N x M
        
        # Term1: - S @ T.T
        term1 = - (S @ test_pts_opt.T) # N x 3
        # Term2: P * sum(S * U, axis=1)
        term2_coeff = np.sum(S * U, axis=1) # N
        term2 = P * term2_coeff[:, None] # N x 3
        
        grad = term1 + term2
        return obj, grad.flatten()
        
    res = minimize(obj_and_grad, pts.flatten(), method='L-BFGS-B', jac=True, 
                   options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
    
    pts_opt = _normalize(res.x.reshape(N, 3))
    
    return pts_opt
# EVOLVE-BLOCK-END

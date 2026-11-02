import numpy as np
from scipy.optimize import minimize

N = 11

def construct():
    # 1. Define the triangle vertices
    v1 = np.array([0.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v3 = np.array([0.5, np.sqrt(3) / 2])
    
    # Bounding box for L-BFGS-B
    h = np.sqrt(3) / 2
    bounds = [(0.0, 1.0), (0.0, h)] * N
    
    # Function to calculate barycentric coordinates (u, v) for penalty
    def get_barycentric(p, v1, v2, v3):
        v0 = v3 - v1
        v1v = v2 - v1
        v2v = p - v1
        
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1v)
        d02 = np.dot(v0, v2v)
        d11 = np.dot(v1v, v1v)
        d12 = np.dot(v1v, v2v)
        
        denom = d00 * d11 - d01 * d01
        if denom == 0:
            return 0.0, 0.0
            
        u = (d11 * d02 - d01 * d12) / denom
        v = (d00 * d12 - d01 * d02) / denom
        
        return u, v

    # Energy function: Repulsive + Boundary Penalty
    def energy(params):
        pts = params.reshape(N, 2)
        
        # Boundary Penalty
        pen = 0.0
        for p in pts:
            u, v = get_barycentric(p, v1, v2, v3)
            if u < 0: pen += u**2
            if v < 0: pen += v**2
            if u + v > 1: pen += (u + v - 1)**2
        
        # Repulsive Energy: Sum of 1/A^2
        e_rep = 0.0
        eps = 1e-12
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    p1, p2, p3 = pts[i], pts[j], pts[k]
                    v1v = p2 - p1
                    v2v = p3 - p1
                    area = 0.5 * abs(v1v[0]*v2v[1] - v1v[1]*v2v[0])
                    e_rep += 1.0 / (area + eps)**2
                    
        return e_rep + 1000.0 * pen

    best_energy = float('inf')
    best_params = None
    
    # Multiple restarts
    n_restarts = 30
    np.random.seed(42)
    
    for r in range(n_restarts):
        # Initialize points randomly inside the triangle
        bary = np.random.dirichlet(np.ones(3), size=N)
        x0 = np.zeros((N, 2))
        x0[:, 0] = bary[:, 1] * v2[0] + bary[:, 2] * v3[0]
        x0[:, 1] = bary[:, 1] * v2[1] + bary[:, 2] * v3[1]
        
        # Optimize using L-BFGS-B
        res = minimize(energy, x0.flatten(), method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if res.fun < best_energy:
            best_energy = res.fun
            best_params = res.x.copy()
            
    if best_params is None:
        # Fallback: return a valid configuration (e.g., first restart)
        bary = np.random.dirichlet(np.ones(3), size=N)
        best_params = np.zeros((N, 2))
        best_params[:, 0] = bary[:, 1] * v2[0] + bary[:, 2] * v3[0]
        best_params[:, 1] = bary[:, 1] * v2[1] + bary[:, 2] * v3[1]

    best_points = best_params.reshape(N, 2)
    
    return best_points

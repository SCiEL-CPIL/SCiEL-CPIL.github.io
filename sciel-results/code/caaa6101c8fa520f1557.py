import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

N = 100

def construct():
    # 1. Generate initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    
    # Convert to Cartesian directly for initialization
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    pts_init = np.column_stack((x, y, z))
    
    # Flatten for optimization: shape (3*N,)
    x0 = pts_init.flatten()
    
    # Objective function
    def energy(params):
        pts = params.reshape(N, 3)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        d = dists[iu]
        if np.any(d < 1e-8):
            return 1e10
        return -np.sum(np.log(d))
        
    # Define the constraint function: sum of squared norms - N = 0 is not enough,
    # we need each point to be on the sphere.
    # So we have N separate constraints.
    # trust-constr can handle a vector of constraints.
    
    def constraint_func(x):
        pts = x.reshape(N, 3)
        norms_sq = np.sum(pts**2, axis=1)
        return norms_sq - 1.0
        
    def constraint_jac(x):
        pts = x.reshape(N, 3)
        # Jacobian is a block diagonal matrix with 2*pts_i on the diagonal blocks
        # Shape: (N, 3*N)
        J = np.zeros((N, 3*N))
        for i in range(N):
            J[i, 3*i:3*i+3] = 2 * pts[i]
        return J
        
    cons = NonlinearConstraint(constraint_func, 0, 0, jac=constraint_jac)
    
    # Use trust-constr
    res = minimize(energy, x0, method='trust-constr', constraints=cons, 
                   options={'maxiter': 1000, 'gtol': 1e-8, 'xtol': 1e-8})
    
    if res.success or res.fun < energy(x0):
        best_params = res.x
    else:
        best_params = x0
        
    pts = best_params.reshape(N, 3)
    
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    pts = pts / norms
    
    return pts

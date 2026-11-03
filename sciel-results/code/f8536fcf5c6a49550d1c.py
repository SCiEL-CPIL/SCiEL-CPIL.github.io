import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # 1. Generate initial configurations
    # Config 1: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    
    theta1 = np.arccos(z)
    phi1 = np.arctan2(r * np.sin(phi), r * np.cos(phi)) % (2 * np.pi)
    
    x0_1 = np.zeros(2 * N)
    x0_1[0::2] = theta1
    x0_1[1::2] = phi1

    # Config 2: Random points (seeded for reproducibility)
    rng = np.random.default_rng(42)
    theta2 = rng.uniform(0, np.pi, N)
    phi2 = rng.uniform(0, 2 * np.pi, N)
    x0_2 = np.zeros(2 * N)
    x0_2[0::2] = theta2
    x0_2[1::2] = phi2

    # Config 3: Perturbed Fibonacci
    noise_theta = rng.normal(0, 0.1, N)
    noise_phi = rng.normal(0, 0.1, N)
    theta3 = theta1 + noise_theta
    theta3 = np.clip(theta3, 1e-8, np.pi - 1e-8)
    phi3 = (phi1 + noise_phi) % (2 * np.pi)
    x0_3 = np.zeros(2 * N)
    x0_3[0::2] = theta3
    x0_3[1::2] = phi3

    # Objective function
    def energy(params):
        thetas = params[0::2]
        phis = params[1::2]
        
        sin_t = np.sin(thetas)
        x = sin_t * np.cos(phis)
        y = sin_t * np.sin(phis)
        z = np.cos(thetas)
        
        pts = np.column_stack((x, y, z))
        
        # Calculate pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        iu = np.triu_indices(N, k=1)
        d = dists[iu]
        
        if np.any(d < 1e-8):
            return 1e10
        
        return -np.sum(np.log(d))
        
    bounds = []
    for i in range(N):
        bounds.append((1e-8, np.pi - 1e-8))
        bounds.append((0, 2 * np.pi))
        
    options = {'maxiter': 3000, 'ftol': 1e-10, 'gtol': 1e-8}
    
    best_energy = np.inf
    best_params = x0_1
    
    # Run optimization from multiple starts
    for x0 in [x0_1, x0_2, x0_3]:
        try:
            res = minimize(energy, x0, method='L-BFGS-B', bounds=bounds, options=options)
            if res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x
        except Exception:
            pass

    # Convert best_params back to Cartesian
    thetas = best_params[0::2]
    phis = best_params[1::2]
    
    sin_t = np.sin(thetas)
    x = sin_t * np.cos(phis)
    y = sin_t * np.sin(phis)
    z = np.cos(thetas)
    
    pts = np.column_stack((x, y, z))
    
    # Normalize to ensure they are exactly on the sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1.0, norms)
    pts = pts / norms
    
    return pts

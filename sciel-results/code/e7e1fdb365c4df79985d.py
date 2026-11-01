import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # 1. Define a base grid
    # 4 x 5 x 5 grid
    x_coords = np.linspace(-0.4, 0.4, 4)
    y_coords = np.linspace(-0.4, 0.4, 5)
    z_coords = np.linspace(-0.4, 0.4, 5)
    
    grid = np.array(np.meshgrid(x_coords, y_coords, z_coords)).T.reshape(-1, 3)
    grid = grid[:N]
    
    # 2. Define Fourier Feature Parameterization
    # This approach uses a linear combination of Fourier basis functions to generate displacements.
    # The parameters are the weights of these basis functions.
    # This creates a smooth, low-dimensional landscape for optimization.
    
    rng = np.random.default_rng(42)
    
    # Number of Fourier components
    K = 15
    # Frequencies: chosen to resolve grid spacing (~0.2)
    # Grid spacing is approx 0.2 to 0.26.
    # We want frequencies from low (global) to medium (local).
    omega = rng.uniform(5, 50, size=(K, 3))
    
    # Precompute Fourier features for the grid points
    # phi: (N, 2*K + 1)
    proj = grid @ omega.T # (N, K)
    phi = np.hstack([np.cos(proj), np.sin(proj)])
    # Add bias term (1)
    phi = np.hstack([np.ones((N, 1)), phi])
    
    n_features = phi.shape[1]
    
    # Weights W: (n_features, 3)
    # Initialize to 0 so we start at the grid
    W0 = np.zeros((n_features, 3))
    
    def centers_from_W(W_flat):
        W = W_flat.reshape(n_features, 3)
        disp = phi @ W # (N, 3)
        centers = grid + disp
        return centers

    def softmin(x, tau=0.005):
        m = np.min(x)
        return m - tau * np.log(np.sum(np.exp(-(x - m) / tau)))

    def objective(W_flat):
        centers = centers_from_W(W_flat)
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        pair_constraints = dists[iu] / 2.0
        
        # Wall constraints
        walls = 0.5 - np.abs(centers)
        wall_constraints = walls.ravel()
        
        all_constraints = np.concatenate([pair_constraints, wall_constraints])
        
        # Maximize the minimum constraint
        sm = softmin(all_constraints, tau=0.005)
        return -sm

    # 3. Optimization
    # Stage 1: Optimize W with L-BFGS-B
    res1 = minimize(objective, W0.ravel(), method='L-BFGS-B', 
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
    
    centers = centers_from_W(res1.x)
    
    # Stage 2: Direct coordinate optimization (Local Polish)
    # This allows escaping the linear subspace of the Fourier features.
    def objective_direct(centers_flat):
        centers = centers_flat.reshape(N, 3)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        pair_constraints = dists[iu] / 2.0
        walls = 0.5 - np.abs(centers)
        wall_constraints = walls.ravel()
        all_constraints = np.concatenate([pair_constraints, wall_constraints])
        sm = softmin(all_constraints, tau=0.002)
        return -sm

    res2 = minimize(objective_direct, centers.ravel(), method='L-BFGS-B', 
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
    centers = res2.x.reshape(N, 3)
    
    # Final safety clip
    centers = np.clip(centers, -0.49999, 0.49999)
    
    return centers

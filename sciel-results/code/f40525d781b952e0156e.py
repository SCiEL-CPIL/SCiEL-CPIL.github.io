import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # 1. Define Input Features for the Neural Network
    # We use a coarse 5x5x4 grid as the input context for each sphere.
    # This provides spatial inductive bias.
    x_coords = np.linspace(-1, 1, 5)
    y_coords = np.linspace(-1, 1, 5)
    z_coords = np.linspace(-1, 1, 4)
    grid = np.array(np.meshgrid(x_coords, y_coords, z_coords)).T.reshape(-1, 3)
    X = grid[:N]  # Shape (100, 3)
    
    # 2. Define MLP Architecture
    # Input: 3 (x, y, z)
    # Hidden: 16 units, Tanh activation
    # Output: 3 (dx, dy, dz) -> scaled to [-0.5, 0.5] via tanh
    
    hidden_dim = 16
    
    # Initialize weights
    np.random.seed(42)
    W1 = np.random.randn(3, hidden_dim) * 0.1
    b1 = np.zeros(hidden_dim)
    W2 = np.random.randn(hidden_dim, 3) * 0.1
    b2 = np.zeros(3)
    
    # Flatten parameters
    p0 = np.concatenate([W1.ravel(), b1, W2.ravel(), b2])
    
    def unpack_params(p):
        w1 = p[:3*hidden_dim].reshape(3, hidden_dim)
        b1 = p[3*hidden_dim:3*hidden_dim+hidden_dim]
        w2 = p[3*hidden_dim+hidden_dim:3*hidden_dim+hidden_dim+hidden_dim*3].reshape(hidden_dim, 3)
        b2 = p[3*hidden_dim+hidden_dim+hidden_dim*3:]
        return w1, b1, w2, b2
    
    def get_centers(p):
        w1, b1, w2, b2 = unpack_params(p)
        h = np.tanh(X @ w1 + b1)  # (100, hidden_dim)
        y = h @ w2 + b2           # (100, 3)
        centers = 0.5 * np.tanh(y) # Map to (-0.5, 0.5)
        return centers

    def softmin(x, tau):
        m = np.min(x)
        return m - tau * np.log(np.sum(np.exp(-(x - m) / tau)))

    def objective(p, tau):
        centers = get_centers(p)
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        pair_constraints = dists[iu] / 2.0
        
        # Wall constraints
        walls = 0.5 - np.abs(centers)
        wall_constraints = walls.ravel()
        
        all_constraints = np.concatenate([pair_constraints, wall_constraints])
        sm = softmin(all_constraints, tau)
        return -sm

    # 3. Optimization
    # Start with a larger tau to find a good basin, then anneal
    taus = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    p_opt = p0
    
    for tau in taus:
        res = minimize(lambda p: objective(p, tau), p_opt, method='L-BFGS-B',
                       options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 100})
        p_opt = res.x
        
    centers = get_centers(p_opt)
    
    # Final safety check: ensure finite and within bounds
    # The tanh ensures bounds, but let's clip just in case of numerical issues
    centers = np.clip(centers, -0.499999, 0.499999)
    
    return centers

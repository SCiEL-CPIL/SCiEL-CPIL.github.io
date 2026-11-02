import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # Small MLP-based parameterization for sphere centers
    # Architecture: input (N,) -> hidden (64,) -> output (N*3,)
    # We use a fixed random projection as input features to break symmetry
    
    # Fixed random input features for each sphere (deterministic seed)
    rng = np.random.RandomState(42)
    input_dim = 16
    X = rng.randn(N, input_dim) * 0.1  # Small scale input
    
    # MLP parameters:
    # Layer 1: W1 (input_dim, 64), b1 (64,)
    # Layer 2: W2 (64, 3), b2 (3,)
    # Total params: 16*64 + 64 + 64*3 + 3 = 1024 + 64 + 192 + 3 = 1283
    
    def mlp_forward(params):
        W1 = params[0:1024].reshape(input_dim, 64)
        b1 = params[1024:1088]
        W2 = params[1088:1280].reshape(64, 3)
        b2 = params[1280:1283]
        
        h = X @ W1 + b1
        h = np.tanh(h)
        centers = h @ W2 + b2
        return centers
    
    def softmin(x, tau):
        m = np.min(x)
        return m - tau * np.log(np.sum(np.exp(-(x - m) / tau)))
    
    def objective(params, tau):
        centers = mlp_forward(params)
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        pair_constraints = dists[iu] / 2.0
        
        # Wall constraints
        walls = 0.5 - np.abs(centers)
        wall_constraints = walls.ravel()
        
        all_constraints = np.concatenate([pair_constraints, wall_constraints])
        
        # Maximize the minimum constraint
        sm = softmin(all_constraints, tau)
        return -sm
    
    # Initialize MLP weights
    rng_init = np.random.RandomState(123)
    W1_init = rng_init.randn(input_dim, 64) * 0.1
    b1_init = np.zeros(64)
    W2_init = rng_init.randn(64, 3) * 0.01
    b2_init = np.zeros(3)
    
    params_init = np.concatenate([W1_init.ravel(), b1_init, W2_init.ravel(), b2_init])
    
    # Multi-stage optimization with decreasing tau
    taus = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    params_opt = params_init
    for tau in taus:
        res = minimize(lambda p: objective(p, tau), params_opt, method='L-BFGS-B', 
                       options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
        params_opt = res.x
    
    centers = mlp_forward(params_opt)
    
    # Stage 2: Direct coordinate optimization (Local Polish)
    def objective_direct(centers_flat, tau):
        centers = centers_flat.reshape(N, 3)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, k=1)
        pair_constraints = dists[iu] / 2.0
        walls = 0.5 - np.abs(centers)
        wall_constraints = walls.ravel()
        all_constraints = np.concatenate([pair_constraints, wall_constraints])
        sm = softmin(all_constraints, tau)
        return -sm

    # Further refine with direct coordinates
    taus_direct = [0.002, 0.001, 0.0005, 0.0002]
    c_opt = centers.ravel()
    for tau in taus_direct:
        res = minimize(lambda c: objective_direct(c, tau), c_opt, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
        c_opt = res.x
    
    centers = c_opt.reshape(N, 3)
    
    # Final safety clip
    centers = np.clip(centers, -0.49999, 0.49999)
    
    return centers

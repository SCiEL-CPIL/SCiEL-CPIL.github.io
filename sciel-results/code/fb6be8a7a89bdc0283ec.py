import numpy as np
from scipy.optimize import minimize

N = 100

def construct():
    # --- 1. Deterministic Initialization ---
    # Use a 4x5x5 grid, taking the first 100 points
    x_coords = np.linspace(-0.4, 0.4, 4)
    y_coords = np.linspace(-0.4, 0.4, 5)
    z_coords = np.linspace(-0.4, 0.4, 5)
    grid = np.array(np.meshgrid(x_coords, y_coords, z_coords)).T.reshape(-1, 3)
    grid = grid[:N]

    # --- 2. Neural Network Parameterization (MLP) ---
    # Input: grid positions (3)
    # Hidden layers: 24, 24
    # Output: 3 (displacements)
    # This creates a non-linear mapping from the fixed grid to the optimized centers.
    # The non-linearity allows the network to learn complex local adjustments
    # that a simple linear Fourier expansion might miss, especially near boundaries
    # or in asymmetric packing regions.

    # Initialize weights with small random values (deterministic seed for reproducibility)
    rng = np.random.default_rng(42)

    # Layer 1: 3 -> 24
    W1 = rng.normal(0, 0.1, (3, 24))
    b1 = np.zeros(24)

    # Layer 2: 24 -> 24
    W2 = rng.normal(0, 0.1, (24, 24))
    b2 = np.zeros(24)

    # Layer 3: 24 -> 3
    W3 = np.zeros((24, 3))  # Start with zero output so initial centers = grid
    b3 = np.zeros(3)

    def mlp_forward(params):
        """
        params: flattened vector of [W1, b1, W2, b2, W3, b3]
        Returns: (N, 3) centers
        """
        # Unpack parameters
        size_W1 = 3 * 24
        size_b1 = 24
        size_W2 = 24 * 24
        size_b2 = 24
        size_W3 = 24 * 3
        size_b3 = 3

        idx = 0
        W1_flat = params[idx:idx + size_W1].reshape(3, 24)
        idx += size_W1
        b1_flat = params[idx:idx + size_b1]
        idx += size_b1
        W2_flat = params[idx:idx + size_W2].reshape(24, 24)
        idx += size_W2
        b2_flat = params[idx:idx + size_b2]
        idx += size_b2
        W3_flat = params[idx:idx + size_W3].reshape(24, 3)
        idx += size_W3
        b3_flat = params[idx:idx + size_b3]
        idx += size_b3

        # Forward pass
        h1 = np.tanh(grid @ W1_flat + b1_flat)  # (N, 24)
        h2 = np.tanh(h1 @ W2_flat + b2_flat)    # (N, 24)
        disp = h2 @ W3_flat + b3_flat           # (N, 3)
        centers = grid + disp
        return centers

    # Flatten initial parameters
    params0 = np.concatenate([
        W1.ravel(), b1.ravel(),
        W2.ravel(), b2.ravel(),
        W3.ravel(), b3.ravel()
    ])

    # --- 3. Objective Function ---
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

    # --- 4. Optimization with Annealing ---
    # Multi-stage optimization with decreasing temperature tau
    taus = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

    params_opt = params0
    for tau in taus:
        res = minimize(
            lambda p: objective(p, tau),
            params_opt,
            method='L-BFGS-B',
            options={'maxiter': 800, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 50}
        )
        params_opt = res.x

    centers = mlp_forward(params_opt)

    # --- 5. Direct Coordinate Polish ---
    # The MLP may have converged to a local minimum in parameter space.
    # Direct coordinate optimization allows for fine-tuning in the full 300D space.
    def objective_direct(centers_flat, tau):
        centers = centers_flat.reshape(N, 3)

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

    taus_direct = [0.002, 0.001, 0.0005, 0.0001]
    c_opt = centers.ravel()
    for tau in taus_direct:
        res = minimize(
            lambda c: objective_direct(c, tau),
            c_opt,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        c_opt = res.x

    centers = c_opt.reshape(N, 3)

    # Final safety clip to ensure centers are strictly inside the cube
    centers = np.clip(centers, -0.499999, 0.499999)

    return centers

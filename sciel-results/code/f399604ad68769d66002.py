"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, Bounds


def construct():
    """
    Improved Lennard-Jones style minimization with explicit wall repulsion
    to maximize minimum inter-particle distance and wall clearance.
    """
    best_score = 0.0
    best_result = None
    
    # Wall repulsion strength
    wall_alpha = 1.0
    # Exponent for repulsion (higher = steeper, focuses on min distance)
    exponent = 12
    
    def energy_and_grad(pos):
        pos = pos.reshape(N, 3)
        
        # Pair interactions
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        
        iu = np.triu_indices(N, k=1)
        d_sq = dist_sq[iu]
        d_sq = np.maximum(d_sq, 1e-12)
        
        # Energy: sum(d^-12)
        pair_energy = np.sum(1.0 / (d_sq ** (exponent / 2)))
        
        # Gradient of pair energy
        # d/dx (d^-12) = -12 * d^-14 * (x_i - x_j)
        # coeff = -12 * (dist_sq)^-(12/2 + 1) = -12 * (dist_sq)^-7
        coeff = -12.0 / (dist_sq ** 7)
        np.fill_diagonal(coeff, 0.0)
        grad_pair = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
        
        # Wall interactions
        # Distance to walls: for each coord k, dist = 0.5 - |pos_k|
        # We want to penalize small distances.
        # Let w_k = 0.5 - abs(pos_k). If w_k is small, penalty is high.
        # Penalty term: sum(w_k^-12)
        # Note: if pos is outside, w_k is negative. But we use bounds, so pos is inside.
        # To be safe, clip w_k to a small positive value.
        
        abs_pos = np.abs(pos)
        w = 0.5 - abs_pos
        w = np.maximum(w, 1e-6) # Prevent division by zero or negative base
        
        wall_energy = wall_alpha * np.sum(1.0 / (w ** exponent))
        
        # Gradient of wall energy
        # d/dx_k ( (0.5 - |x_k|)^-12 )
        # Let u = 0.5 - |x_k|. du/dx_k = -sign(x_k)
        # d/dx_k (u^-12) = -12 * u^-13 * du/dx_k = 12 * sign(x_k) * u^-13
        # So grad_wall_k = 12 * sign(pos_k) * (w_k)^-13
        
        sign_pos = np.sign(pos)
        # Handle pos=0 case (sign is 0, but derivative is undefined/0 in symmetric sense, or we can just let it be 0)
        grad_wall = wall_alpha * 12.0 * sign_pos * (w ** (-13))
        
        total_energy = pair_energy + wall_energy
        total_grad = grad_pair + grad_wall
        
        return total_energy, total_grad.ravel()
    
    bounds = Bounds(np.full(N * 3, -0.5), np.full(N * 3, 0.5))
    
    # Multiple random initializations
    np.random.seed(42)
    
    num_trials = 8
    for trial in range(num_trials):
        if trial == 0:
            # Grid initialization with jitter
            nx, ny, nz = 5, 4, 5
            x = np.linspace(-0.5 + 0.5/nx, 0.5 - 0.5/nx, nx)
            y = np.linspace(-0.5 + 0.5/ny, 0.5 - 0.5/ny, ny)
            z = np.linspace(-0.5 + 0.5/nz, 0.5 - 0.5/nz, nz)
            positions = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
            np.random.seed(123 + trial)
            positions += np.random.uniform(-0.02, 0.02, size=positions.shape)
        else:
            # Random initialization with varying seeds
            np.random.seed(42 + trial * 7 + 100)
            positions = np.random.uniform(-0.45, 0.45, size=(N, 3))
        
        positions = np.clip(positions, -0.49, 0.49)
        
        # Run L-BFGS-B optimization
        # We need a function that returns only energy for scipy if we don't use jac, 
        # but we have grad. Let's wrap it.
        
        def fun(x):
            e, g = energy_and_grad(x)
            return e
        
        def jac(x):
            e, g = energy_and_grad(x)
            return g
        
        result = minimize(
            fun,
            positions.ravel(),
            jac=jac,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 100}
        )
        
        pos = result.x.reshape(N, 3)
        
        # Calculate score
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        iu = np.triu_indices(N, k=1)
        min_dist = np.sqrt(dist_sq[iu].min())
        
        wall_clearance = 0.5 - np.max(np.abs(pos))
        
        current_score = min(min_dist / 2.0, wall_clearance)
        
        if current_score > best_score:
            best_score = current_score
            best_result = pos.copy()
    
    # Final polish with tighter tolerances
    if best_result is not None:
        result = minimize(
            fun,
            best_result.ravel(),
            jac=jac,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 200}
        )
        best_result = result.x.reshape(N, 3)
    
    # Ensure validity
    best_result = np.clip(best_result, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_result
# EVOLVE-BLOCK-END

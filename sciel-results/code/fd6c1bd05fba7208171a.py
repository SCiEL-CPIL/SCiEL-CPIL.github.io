"""Bermudan put exercise policy via high-resolution backward induction on a log-grid."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 138.0  # Safety margin before 150s kill

    # Extract parameters from contract
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)

    dt = T / n_exercise
    disc = np.exp(-r * dt)

    # Grid setup: log-spaced grid with finer resolution
    n_grid = 16384
    # Range: wide enough to cover relevant prices. Put is relevant for S < K.
    # Low end: 0.005*K for better tail coverage. High end: 10*K.
    log_S_min = np.log(0.005 * K)
    log_S_max = np.log(10.0 * K)
    log_grid = np.linspace(log_S_min, log_S_max, n_grid)
    S_grid = np.exp(log_grid)

    # Gauss-Legendre quadrature on a truncated normal range
    # We integrate over z in [-L, L] where the normal density is negligible beyond
    # Use adaptive range based on sigma*sqrt(dt)
    vol = sigma * np.sqrt(dt)
    mu = (r - 0.5 * sigma**2) * dt
    
    # Truncation range: 6 sigma covers 99.99994% of the normal
    L = 6.0
    n_quad = 128  # Higher order Gauss-Legendre for better accuracy
    
    # Gauss-Legendre nodes and weights on [-1, 1]
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_quad)
    
    # Map to [-L, L]
    quad_nodes = L * gl_nodes
    quad_weights = 0.5 * L * gl_weights
    
    # Normalize by the truncated normal integral
    # P(-L < Z < L) = erf(L/sqrt(2))
    from math import erf, sqrt
    norm_factor = erf(L / sqrt(2.0))
    quad_weights = quad_weights / norm_factor

    # Precompute the shift in log-space for each quadrature node
    # log(S_next) = log(S_t) + mu + vol * Z
    log_shifts = mu + vol * quad_nodes  # shape (n_quad,)

    def compute_continuation(V_next, lg=None, vg=None):
        """
        Compute continuation value on the grid given the value function at next step.
        V_next: array of shape (n_grid,)
        Returns: array of shape (n_grid,)
        """
        if lg is None:
            lg = log_grid
        if vg is None:
            vg = V_next
        
        n_local = len(lg)
        # Create the target log positions using broadcasting
        quad_pos = lg[:, None] + log_shifts[None, :]
        
        # Flatten for interpolation
        quad_pos_flat = quad_pos.ravel()
        
        # Interpolate V_next at these positions
        V_interp_flat = np.interp(quad_pos_flat, lg, vg)
        
        # Reshape back
        V_interp = V_interp_flat.reshape(n_local, n_quad)
        
        # Compute weighted sum
        cont = V_interp @ quad_weights
        
        return cont

    # Backward induction
    V = np.empty((n_exercise, n_grid))
    
    # Last exercise date: intrinsic value
    intrinsic = np.maximum(K - S_grid, 0.0)
    V[-1] = intrinsic

    # Store continuation values for policy determination
    cont_values = [None] * n_exercise

    for i in range(n_exercise - 2, -1, -1):
        if time.time() > deadline - 12.0:
            # Fallback: fill remaining with intrinsic value
            for j in range(i, -1, -1):
                V[j] = intrinsic
                cont_values[j] = np.zeros_like(intrinsic)
            break
        
        # Compute continuation value
        cont = compute_continuation(V[i+1])
        cont_values[i] = cont
        
        # Optimal value
        V[i] = np.maximum(intrinsic, disc * cont)

    # Refinement pass: refine the region near the exercise boundary
    # For each time step, find the approximate boundary and create a finer local grid
    if time.time() < deadline - 30.0:
        for i in range(n_exercise - 2, -1, -1):
            if time.time() > deadline - 15.0:
                break
            
            # Find approximate boundary: where intrinsic == disc * cont
            cont = cont_values[i]
            if cont is None:
                continue
            
            # Find where the difference changes sign
            diff = intrinsic - disc * cont
            sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
            
            if len(sign_changes) == 0:
                continue
            
            # Use the first sign change as the approximate boundary
            idx = sign_changes[0]
            S_boundary = S_grid[idx]
            
            # Create a finer local grid around the boundary
            # Range: [S_boundary * 0.5, S_boundary * 1.5]
            local_S_min = max(S_boundary * 0.5, 0.01 * K)
            local_S_max = min(S_boundary * 1.5, 10.0 * K)
            
            if local_S_min >= local_S_max:
                continue
            
            n_local = 4096
            log_local_min = np.log(local_S_min)
            log_local_max = np.log(local_S_max)
            log_local_grid = np.linspace(log_local_min, log_local_max, n_local)
            local_S_grid = np.exp(log_local_grid)
            
            # Interpolate V[i+1] onto the local grid
            V_next_local = np.interp(log_local_grid, log_grid, V[i+1])
            
            # Compute continuation on the local grid
            cont_local = compute_continuation(V_next_local, log_local_grid, V_next_local)
            
            # Interpolate the refined continuation back onto the main grid
            # Only update points that fall within the local grid range
            mask = (log_grid >= log_local_min) & (log_grid <= log_local_max)
            if np.any(mask):
                cont_values[i][mask] = np.interp(log_grid[mask], log_local_grid, cont_local)
                # Update V[i] with the refined continuation
                V[i][mask] = np.maximum(intrinsic[mask], disc * cont_values[i][mask])

    class ExercisePolicy:
        def __init__(self, cont_values, log_grid, K, n_exercise, disc):
            self.cont_values = cont_values
            self.log_grid = log_grid
            self.K = K
            self.n_exercise = n_exercise
            self.disc = disc

        def exercise(self, current_state, time_idx):
            """
            Determine exercise decision for given states at a specific time index.
            """
            S = np.asarray(current_state, dtype=float)
            scalar_input = S.ndim == 0
            if scalar_input:
                S = S.reshape(1)
            
            n_states = S.shape[0]
            
            if time_idx < 0 or time_idx >= self.n_exercise:
                return np.zeros(n_states, dtype=bool) if not scalar_input else False
            
            # Last exercise date: always exercise if in the money
            if time_idx == self.n_exercise - 1:
                result = S < self.K
                return result[0] if scalar_input else result
            
            cont = self.cont_values[time_idx]
            if cont is None:
                # Fallback
                result = S < self.K
                return result[0] if scalar_input else result
            
            # Interpolate continuation value
            log_S = np.log(S + 1e-15)
            cont_interp = np.interp(log_S, self.log_grid, cont)
            
            # Intrinsic value
            intrinsic = np.maximum(self.K - S, 0.0)
            
            # Exercise if intrinsic >= disc * continuation
            result = intrinsic >= self.disc * cont_interp
            
            return result[0] if scalar_input else result

    return ExercisePolicy(cont_values, log_grid, K, n_exercise, disc)
# EVOLVE-BLOCK-END

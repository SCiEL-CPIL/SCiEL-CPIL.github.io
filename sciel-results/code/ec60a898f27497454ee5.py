"""Bermudan put exercise policy via high-resolution backward induction on a log-grid."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Extract parameters from contract
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)

    dt = T / n_exercise
    disc = np.exp(-r * dt)

    # Grid setup: log-spaced grid
    # Increase resolution for better boundary capture
    n_grid = 8192
    # Range: wide enough to cover relevant prices. Put is relevant for S < K.
    # Low end: 0.01*K. High end: 10*K (continuation value is 0 there).
    log_S_min = np.log(0.01 * K)
    log_S_max = np.log(10.0 * K)
    log_grid = np.linspace(log_S_min, log_S_max, n_grid)
    S_grid = np.exp(log_grid)

    # Gauss-Hermite quadrature for E[f(Z)] where Z ~ N(0,1)
    # E[f(Z)] = 1/sqrt(pi) * sum w_i f(sqrt(2) x_i)
    from numpy.polynomial.hermite import hermgauss
    n_quad = 96  # Higher order for better accuracy in tails
    gh_nodes, gh_weights = hermgauss(n_quad)
    quad_nodes = np.sqrt(2.0) * gh_nodes
    quad_weights = gh_weights / np.sqrt(np.pi)

    # Log-return parameters
    mu = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    # Precompute the shift in log-space for each quadrature node
    # log(S_next) = log(S_t) + mu + vol * Z
    # Z = quad_nodes[j]
    log_shifts = mu + vol * quad_nodes  # shape (n_quad,)

    def compute_continuation(V_next):
        """
        Compute continuation value on the grid given the value function at next step.
        V_next: array of shape (n_grid,)
        Returns: array of shape (n_grid,)
        """
        n_grid_local = len(log_grid)
        # We need to interpolate V_next at log_grid[i] + log_shifts[j]
        # To avoid huge memory usage (8192 x 96 is ~800k floats, which is fine),
        # we can do it in chunks if necessary, but 8192*96 is small enough.
        
        # Create the target log positions
        # quad_pos[i, j] = log_grid[i] + log_shifts[j]
        # Using broadcasting: (n_grid, 1) + (1, n_quad)
        quad_pos = log_grid[:, None] + log_shifts[None, :]
        
        # Flatten for interpolation
        quad_pos_flat = quad_pos.ravel()
        
        # Interpolate V_next at these positions
        # V_next is defined on log_grid
        V_interp_flat = np.interp(quad_pos_flat, log_grid, V_next)
        
        # Reshape back
        V_interp = V_interp_flat.reshape(n_grid_local, n_quad)
        
        # Compute weighted sum
        # cont[i] = sum_j quad_weights[j] * V_interp[i, j]
        cont = V_interp @ quad_weights
        
        return cont

    # Backward induction
    V = np.empty((n_exercise, n_grid))
    
    # Last exercise date: intrinsic value
    V[-1] = np.maximum(K - S_grid, 0.0)

    # Store continuation values for policy determination
    cont_values = [None] * n_exercise

    for i in range(n_exercise - 2, -1, -1):
        if time.time() > deadline - 10.0:
            # Fallback: fill remaining with intrinsic value
            for j in range(i, -1, -1):
                V[j] = np.maximum(K - S_grid, 0.0)
            # Set cont_values to something that makes exercise always true if ITM
            # or use a simple heuristic.
            for j in range(i, -1, -1):
                cont_values[j] = np.maximum(K - S_grid, 0.0) * 0.0 # Force exercise if ITM?
                # Actually, if we run out of time, we should just use the intrinsic value logic
                # cont = 0 means we always exercise if intrinsic > 0.
            break
        
        # Compute continuation value
        cont = compute_continuation(V[i+1])
        cont_values[i] = cont
        
        # Intrinsic value
        intrinsic = np.maximum(K - S_grid, 0.0)
        
        # Optimal value
        V[i] = np.maximum(intrinsic, disc * cont)

    class ExercisePolicy:
        def __init__(self, cont_values, log_grid, K, n_exercise):
            self.cont_values = cont_values
            self.log_grid = log_grid
            self.K = K
            self.n_exercise = n_exercise

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
            
            # Exercise if intrinsic >= continuation value
            # Note: V[i] = max(intrinsic, disc * cont).
            # We stored cont (undiscounted expectation).
            # So we compare intrinsic vs disc * cont?
            # In the backward step: V[i] = max(intrinsic, disc * E[V[i+1]]).
            # The decision is: Exercise if intrinsic >= disc * E[V[i+1]].
            # So we need to compare intrinsic with disc * cont_interp.
            
            result = intrinsic >= disc * cont_interp
            
            return result[0] if scalar_input else result

    return ExercisePolicy(cont_values, log_grid, K, n_exercise)
# EVOLVE-BLOCK-END

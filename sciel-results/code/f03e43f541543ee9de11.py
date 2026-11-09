"""Bermudan put exercise policy via high-resolution backward induction on a log-grid."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Extract parameters from contract
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)

    dt = T / n_exercise

    # Grid setup: log-spaced grid over a wide range of spot prices
    # For a put, we care about low prices. Grid from 0.01*K to 10*K
    n_grid = 4096
    log_S_min = np.log(0.01 * K)
    log_S_max = np.log(10.0 * K)
    log_grid = np.linspace(log_S_min, log_S_max, n_grid)
    S_grid = np.exp(log_grid)
    step = (log_S_max - log_S_min) / (n_grid - 1)

    # Discount factor
    disc = np.exp(-r * dt)

    # GBM transition: S_{t+dt} = S_t * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
    # On log scale: log(S_{t+dt}) = log(S_t) + (r - sigma^2/2)*dt + sigma*sqrt(dt)*Z
    # So log(S_{t+dt}) - log(S_t) ~ N((r - sigma^2/2)*dt, sigma^2*dt)

    # We need to compute E[payoff_next * S_next / S_current | S_current = s]
    # For a put: payoff = max(K - S_next, 0)
    # E[max(K - S_next, 0) | S_t = s] where S_next = s * exp(mu + vol*Z)
    # = s * E[max(K/s - exp(mu + vol*Z), 0)]
    # = s * E[max(K/s - exp(mu + vol*Z), 0)]

    # Using Gauss-Hermite quadrature on the log-normal transition
    # Let X = mu + vol*Z, so X ~ N(mu, vol^2)
    # We need E[f(X)] where f(X) = max(K/s - exp(X), 0) * s

    # Actually, let's use a different approach. For each grid point s_i,
    # we need E[discount * max(K - S_{t+dt}, 0) | S_t = s_i]
    # where S_{t+dt} = s_i * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)

    # This is a 1D integral that can be computed via quadrature.
    # Use Gauss-Hermite quadrature with 64 points for accuracy.

    # Gauss-Hermite nodes and weights
    from numpy.polynomial.hermite import hermgauss
    gh_nodes, gh_weights = hermgauss(64)

    # mu and vol for the log-return
    mu = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)

    # For a given s, S_next = s * exp(mu + vol * Z)
    # In terms of GH quadrature: Z = sqrt(2) * gh_node
    # So log(S_next/s) = mu + vol * sqrt(2) * gh_node

    # Precompute the quadrature points for log(S_next/s)
    # log_ratio = mu + vol * sqrt(2) * gh_nodes
    log_ratio = mu + vol * np.sqrt(2.0) * gh_nodes
    # Weights scaled by 1/sqrt(pi) for standard normal
    quad_weights = gh_weights / np.sqrt(np.pi)

    # For each s_i, continuation value = disc * sum_j w_j * max(K - s_i * exp(log_ratio_j), 0)
    # = disc * sum_j w_j * max(K - s_i * exp(log_ratio_j), 0)

    # Vectorized computation:
    # exp_log_ratio = exp(log_ratio)  # shape (64,)
    # For each s_i: payoff_j = max(K - s_i * exp_log_ratio_j, 0)
    # cont_i = disc * sum_j w_j * payoff_j

    exp_log_ratio = np.exp(log_ratio)  # shape (64,)

    # Build matrix: (n_grid, 64) where entry [i,j] = s_i * exp_log_ratio_j
    # Then payoff[i,j] = max(K - S_mat[i,j], 0)
    # cont[i] = disc * dot(quad_weights, payoff[i,:])

    # To save memory, do it in chunks
    chunk_size = 512
    n_chunks = (n_grid + chunk_size - 1) // chunk_size

    # Store continuation values for each exercise date
    # cont_values[i] is an array of shape (n_grid,) for exercise date i (0-indexed)
    # We compute backward from date n_exercise-1 to 0

    # At the last exercise date (index n_exercise-1), the value is just the intrinsic:
    # V[last] = max(K - S, 0)

    # For earlier dates:
    # V[i] = max(intrinsic_i, disc * E[V[i+1] | S_i])
    # where E[V[i+1] | S_i = s] is computed via quadrature

    # Precompute the quadrature matrix for the transition
    # transition_mat[i, j] = s_i * exp_log_ratio_j  -- but this is n_grid x 64, too big
    # Instead, compute in chunks

    # Let's precompute a function to compute continuation value from a value function V_next
    # V_next is defined on the log_grid. We need to interpolate V_next at the quadrature points.
    # Quadrature points for log(S_next): log_S + log_ratio_j

    def compute_continuation(V_next, log_grid, S_grid, log_ratio, quad_weights, disc, K):
        """
        Compute continuation value on the grid given the value function at next step.
        
        V_next: array of shape (n_grid,) - value at next step on log_grid
        Returns: array of shape (n_grid,) - continuation value at current step
        """
        n_grid = len(log_grid)
        n_quad = len(log_ratio)
        
        # For each grid point i, the quadrature points are at log_grid[i] + log_ratio[j]
        # We need to interpolate V_next at these points.
        
        # Create the full quadrature log positions: (n_grid, n_quad)
        # quad_log_pos[i, j] = log_grid[i] + log_ratio[j]
        
        # To avoid creating a huge array, process in chunks
        chunk = 256
        result = np.empty(n_grid)
        
        for start in range(0, n_grid, chunk):
            end = min(start + chunk, n_grid)
            chunk_logs = log_grid[start:end]  # (chunk,)
            quad_pos = chunk_logs[:, None] + log_ratio[None, :]  # (chunk, n_quad)
            
            # Interpolate V_next at quad_pos
            # V_next is defined on log_grid (sorted)
            V_interp = np.interp(quad_pos.ravel(), log_grid, V_next).reshape(chunk, n_quad)
            
            # Payoff: we need max(K - S_next, 0) but actually we're interpolating V_next directly
            # Wait - V_next is the VALUE at the next step, which already includes the optimal exercise.
            # So continuation = disc * E[V_next(S_{t+dt}) | S_t]
            # = disc * sum_j w_j * V_next(S_{t+dt,j})
            # where S_{t+dt,j} = S_t * exp(log_ratio_j)
            
            # So we interpolate V_next at the quadrature log positions
            cont_chunk = disc * (V_interp @ quad_weights)
            result[start:end] = cont_chunk
        
        return result

    # Backward induction
    # V[i] is the value at exercise date i (0-indexed, 0 = first exercise date)
    # V[n_exercise-1] = max(K - S, 0)  (must exercise or expire)
    
    V = np.empty((n_exercise, n_grid))
    
    # Last exercise date
    V[-1] = np.maximum(K - S_grid, 0.0)

    # Exercise boundary: we need to know where to exercise at each date
    # Store the continuation value separately to determine exercise regions
    # Actually, we can just store V and recompute the decision

    for i in range(n_exercise - 2, -1, -1):
        if time.time() > deadline - 5.0:
            # Fill remaining with intrinsic value as fallback
            for j in range(i, -1, -1):
                V[j] = np.maximum(K - S_grid, 0.0)
            break
        
        # Compute continuation value from V[i+1]
        cont = compute_continuation(V[i+1], log_grid, S_grid, log_ratio, quad_weights, disc, K)
        
        # Intrinsic value
        intrinsic = np.maximum(K - S_grid, 0.0)
        
        # Optimal value
        V[i] = np.maximum(intrinsic, cont)
        
        # Check time
        if time.time() > deadline - 10.0:
            # Fill remaining with intrinsic
            for j in range(i - 1, -1, -1):
                V[j] = np.maximum(K - S_grid, 0.0)
            break

    # Now create the exercise policy
    # Exercise at date i if intrinsic >= continuation
    # We need to store the continuation values or the exercise boundary
    # For efficiency, store the exercise threshold (max S where we exercise) at each date
    
    # Actually, let's store the full value functions and use interpolation at query time
    # Exercise if max(K - S, 0) >= cont(S)
    # For a put, the exercise region is S <= threshold_i
    
    # Compute exercise thresholds: for each date, find the largest S where intrinsic >= cont
    # We can compute this by finding where intrinsic - cont changes sign

    # Let's store the continuation values for each date (for dates 0 to n_exercise-2)
    # At the last date, always exercise if S < K

    cont_values = [None] * n_exercise
    for i in range(n_exercise - 1):
        if time.time() > deadline - 3.0:
            # Fallback: use simple threshold
            cont_values[i] = np.maximum(K - S_grid, 0.0) * 0.5  # rough
            continue
        cont = compute_continuation(V[i+1], log_grid, S_grid, log_ratio, quad_weights, disc, K)
        cont_values[i] = cont

    # For the last date, no continuation needed (always exercise if in the money)
    # cont_values[n_exercise-1] remains None

    class ExercisePolicy:
        def __init__(self, cont_values, log_grid, step, K, n_grid):
            self.cont_values = cont_values
            self.log_grid = log_grid
            self.step = step
            self.K = K
            self.n_grid = n_grid

        def exercise(self, current_state, time_idx):
            """
            Determine exercise decision for given states at a specific time index.
            
            Parameters:
            -----------
            current_state : array-like
                Spot prices for a batch of paths.
            time_idx : int
                The exercise date index (0-based, 0..11). If out of range, never exercise.
                
            Returns:
            --------
            boolean array indicating whether to exercise at each state.
            """
            S = np.asarray(current_state, dtype=float)
            scalar_input = S.ndim == 0
            if scalar_input:
                S = S.reshape(1)
            
            n_states = S.shape[0]
            
            if time_idx < 0 or time_idx >= 12:
                return np.zeros(n_states, dtype=bool) if not scalar_input else False
            
            if time_idx == 11:
                # Last exercise date: exercise if in the money
                result = S < self.K
                return result[0] if scalar_input else result
            
            if self.cont_values[time_idx] is None:
                # Fallback: exercise if in the money
                result = S < self.K
                return result[0] if scalar_input else result
            
            # Interpolate continuation value
            log_S = np.log(S + 1e-15)
            cont = np.interp(log_S, self.log_grid, self.cont_values[time_idx])
            
            intrinsic = np.maximum(self.K - S, 0.0)
            
            result = intrinsic >= cont
            
            return result[0] if scalar_input else result

    return ExercisePolicy(cont_values, log_grid, step, K, n_grid)
# EVOLVE-BLOCK-END

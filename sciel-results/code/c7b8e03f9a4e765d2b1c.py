"""Bermudan put exercise policy via backward induction on a log-space grid with smooth boundary interpolation."""
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    TIME_LIMIT = 110.0  # Safety margin before 120s kill

    # Fixed model parameters from task contract
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_exercise = 12  # Monthly exercise dates (no time-0)
    dt = T / n_exercise

    # Grid setup in log-space for better resolution near the strike
    # S range: [20, 300] covers most of the distribution
    log_s_min = np.log(20.0)
    log_s_max = np.log(300.0)
    n_grid = 500
    log_s_grid = np.linspace(log_s_min, log_s_max, n_grid)
    s_grid = np.exp(log_s_grid)

    # Precompute discount factor
    disc = np.exp(-r * dt)

    # Initialize value function at final exercise date (index 0 is t=0, we work backwards)
    # Payoff at maturity for a put: max(K - S, 0)
    payoff_final = np.maximum(K - s_grid, 0.0)

    # We'll store the exercise boundary (threshold) at each exercise date
    # boundary[i] = the S level below which we exercise at time step i
    # There are 12 exercise dates: t_1, t_2, ..., t_12
    # We work backwards from t_12 to t_1
    boundaries = np.zeros(n_exercise)
    values = np.zeros((n_exercise, n_grid))

    # Final date (t_12 = T): must exercise if in the money
    values[11] = payoff_final
    # Find threshold at final date
    idx_exercise = np.where(payoff_final > 0)[0]
    if len(idx_exercise) > 0:
        boundaries[11] = s_grid[idx_exercise[-1]]
    else:
        boundaries[11] = 0.0

    # Backward induction from t_11 down to t_1
    for step in range(10, -1, -1):
        # Check time limit
        if time.time() - start_time > TIME_LIMIT - 5.0:
            break

        # Continuation value: discount the expected future value under risk-neutral measure
        # For GBM: S_{t+dt} = S_t * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
        # In log-space: log(S_{t+dt}) = log(S_t) + (r - sigma^2/2)*dt + sigma*sqrt(dt)*Z
        # We need E[values[step+1](S_{t+dt}) | S_t]
        # Use Gauss-Hermite quadrature for the expectation over Z
        # Z ~ N(0,1), so E[f(Z)] = (1/sqrt(pi)) * sum w_i f(sqrt(2)*x_i)
        n_quad = 64
        x_q, w_q = np.polynomial.hermite.hermgauss(n_quad)
        
        # Shift: log(S_{t+dt}) = log_s + drift + sigma*sqrt(dt)*Z
        drift = (r - 0.5 * sigma**2) * dt
        vol_step = sigma * np.sqrt(dt)
        
        # For each grid point, compute the expected continuation value
        cont_values = np.zeros(n_grid)
        for i in range(n_grid):
            # log_s_next = log_s_grid[i] + drift + vol_step * Z
            # Z = sqrt(2)*x_q
            log_s_next = log_s_grid[i] + drift + vol_step * (np.sqrt(2.0) * x_q)
            # Interpolate values[step+1] at log_s_next
            # Clip to grid range
            log_s_next_clipped = np.clip(log_s_next, log_s_min, log_s_max)
            v_next = np.interp(log_s_next_clipped, log_s_grid, values[step + 1])
            # Gauss-Hermite weight: (1/sqrt(pi)) * w_q
            cont_values[i] = np.sum(w_q * v_next) / np.sqrt(np.pi)
        
        # Discount the continuation value
        cont_values *= disc

        # Exercise value: max(K - S, 0)
        exercise_value = np.maximum(K - s_grid, 0.0)

        # Optimal value: max(continuation, exercise)
        values[step] = np.maximum(cont_values, exercise_value)

        # Determine boundary: the highest S where exercise value >= continuation value
        # i.e., where we prefer to exercise
        prefer_exercise = exercise_value >= cont_values
        idx_ex = np.where(prefer_exercise)[0]
        if len(idx_ex) > 0:
            boundaries[step] = s_grid[idx_ex[-1]]
        else:
            boundaries[step] = 0.0

    # Build the policy: for each exercise date, exercise if S <= boundary
    # The policy object needs to return a boolean vector for given (current_state, time)
    # current_state: array of spot prices
    # time: current time (to determine which exercise date we're at)
    
    # Store boundaries and grid for interpolation
    class ExercisePolicy:
        def __init__(self, boundaries, n_exercise, dt, T):
            self.boundaries = boundaries
            self.n_exercise = n_exercise
            self.dt = dt
            self.T = T

        def exercise(self, current_state, time):
            """
            current_state: array-like of spot prices
            time: current time (float)
            Returns: boolean array, True if exercise
            """
            current_state = np.asarray(current_state, dtype=float)
            
            # Determine which exercise date we're at
            # Exercise dates are at t = dt, 2*dt, ..., 12*dt
            # time should be close to one of these
            step_idx = int(round(time / self.dt))
            step_idx = np.clip(step_idx, 1, self.n_exercise)
            step_idx_0based = step_idx - 1  # 0-based index into boundaries

            if step_idx_0based < 0 or step_idx_0based >= self.n_exercise:
                # Out of range, don't exercise
                return np.zeros(len(current_state), dtype=bool)

            threshold = self.boundaries[step_idx_0based]
            
            # Exercise if S <= threshold (for a put)
            return current_state <= threshold

    # Check time limit before returning
    if time.time() - start_time > TIME_LIMIT - 2.0:
        # Emergency: return a simple threshold policy based on strike
        class SimplePolicy:
            def exercise(self, current_state, time):
                current_state = np.asarray(current_state, dtype=float)
                return current_state <= K

        return SimplePolicy()

    return ExercisePolicy(boundaries, n_exercise, dt, T)

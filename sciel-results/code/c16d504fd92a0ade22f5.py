"""Bermudan exercise-policy research using Gauss-Hermite quadrature and bisection thresholds."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.special import roots_hermite

class BermudanExercisePolicy:
    """Bermudan put exercise policy using high-accuracy Gauss-Hermite quadrature."""

    def __init__(self, K, r, sigma, T, n_exercise_dates, exercise_times):
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.n_exercise_dates = n_exercise_dates
        self.exercise_times = np.asarray(exercise_times, dtype=float)
        self._compute_thresholds()

    def _compute_thresholds(self):
        """Compute critical stock price thresholds via backward induction with GH quadrature."""
        K = self.K
        r = self.r
        sigma = self.sigma
        n_ex = self.n_exercise_dates
        ex_times = self.exercise_times

        # Grid for storing V values (used for interpolation if needed, but mainly for GH eval context)
        # We don't actually need a huge grid if we compute GH on the fly for bisection.
        # However, we need to represent V_{i+1}(S) for the GH integration.
        # V_{i+1}(S) is piecewise: max(K-S, C_{i+1}(S)).
        # We will store the thresholds and the continuation value function implicitly.
        
        # For GH quadrature, we need to evaluate V_{i+1}(S_next) where S_next = S * exp(drift + vol * z).
        # V_{i+1}(S) is known analytically in terms of the threshold S^*_{i+1} and the continuation value C_{i+1}(S).
        # C_{i+1}(S) is the expected discounted value from the next step.
        # This creates a recursive dependency. 
        # Standard approach: Discretize S into a grid, compute V on grid, interpolate.
        # To maintain high accuracy, we use a fine grid for V, and Bisection for thresholds.
        
        n_grid = 2000
        S_max = 250.0
        S_min = 0.01
        S_grid = np.linspace(S_min, S_max, n_grid)

        # Gauss-Hermite nodes and weights
        n_z = 50
        x, w = roots_hermite(n_z)
        # Transform to standard normal integral:
        # Integral = 1/sqrt(pi) * sum(w_i * f(sqrt(2)*x_i))
        z_vals = x * np.sqrt(2)
        z_weights = w / np.sqrt(np.pi)

        # V[i] will be the value function at exercise date i on S_grid
        # thresholds[i] will be the critical price at exercise date i
        thresholds = np.zeros(n_ex)
        
        # Last exercise date: V = intrinsic
        V = np.maximum(K - S_grid, 0.0)
        thresholds[n_ex - 1] = K

        # Backward induction
        for i in range(n_ex - 1, 0, -1):
            t_curr = ex_times[i - 1]
            t_next = ex_times[i]
            dt = t_next - t_curr

            if dt <= 0:
                thresholds[i - 1] = thresholds[i]
                # V remains intrinsic if no time passes? No, if dt=0, it's the same date.
                # But usually dt > 0.
                continue

            drift = (r - 0.5 * sigma ** 2) * dt
            vol_term = sigma * np.sqrt(dt)
            discount = np.exp(-r * dt)

            # Compute Continuation Value C_i(S) for all S in S_grid
            # C_i(S) = discount * E[ V_{i+1}(S * exp(drift + vol * Z)) ]
            
            # We need V_{i+1} on the grid. We have it.
            # For each S in S_grid, we compute the expectation.
            # S_next_matrix = S_grid[:, np.newaxis] * exp(drift + vol_term * z_vals[np.newaxis, :])
            # V_next_vals = interp(S_next_matrix, S_grid, V)
            # C_i = discount * sum(V_next_vals * z_weights, axis=1)
            
            # Vectorized computation
            S_next = S_grid[:, np.newaxis] * np.exp(drift + vol_term * z_vals[np.newaxis, :])
            V_next = np.interp(S_next, S_grid, V, left=0.0, right=0.0)
            cont_val = discount * np.sum(V_next * z_weights[np.newaxis, :], axis=1)

            # Update V on grid
            exercise_val = np.maximum(K - S_grid, 0.0)
            V = np.maximum(exercise_val, cont_val)

            # Find threshold using Bisection
            # We want S* such that (K - S*) = C_i(S*)
            # Define h(S) = (K - S) - C_i(S)
            # h(S) is decreasing in S (K-S decreases, C_i(S) increases? No, C_i is convex, but generally increases as S moves away from 0? 
            # Actually, for a put, C_i(S) is the value of waiting. As S increases, the put is less likely to be ITM, so C_i decreases.
            # So K-S decreases (slope -1), C_i decreases (slope > -1? or < -1?).
            # At S=0, K-S=K, C_i(0) < K (usually). So h(0) > 0.
            # At S=K, K-S=0, C_i(K) > 0. So h(K) < 0.
            # So there is a root in (0, K).
            
            # We need a function to evaluate C_i(S) for any S, not just on the grid.
            # We can interpolate cont_val from the grid.
            def c_val(S_point):
                return np.interp(S_point, S_grid, cont_val, left=cont_val[0], right=cont_val[-1])
            
            def h_func(S_point):
                return (K - S_point) - c_val(S_point)
            
            # Bisection
            lo = 0.0
            hi = K
            # Ensure h(lo) > 0 and h(hi) < 0
            # If h(K) > 0, threshold is > K? No, for a put, you never exercise if S > K (unless interest rates are weird, but r>0).
            # If h(0) < 0, threshold is < 0? No, always exercise at S=0.
            
            # Check bounds
            h_lo = h_func(lo)
            h_hi = h_func(hi)
            
            if h_lo < 0:
                threshold = 0.0
            elif h_hi > 0:
                threshold = K # Should not happen for standard put
            else:
                for _ in range(50): # 50 iterations gives ~1e-15 precision
                    mid = (lo + hi) / 2.0
                    h_mid = h_func(mid)
                    if h_mid > 0:
                        lo = mid
                    else:
                        hi = mid
                threshold = (lo + hi) / 2.0
                
            thresholds[i - 1] = threshold

        self.thresholds = thresholds

    def exercise(self, current_state, time):
        """Return boolean array: True if should exercise at given time for each state."""
        current_state = np.asarray(current_state, dtype=float)
        # Find the exercise date index closest to the given time
        idx = int(np.argmin(np.abs(self.exercise_times - time)))
        threshold = self.thresholds[idx]
        return current_state < threshold


def fit_exercise_policy(training_paths, contract):
    """Fit a Bermudan exercise policy using Gauss-Hermite quadrature and bisection."""
    start_time = time.time()

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise_dates = contract.get('n_exercise_dates', 12)

    # Exercise dates: 12 monthly, no time-0
    exercise_times = np.linspace(T / n_exercise_dates, T, n_exercise_dates)

    policy = BermudanExercisePolicy(K, r, sigma, T, n_exercise_dates, exercise_times)

    # Safety check
    elapsed = time.time() - start_time
    if elapsed > 100.0:
        pass

    return policy
# EVOLVE-BLOCK-END

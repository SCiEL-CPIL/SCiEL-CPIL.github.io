"""Bermudan put exercise policy via Iterative Longstaff-Schwartz Monte Carlo method."""
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

    # Exercise dates: 12 monthly, no time-0
    dt = T / n_exercise

    # Get training paths shape: (n_paths, n_steps)
    n_paths, n_steps = training_paths.shape

    if n_steps == n_exercise:
        exercise_states = training_paths
    elif n_steps == n_exercise + 1:
        exercise_states = training_paths[:, 1:]
    else:
        exercise_states = training_paths[:, -n_exercise:]

    n_paths = exercise_states.shape[0]
    discount = np.exp(-r * dt)

    # Feature function: richer basis for better approximation
    def make_features(S):
        n = S.shape[0]
        log_S = np.log(S + 1e-10)
        intrinsic = np.maximum(K - S, 0.0)
        X = np.column_stack([
            np.ones(n),
            S,
            S**2,
            S**3,
            log_S,
            log_S**2,
            log_S**3,
            intrinsic,
            S * log_S,
            S * intrinsic,
        ])
        return X

    # Pre-compute features for all paths at all exercise dates to save time
    # Features[i] is the feature matrix for date i
    features = [None] * n_exercise
    for i in range(n_exercise):
        features[i] = make_features(exercise_states[:, i])

    # Initialize values
    values = np.zeros((n_paths, n_exercise))
    values[:, -1] = np.maximum(K - exercise_states[:, -1], 0.0)

    # Iterative Longstaff-Schwartz
    # We perform multiple passes to refine the exercise region
    n_iterations = 10
    reg_coeffs = [None] * n_exercise
    
    for _iter in range(n_iterations):
        if time.time() > deadline - 10.0:
            break
            
        # Backward pass
        for i in range(n_exercise - 2, -1, -1):
            intrinsic = np.maximum(K - exercise_states[:, i], 0.0)
            target = discount * values[:, i + 1]
            X = features[i]
            
            # Determine "alive" paths: those where continuation was chosen in previous iteration
            # For the first iteration, we assume all are alive (or use intrinsic > 0)
            if _iter == 0:
                mask = np.ones(n_paths, dtype=bool)
            else:
                # In the previous iteration, we stored the decision in values
                # We can reconstruct the "alive" mask by checking if the value was
                # strictly greater than intrinsic (meaning we continued)
                # However, for regression, standard LS uses ITM paths.
                # Let's use ITM paths for regression to estimate continuation.
                mask = intrinsic > 0
            
            # If not enough ITM paths, use all
            if np.sum(mask) < 50:
                mask = np.ones(n_paths, dtype=bool)
            
            X_itm = X[mask, :]
            target_itm = target[mask]
            
            # Ridge Regression for stability
            # lambda = 1e-4 * trace(X^T X) / n_features
            # Simpler: fixed small lambda
            lambda_reg = 1e-4
            try:
                # Solve (X^T X + lambda * I) beta = X^T y
                # We don't penalize the intercept
                X_reg = X_itm.copy()
                # Don't penalize the first column (intercept)
                # Create penalty matrix
                P = np.eye(X_reg.shape[1])
                P[0, 0] = 0.0
                A = X_reg.T @ X_reg + lambda_reg * P
                b = X_reg.T @ target_itm
                coeffs = np.linalg.solve(A, b)
                
                # Predict continuation for all paths
                continuation = X @ coeffs
            except Exception:
                # Fallback to OLS if solve fails
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(X_itm, target_itm, rcond=None)
                    continuation = X @ coeffs
                except Exception:
                    coeffs = np.array([np.mean(target_itm)] + [0.0] * (X.shape[1] - 1))
                    continuation = np.full(n_paths, coeffs[0])
            
            # Update value
            values[:, i] = np.maximum(intrinsic, continuation)
            
            # Store coefficients for the final policy
            # We keep updating this; the last iteration's coefficients are used
            reg_coeffs[i] = coeffs

    # Create the exercise policy object
    class ExercisePolicy:
        def __init__(self, coeffs, K, r, dt, n_exercise):
            self.coeffs = coeffs
            self.K = K
            self.r = r
            self.dt = dt
            self.n_exercise = n_exercise

        def _make_features(self, S):
            if np.isscalar(S):
                S = np.array([S])
            n = S.shape[0]
            log_S = np.log(S + 1e-10)
            intrinsic = np.maximum(self.K - S, 0.0)
            X = np.column_stack([
                np.ones(n),
                S,
                S**2,
                S**3,
                log_S,
                log_S**2,
                log_S**3,
                intrinsic,
                S * log_S,
                S * intrinsic,
            ])
            return X

        def exercise(self, current_state, time_idx):
            if np.isscalar(current_state):
                S = np.array([current_state])
                scalar_input = True
            else:
                S = np.asarray(current_state)
                scalar_input = False
            
            if time_idx >= self.n_exercise:
                result = np.zeros(S.shape, dtype=bool)
                return result[0] if scalar_input else result
            
            if time_idx == self.n_exercise - 1:
                result = S < self.K
                return result[0] if scalar_input else result
            
            if self.coeffs[time_idx] is None:
                result = S < self.K
                return result[0] if scalar_input else result
            
            X = self._make_features(S)
            continuation = X @ self.coeffs[time_idx]
            intrinsic = np.maximum(self.K - S, 0.0)
            
            result = intrinsic >= continuation
            return result[0] if scalar_input else result

    return ExercisePolicy(reg_coeffs, K, r, dt, n_exercise)
# EVOLVE-BLOCK-END

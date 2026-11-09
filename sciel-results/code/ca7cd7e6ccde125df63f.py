import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Extract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = 12  # monthly exercise dates, no time-0

    training_paths = np.asarray(training_paths, dtype=float)
    N, n_tp = training_paths.shape
    dt = T / n_exercise
    discount = np.exp(-r * dt)

    # Build feature matrix: [1, S, S^2] for polynomial regression
    # This captures path-dependent features and avoids grid truncation errors
    S_all = training_paths[:, 1:]  # Shape (N, n_exercise)
    
    # Precompute features for all time steps
    features_list = []
    for k in range(n_exercise):
        S_k = S_all[:, k]
        F_k = np.column_stack([np.ones_like(S_k), S_k, S_k ** 2])
        features_list.append(F_k)

    # Backward induction using Longstaff-Schwartz least squares
    # V_next is the value function at time k+1, represented by regression coefficients
    # We store the value at each path for the next step
    V_next_paths = np.maximum(K - S_all[:, -1], 0.0)  # Maturity value

    # Store thresholds for each exercise date
    thresholds = np.zeros(n_exercise)
    thresholds[n_exercise - 1] = K  # At maturity, exercise if S < K

    # Regression degrees of freedom
    n_features = 3

    for k in range(n_exercise - 1, 0, -1):
        if time.time() > deadline - 2:
            # Not enough time, use simple threshold
            for kk in range(k, 0, -1):
                thresholds[kk - 1] = K * 0.9
            break

        S_k = S_all[:, k]
        F_k = features_list[k]

        # Continuation value: discount * E[V_next_paths | S_k]
        # Regress V_next_paths on F_k
        try:
            # Solve least squares: F_k @ coeffs = V_next_paths
            coeffs, residuals, rank, sv = np.linalg.lstsq(F_k, V_next_paths, rcond=None)
            V_cont_k = F_k @ coeffs
            V_cont_k = discount * V_cont_k
        except np.linalg.LinAlgError:
            # Fallback: simple average
            V_cont_k = np.full_like(S_k, discount * np.mean(V_next_paths))

        # Exercise value
        V_ex_k = np.maximum(K - S_k, 0.0)

        # Optimal value: max(Exercise, Continuation)
        V_k_paths = np.maximum(V_ex_k, V_cont_k)

        # Determine threshold: max S where exercise is optimal
        should_exercise = V_ex_k >= V_cont_k
        if np.any(should_exercise):
            thresholds[k - 1] = S_k[should_exercise].max()
        else:
            thresholds[k - 1] = 0.0

        # Update V_next_paths for next iteration
        V_next_paths = V_k_paths

    # Create policy object
    class BermudanPutPolicy:
        def __init__(self, K, n_exercise, thresholds):
            self.K = K
            self.n_exercise = n_exercise
            self.thresholds = thresholds

        def exercise(self, current_state, time_step):
            current_state = np.asarray(current_state, dtype=float)
            try:
                if isinstance(time_step, (int, np.integer)):
                    idx = int(time_step)
                else:
                    idx = int(np.round(time_step * 12))

                if idx < 1 or idx > self.n_exercise:
                    return np.zeros(len(current_state), dtype=bool)

                # Maturity: exercise if in the money
                if idx == self.n_exercise:
                    return current_state < self.K

                # Use threshold-based decision
                thresh = self.thresholds[idx - 1]
                return current_state <= thresh

            except Exception:
                return np.zeros(len(current_state), dtype=bool)

    return BermudanPutPolicy(K, n_exercise, thresholds)

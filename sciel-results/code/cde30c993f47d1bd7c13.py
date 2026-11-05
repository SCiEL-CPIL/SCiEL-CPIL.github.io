"""Bermudan put exercise policy via Least Squares Monte Carlo (LSM)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using Least Squares Monte Carlo.
    
    Returns an object with .exercise(current_state, time) method that
    returns a boolean vector indicating whether to exercise at each path.
    """
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin before 120s kill

    # Extract parameters from contract
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    
    # training_paths shape: (N_paths, n_exercise) or (N_paths, n_exercise + 1)
    # Columns correspond to t_1, t_2, ..., t_n
    N_paths, N_cols = training_paths.shape
    
    # Handle potential S_0 in training_paths
    if N_cols > n_exercise:
        # Assume first column is S_0, ignore it
        training_paths = training_paths[:, 1:]
        N_cols = n_exercise
    
    dt = T / n_exercise
    discount_factor = np.exp(-r * dt)
    
    # Polynomial degree for regression
    poly_degree = 4
    
    # Store coefficients for each time step (index 0 to n_exercise-1)
    # coeffs[k] corresponds to the polynomial for time t_{k+1}
    poly_coeffs = [None] * n_exercise
    
    # V_next: Value at the next time step (t_{k+1}) for all paths
    # Initialize with payoff at maturity (t_n)
    S_maturity = training_paths[:, -1]
    V_next = np.maximum(K - S_maturity, 0.0)
    
    # Backward induction
    # k goes from n_exercise - 2 down to 0
    # k corresponds to the column index in training_paths
    # Time is t_{k+1}
    for k in range(n_exercise - 2, -1, -1):
        if time.time() > deadline:
            break
            
        S_current = training_paths[:, k]
        
        # Target: Discounted value of continuation
        Y = V_next * discount_factor
        
        # Features: Polynomial in S_current / K
        # Use ITM paths for regression (standard LSM practice)
        itm_mask = S_current < K
        
        if np.sum(itm_mask) < poly_degree + 1:
            # Not enough ITM paths, use all
            itm_mask = np.ones(N_paths, dtype=bool)
            
        S_itm = S_current[itm_mask]
        Y_itm = Y[itm_mask]
        
        # Normalize S
        x = S_itm / K
        
        # Build Vandermonde matrix
        # Basis: 1, x, x^2, ..., x^degree
        X = np.vander(x, N=poly_degree + 1, increasing=True)
        
        # Solve least squares
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(X, Y_itm, rcond=None)
            poly_coeffs[k] = coeffs
        except:
            # Fallback: constant estimate
            mean_y = np.mean(Y_itm) if len(Y_itm) > 0 else 0.0
            poly_coeffs[k] = np.array([mean_y] + [0.0]*poly_degree)
            
        # Compute continuation value for all paths
        x_all = S_current / K
        X_all = np.vander(x_all, N=poly_degree + 1, increasing=True)
        V_cont = X_all @ coeffs
        
        # Ensure V_cont is not negative
        V_cont = np.maximum(V_cont, 0.0)
        
        V_exercise = np.maximum(K - S_current, 0.0)
        
        V_current = np.maximum(V_exercise, V_cont)
        
        V_next = V_current
        
    # Create the exercise policy
    class ExercisePolicy:
        def __init__(self, K, poly_coeffs, n_exercise, T):
            self.K = K
            self.poly_coeffs = poly_coeffs
            self.n_exercise = n_exercise
            self.T = T
        
        def exercise(self, current_state, time):
            """
            current_state: array of S values (shape: (n_paths,))
            time: current time t
            Returns boolean array: True if should exercise at this path
            """
            n_paths = len(current_state)
            result = np.zeros(n_paths, dtype=bool)
            
            # Determine which exercise date we're at based on time.
            # Exercise dates are at t = i/n_exercise for i=1,...,n_exercise
            i = int(round(time * self.n_exercise))
            
            # Clamp i to [1, n_exercise]
            if i < 1:
                return result
            if i > self.n_exercise:
                # At maturity or beyond, exercise if ITM
                return current_state < self.K
            
            ex_idx = i - 1  # 0-indexed
            
            # At the last exercise date (maturity), always exercise if ITM
            if ex_idx == self.n_exercise - 1:
                return current_state < self.K
            
            if ex_idx < len(self.poly_coeffs) and self.poly_coeffs[ex_idx] is not None:
                coeffs = self.poly_coeffs[ex_idx]
                x = current_state / self.K
                # Compute polynomial
                V_cont = np.zeros(n_paths)
                for d in range(len(coeffs)):
                    V_cont += coeffs[d] * (x ** d)
                
                V_cont = np.maximum(V_cont, 0.0)
                V_exercise = np.maximum(self.K - current_state, 0.0)
                
                # Exercise if ITM and Exercise Value > Continuation Value
                result = (V_exercise > V_cont) & (current_state < self.K)
            
            return result
    
    policy = ExercisePolicy(K, poly_coeffs, n_exercise, T)
    return policy

# EVOLVE-BLOCK-END

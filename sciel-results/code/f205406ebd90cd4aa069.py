"""Bermudan exercise-policy research: LSMC with Chebyshev basis and control variates."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import eval_cheby

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using Longstaff-Schwartz Monte Carlo.
    
    Args:
        training_paths: numpy array of shape (N, num_exercise_dates) where each row
                        is a simulated path of spot prices at the 12 exercise dates.
        contract: dict with keys K, r, sigma, T, and possibly num_steps.
    
    Returns:
        An object with an .exercise(current_state, time) method returning a boolean vector.
    """
    start_time = time.time()
    DEADLINE = 120.0  # seconds
    
    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    num_steps = contract.get('num_steps', 12)
    
    # Ensure training_paths is a 2D numpy array
    training_paths = np.asarray(training_paths, dtype=np.float64)
    if training_paths.ndim == 1:
        training_paths = training_paths.reshape(1, -1)
    
    N, total_cols = training_paths.shape
    
    # Determine the number of exercise dates
    # If total_cols == num_steps + 1, index 0 is t=0 (initial), 1..num_steps are exercise dates
    # If total_cols == num_steps, index 0..num_steps-1 are exercise dates t=1..t=num_steps
    if total_cols == num_steps + 1:
        # Columns 1..num_steps correspond to exercise dates 1..num_steps
        S = training_paths[:, 1:]  # shape (N, num_steps)
        offset = 1
    elif total_cols == num_steps:
        # Columns 0..num_steps-1 correspond to exercise dates 1..num_steps
        S = training_paths[:, :]  # shape (N, num_steps)
        offset = 0
    else:
        # Fallback: use last num_steps columns
        S = training_paths[:, -num_steps:]
        offset = total_cols - num_steps
    
    # Number of exercise dates
    M = num_steps  # 12 monthly exercise dates
    
    # Time grid
    dt = T / M
    # Exercise times: t_1 = dt, t_2 = 2*dt, ..., t_M = T
    # Discount factor for one period
    df = np.exp(-r * dt)
    
    # Intrinsic value at exercise date j for a put: max(K - S_j, 0)
    # S has shape (N, M), where S[:, j] is the spot at exercise date j+1 (0-indexed: j=0 is t=dt)
    
    # We'll work backwards from the last exercise date
    # V_M = max(K - S_M, 0)  (at the last exercise date, you must decide to exercise or not,
    #                         but since it's the last date, continuation = 0, so V_M = intrinsic)
    
    # For LSMC, we estimate the continuation value at each date j by regressing
    # the discounted expected payoff from j+1 to M on basis functions of S_j.
    
    # We'll use Chebyshev polynomials of degree up to d as basis functions.
    # Let's use degree 4 or 5 for a good balance.
    degree = 5
    
    # Precompute Chebyshev basis for each path at each exercise date
    # We need to normalize S to [-1, 1] for Chebyshev polynomials.
    # Use a reasonable range for S. For a put with K=100, S is typically in [50, 150].
    # Let's use S_min = 0.5 * K, S_max = 1.5 * K for normalization.
    S_min = 0.5 * K
    S_max = 1.5 * K
    S_norm = (S - S_min) / (S_max - S_min)  # shape (N, M), values in [0, 1]
    # Map to [-1, 1] for Chebyshev
    S_cheb = 2.0 * S_norm - 1.0  # shape (N, M), values in [-1, 1]
    
    # Clip to [-1, 1] to avoid numerical issues
    S_cheb = np.clip(S_cheb, -1.0, 1.0)
    
    # Precompute Chebyshev basis matrices for each exercise date
    # basis[j] has shape (N, degree+1)
    basis = []
    for j in range(M):
        x = S_cheb[:, j]  # shape (N,)
        B = np.ones((N, degree + 1))
        for d in range(degree + 1):
            B[:, d] = eval_cheby(d, x)
        basis.append(B)
    
    # Backward induction
    # V[j] is the estimated value at exercise date j (0-indexed), shape (N,)
    # V[M-1] = intrinsic value at the last exercise date
    V = np.zeros((M, N), dtype=np.float64)
    
    # Last exercise date: V[M-1] = max(K - S[:, M-1], 0)
    V[M-1] = np.maximum(K - S[:, M-1], 0.0)
    
    # For j = M-2 down to 0:
    # Continuation value at j = df * E[V[j+1] | S_j]
    # We regress V[j+1] on basis[j] to get coefficients, then compute the fitted values.
    # Exercise if intrinsic > continuation
    
    coefficients = []
    for j in range(M - 2, -1, -1):
        # Regress V[j+1] on basis[j]
        B_j = basis[j]  # shape (N, degree+1)
        y = V[j+1]  # shape (N,)
        
        # Solve least squares: B_j @ coeffs = y
        # Use np.linalg.lstsq for stability
        coeffs, residuals, rank, sv = np.linalg.lstsq(B_j, y, rcond=None)
        coefficients.append(coeffs)
        
        # Fitted continuation value (before discounting)
        cont_fitted = B_j @ coeffs  # shape (N,)
        
        # Discounted continuation value
        cont_discounted = df * cont_fitted  # shape (N,)
        
        # Intrinsic value at date j
        intrinsic = np.maximum(K - S[:, j], 0.0)  # shape (N,)
        
        # Exercise if intrinsic > continuation
        # V[j] = max(intrinsic, continuation)
        V[j] = np.maximum(intrinsic, cont_discounted)
    
    # Now we have the optimal policy implied by the LSMC estimates.
    # For each exercise date j, the policy is to exercise if intrinsic > continuation.
    # We store the decision boundary or the fitted continuation values.
    
    # For the exercise method, we need to evaluate the policy at arbitrary current states.
    # The policy at date j is: exercise if max(K - S, 0) > df * (B(S) @ coeffs[j])
    # where B(S) is the Chebyshev basis evaluated at S.
    
    # We'll store the coefficients and the parameters needed to evaluate the policy.
    policy_data = {
        'K': K,
        'r': r,
        'sigma': sigma,
        'T': T,
        'M': M,
        'dt': dt,
        'df': df,
        'degree': degree,
        'S_min': S_min,
        'S_max': S_max,
        'coefficients': coefficients,  # list of arrays, one for each j from M-2 down to 0
    }
    
    # Create the policy object
    class BermudanPolicy:
        def __init__(self, data):
            self.data = data
        
        def exercise(self, current_state, time):
            """
            Determine whether to exercise at each path.
            
            Args:
                current_state: numpy array of shape (N,) with spot prices at the current time.
                time: float, the current time (in years).
            
            Returns:
                Boolean array of shape (N,) indicating whether to exercise.
            """
            data = self.data
            K = data['K']
            df = data['df']
            degree = data['degree']
            S_min = data['S_min']
            S_max = data['S_max']
            M = data['M']
            dt = data['dt']
            coefficients = data['coefficients']
            
            # current_state should be a 1D array of shape (N,)
            current_state = np.asarray(current_state, dtype=np.float64)
            if current_state.ndim == 0:
                current_state = current_state.reshape(1)
            N = current_state.shape[0]
            
            # Determine which exercise date we're at
            # time is in years, exercise dates are at dt, 2*dt, ..., M*dt
            # j = round(time / dt) - 1  (0-indexed)
            j = int(round(time / dt)) - 1
            
            # Clamp j to valid range
            if j < 0:
                j = 0
            if j >= M:
                j = M - 1
            
            # If this is the last exercise date, always exercise if intrinsic > 0
            if j == M - 1:
                intrinsic = np.maximum(K - current_state, 0.0)
                return intrinsic > 0.0
            
            # Otherwise, compare intrinsic to continuation
            # Normalize S
            S_norm = (current_state - S_min) / (S_max - S_min)
            S_cheb = 2.0 * S_norm - 1.0
            S_cheb = np.clip(S_cheb, -1.0, 1.0)
            
            # Evaluate Chebyshev basis
            B = np.ones((N, degree + 1))
            for d in range(degree + 1):
                B[:, d] = eval_cheby(d, S_cheb)
            
            # Get coefficients for this exercise date
            # coefficients is stored in reverse order: coefficients[0] is for j=M-2, coefficients[1] for j=M-3, etc.
            # So for exercise date j, the index into coefficients is (M-2) - j
            coeff_idx = (M - 2) - j
            if coeff_idx < 0 or coeff_idx >= len(coefficients):
                # Fallback: just use intrinsic
                intrinsic = np.maximum(K - current_state, 0.0)
                return intrinsic > 0.0
            
            coeffs = coefficients[coeff_idx]
            cont_fitted = B @ coeffs
            cont_discounted = df * cont_fitted
            
            intrinsic = np.maximum(K - current_state, 0.0)
            
            return intrinsic > cont_discounted
    
    return BermudanPolicy(policy_data)

# EVOLVE-BLOCK-END

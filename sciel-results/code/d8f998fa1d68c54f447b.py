import numpy as np
import time
import sys

class ExercisePolicy:
    def __init__(self, coeffs, K, exercise_times):
        self.coeffs = coeffs
        self.K = K
        self.exercise_times = exercise_times
        self.n_steps = len(exercise_times)
        # Precompute basis function powers for efficiency? 
        # No, S varies.
        
    def _basis(self, S):
        # Use normalized log-price for better conditioning
        # S is vector
        # x = ln(S)
        # Basis: 1, x, x^2, ...
        # Let's stick to simple monomials of S/K to keep it simple and robust
        # Or better: Legendre polynomials on [0, 2*K] or similar?
        # Simple monomials of S/K are standard for LS.
        x = S / self.K
        # Degree 5
        return np.column_stack([
            np.ones_like(x),
            x,
            x**2,
            x**3,
            x**4,
            x**5
        ])

    def exercise(self, current_state, time):
        # current_state: array of spots
        # time: scalar time value
        # Find the corresponding step index
        # The exercise times are discrete. We assume 'time' matches one of them or we find the closest.
        # The problem says "12 monthly exercise dates".
        # We should map time to the index in self.exercise_times.
        
        # If time is before first exercise, don't exercise? 
        # Or is time the step index? 
        # "exercise(current_state, time)"
        # Usually time is the current time.
        # Let's find the index i such that exercise_times[i] is closest to time.
        # Or, if time is the step index, use it directly.
        # Given the ambiguity, let's assume time is the actual time value.
        
        # Find closest exercise time
        idx = np.argmin(np.abs(self.exercise_times - time))
        
        # If time is before the first exercise time, we cannot exercise?
        # But the policy is for exercise decisions AT the exercise dates.
        # If called at a non-exercise time, what to do?
        # The contract says "Bermudan put... 12 monthly exercise dates".
        # The evaluator likely calls this at the exercise dates.
        
        S = np.asarray(current_state)
        B = self._basis(S)
        # Continuation value
        # coeffs[idx] is the vector of coefficients for step idx
        # Note: In LS, we regress on the NEXT step's discounted values.
        # So coeffs[i] predicts the value at i+1 discounted to i.
        # Wait, the value at i is max(Payoff_i, Continuation_i).
        # Continuation_i is the regression output.
        
        cont_val = B @ self.coeffs[idx]
        payoff = np.maximum(self.K - S, 0)
        
        # Exercise if payoff > continuation
        return payoff > cont_val

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 135.0  # 15s margin
    
    # Extract parameters
    S0 = contract.get('S0', 100.0)
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    
    # training_paths shape: (N_paths, N_time_steps)
    # We need to identify which columns correspond to the 12 exercise dates.
    # Assuming the paths are generated at the exercise dates.
    # If N_time_steps == n_exercise, then all columns are exercise dates.
    # If not, we need to map.
    # The contract says "12 monthly exercise dates".
    # Let's assume the training_paths are sampled at these 12 dates.
    
    N_paths, N_steps = training_paths.shape
    
    # If N_steps is not 12, we might need to interpolate or assume the last 12?
    # Standard assumption for these tasks: the provided paths are at the exercise dates.
    # Let's verify if N_steps == n_exercise.
    if N_steps != n_exercise:
        # This is a risk. If they don't match, we can't do LS directly without knowing the times.
        # However, usually "training_paths" in these contexts are the paths at the decision points.
        # If N_steps > n_exercise, we might take the last n_exercise? Or every k-th?
        # Let's assume N_steps == n_exercise for now. If not, we resize?
        # Let's try to be robust: if N_steps != n_exercise, we just use the available steps as the exercise dates.
        # But the contract fixes 12 dates.
        # Let's assume the input is aligned.
        pass
    
    # Define exercise times
    dt = T / n_exercise
    exercise_times = np.arange(1, n_exercise + 1) * dt
    
    # If N_steps is different, we might have a problem.
    # Let's assume N_steps is the number of exercise dates provided.
    # If N_steps > 12, we might need to select the correct ones.
    # But without time stamps in the input, we assume the columns are the exercise dates in order.
    # If N_steps != 12, we will just use the columns as they are, but the policy object will be built for N_steps.
    # However, the contract says 12.
    # Let's force n_exercise to be N_steps if they differ? No, the contract is fixed.
    # Let's assume N_steps == 12.
    
    # Initialize coefficients list
    coeffs = [None] * N_steps
    
    # Last step: Value is the payoff (no continuation)
    # Actually, at the last step, we always exercise if ITM.
    # Value = Payoff.
    
    # We need to store the "values" for regression at the next step.
    # V_{i+1} is the value at step i+1.
    # We start from the end.
    
    # V_N (last step) = Payoff_N
    # For i = N-1 down to 0:
    #   Discount V_{i+1}
    #   Regress Discounted V_{i+1} on Basis(S_i)
    #   Store coeffs[i]
    #   V_i = max(Payoff_i, Continuation_i)
    
    # Let's use the last column as the terminal payoff.
    S_last = training_paths[:, -1]
    V_next = np.maximum(K - S_last, 0)
    
    # We need to go backwards.
    # The regression for step i uses S_i and V_{i+1}.
    # So we iterate i from N_steps-2 down to 0.
    
    # Basis function degree
    degree = 5
    # Ridge lambda
    lam = 1e-6
    
    for i in range(N_steps - 2, -1, -1):
        if time.time() > deadline:
            break
            
        S_i = training_paths[:, i]
        
        # Discount V_next
        # dt is the time step
        disc_factor = np.exp(-r * dt)
        V_disc = disc_factor * V_next
        
        # Build basis matrix
        x = S_i / K
        # Use monomials 1, x, x^2, x^3, x^4, x^5
        B = np.column_stack([
            np.ones_like(x),
            x,
            x**2,
            x**3,
            x**4,
            x**5
        ])
        
        # Ridge Regression: (B^T B + lam I) c = B^T V_disc
        # Solve for c
        try:
            A = B.T @ B
            A += lam * np.eye(A.shape[0])
            b = B.T @ V_disc
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fallback to lstsq if singular
            c, _, _, _ = np.linalg.lstsq(B, V_disc, rcond=None)
            
        coeffs[i] = c
        
        # Compute continuation value for all paths at step i
        cont_val = B @ c
        
        # Compute payoff at step i
        payoff_i = np.maximum(K - S_i, 0)
        
        # Value at step i
        V_i = np.maximum(payoff_i, cont_val)
        
        # Update V_next for the next iteration (backwards)
        V_next = V_i
        
    # If we broke out early, coeffs might have Nones at the beginning.
    # We should handle this by using a default policy (e.g., always exercise if ITM) for those steps.
    # Or just return the policy with what we have.
    # For safety, let's fill any None coeffs with a simple heuristic or just leave them and handle in exercise.
    
    # Create policy object
    # We need to ensure coeffs are available for all steps used in exercise.
    # If N_steps in training is 12, and we filled 0..10, then step 11 (last) has no regression, 
    # but we don't need regression for the last step because we always exercise if ITM?
    # Wait, the LS algorithm defines V_i for i=0..N-1.
    # The exercise decision at i is based on V_{i+1} regression.
    # So coeffs[i] is needed for decision at i.
    # For the last step i=N-1, there is no i+1. We always exercise if ITM.
    # So coeffs[N-1] is not used in the same way?
    # In the loop, we calculated coeffs for i = N-2 down to 0.
    # coeffs[N-1] remains None.
    # In the exercise method, if idx == N-1, we should just return payoff > 0.
    
    policy = ExercisePolicy(coeffs, K, exercise_times[:N_steps]) # Use actual number of steps
    
    # Override exercise method to handle the last step and missing coeffs
    def final_exercise(current_state, time):
        S = np.asarray(current_state)
        # Map time to index
        if len(policy.exercise_times) > 0:
            idx = np.argmin(np.abs(policy.exercise_times - time))
        else:
            idx = 0
            
        # If idx is the last step, always exercise if ITM
        if idx == N_steps - 1:
            return np.maximum(policy.K - S, 0) > 0
            
        # If coeffs[idx] is None (due to timeout), fallback to ITM
        if policy.coeffs[idx] is None:
            return np.maximum(policy.K - S, 0) > 0
            
        B = policy._basis(S)
        cont_val = B @ policy.coeffs[idx]
        payoff = np.maximum(policy.K - S, 0)
        return payoff > cont_val
        
    # Wrap in an object that matches the interface
    class FinalPolicy:
        def __init__(self, ex_func, K, times):
            self.ex_func = ex_func
            self.K = K
            self.times = times
            
        def exercise(self, current_state, time):
            return self.ex_func(current_state, time)
            
    return FinalPolicy(final_exercise, K, exercise_times[:N_steps])

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fits the exercise policy for a Bermudan put option using direct Monte Carlo
    stochastic gradient ascent on boundary parameters.
    
    This approach treats the exercise boundaries as parameters of a threshold policy
    and directly optimizes the expected discounted payoff on the training paths.
    This avoids the approximation errors of regression-based methods (like LSM) and
    directly maximizes the objective function of interest.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin within 120s limit
    
    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12
    
    if training_paths is None or len(training_paths) == 0:
        class FallbackPolicy:
            def exercise(self, current_state, time):
                return current_state < K
        return FallbackPolicy()

    paths = np.asarray(training_paths, dtype=float)
    n_paths, n_cols = paths.shape
    
    # Determine index alignment
    if n_cols >= 13:
        start_idx = 1
        end_idx = 12
    else:
        start_idx = 0
        end_idx = 11
        
    S_ex = paths[:, start_idx:end_idx+1]  # Shape (n_paths, 12)
    
    dt = T / n_exercises
    disc = np.exp(-r * dt)
    discount_factors = np.array([disc ** (n_exercises - 1 - j) for j in range(n_exercises)])
    
    # Initialize boundaries
    # Start with a reasonable guess: linear interpolation from 0.5*K to K
    # Or just start at K and let optimization move them.
    # For a put, boundaries should be < K.
    # Let's initialize with a standard heuristic: B_j = K * (1 - (1 - j/n_exercises)^2) or similar?
    # Actually, for American put, boundary is below K.
    # Let's start with B_j = K * (j / n_exercises) + 0.1 * K ?
    # Simpler: start with B_j = K * 0.8 for all, or a ramp.
    # Ramp: B_j = K * (0.5 + 0.5 * j / (n_exercises - 1))
    boundaries = K * (0.5 + 0.5 * np.arange(n_exercises) / (n_exercises - 1))
    # Ensure last boundary is close to K
    boundaries[-1] = K
    
    # Optimization: Stochastic Gradient Ascent
    # We want to maximize E[sum_j discount_j * I(exercise_j) * payoff_j]
    # Let theta be the vector of boundaries.
    # The policy is: exercise at j if S_j < B_j(theta).
    # This is a step function, so gradients are zero almost everywhere.
    # We can use a smooth approximation or finite differences.
    # Given the discrete nature, let's use a simple coordinate ascent or random search.
    # Coordinate ascent: update one boundary at a time by trying small shifts.
    
    # Precompute intrinsic payoffs
    intrinsic = np.maximum(K - S_ex, 0.0)  # Shape (n_paths, n_exercises)
    
    # Helper to compute objective
    def compute_objective(bounds):
        total_value = 0.0
        active = np.ones(n_paths, dtype=bool)
        for j in range(n_exercises):
            if not np.any(active):
                break
            # Exercise decision: S_j < bounds[j] AND active
            exercise_mask = active & (S_ex[:, j] < bounds[j])
            if np.any(exercise_mask):
                # Add discounted payoff
                total_value += discount_factors[j] * np.sum(intrinsic[exercise_mask, j])
                # Update active: remove exercised paths
                active[exercise_mask] = False
        return total_value / n_paths
    
    # Initial objective
    current_obj = compute_objective(boundaries)
    
    # Simple Random Search / Coordinate Ascent
    # Since the landscape is non-smooth, we'll use a simple hill-climbing approach
    # by perturbing each boundary and keeping the change if it improves the objective.
    
    step_size = 0.5  # Initial step size in price units
    n_iter = 5000
    patience = 100
    
    no_improve_count = 0
    best_obj = current_obj
    best_bounds = boundaries.copy()
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
            
        # Pick a random boundary to perturb
        j = np.random.randint(n_exercises)
        
        # Try a few random perturbations
        best_local_obj = current_obj
        best_local_delta = 0.0
        
        for _ in range(5):
            delta = np.random.uniform(-step_size, step_size)
            new_bounds = boundaries.copy()
            new_bounds[j] += delta
            
            # Enforce monotonicity: B_j <= B_{j+1} <= ... <= B_{n-1} = K
            # And B_j >= 0
            # We'll enforce this by projecting the new bounds
            for k in range(1, n_exercises):
                if new_bounds[k] < new_bounds[k-1]:
                    new_bounds[k] = new_bounds[k-1]
            new_bounds = np.maximum(new_bounds, 0.0)
            new_bounds[-1] = K
            
            if np.array_equal(new_bounds, boundaries):
                continue
                
            new_obj = compute_objective(new_bounds)
            
            if new_obj > best_local_obj:
                best_local_obj = new_obj
                best_local_delta = delta
                # Update boundaries immediately if improvement found
                boundaries = new_bounds
                current_obj = new_obj
                
        if current_obj > best_obj:
            best_obj = current_obj
            best_bounds = boundaries.copy()
            no_improve_count = 0
            # Maybe increase step size slightly if we're improving fast?
            # Or keep it constant.
        else:
            no_improve_count += 1
            if no_improve_count > patience:
                # Reduce step size
                step_size *= 0.9
                no_improve_count = 0
                if step_size < 0.01:
                    break
    
    # Final monotonicity check
    for j in range(1, n_exercises):
        if best_bounds[j] < best_bounds[j-1]:
            best_bounds[j] = best_bounds[j-1]
    best_bounds[-1] = K
    best_bounds = np.maximum(best_bounds, 0.0)
    
    class ExercisePolicy:
        def __init__(self, boundaries, n_exercises):
            self.boundaries = boundaries
            self.n_exercises = n_exercises
            
        def exercise(self, current_state, time):
            """
            current_state: array of stock prices.
            time: step index 0..11.
            """
            if time < 0 or time >= self.n_exercises:
                return np.zeros(len(current_state), dtype=bool)
            
            B = self.boundaries[time]
            return current_state < B
    
    return ExercisePolicy(best_bounds, n_exercises)
# EVOLVE-BLOCK-END

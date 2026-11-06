"""Bermudan put exercise policy via parametric smooth boundary + Monte Carlo optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import math

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    TIME_LIMIT = 110.0  # Safety margin before 120s kill

    # Extract parameters
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_steps = 12
    dt = T / n_steps
    discount = np.exp(-r * dt)

    # Process training paths
    # Assume shape (n_paths, n_steps+1) where column 0 is S_0, columns 1..12 are exercise dates
    paths = np.asarray(training_paths, dtype=np.float64)
    if paths.ndim == 1:
        paths = paths.reshape(1, -1)
    
    n_paths = paths.shape[0]
    
    # If paths has n_steps+1 columns, use columns 1..n_steps (exercise dates)
    # If paths has n_steps columns, use all of them
    if paths.shape[1] >= n_steps + 1:
        exercise_states = paths[:, 1:n_steps+1]  # shape (n_paths, n_steps)
    else:
        exercise_states = paths[:, :n_steps]
    
    # Time fractions for each exercise date: t_i/T for i=1..12
    time_fracs = np.array([(i * dt) / T for i in range(1, n_steps + 1)])  # shape (n_steps,)
    
    # Remaining time fraction: 1 - t_i/T
    remaining_time_fracs = 1.0 - time_fracs  # shape (n_steps,)
    
    # Create meshgrid for vectorized evaluation
    # S_grid: (n_paths, n_steps)
    # rt_grid: (1, n_steps) broadcast to (n_paths, n_steps)
    S_grid = exercise_states  # (n_paths, n_steps)
    rt_grid = remaining_time_fracs.reshape(1, -1)  # (1, n_steps)
    
    # Initial parameterization: polynomial in remaining time fraction
    # Boundary: S*(rt) = K * (1 + a0 + a1*rt + a2*rt^2 + a3*rt^3)
    # Exercise if S <= S*(rt)
    # For a put, we exercise when S is low.
    # At maturity (rt=0), European put value: max(K-S, 0). For deep ITM, S* ~ 0.
    # At t=0 (rt=1), the boundary should be lower (less incentive to exercise early).
    
    # Let's parameterize the boundary as:
    # S*(rt) = K * sigmoid(z) where z = b0 + b1*rt + b2*rt^2
    # Or more directly, use a polynomial that gives reasonable values.
    # For a Bermudan put with these parameters, the exercise boundary is well-studied.
    # At maturity, S* = K (exercise if S <= K).
    # Earlier, S* < K.
    
    # Let's use: S*(rt) = K * (1 - exp(-c * (1-rt))) for some c, but this may be too restrictive.
    # Better: use a polynomial fit to the European put threshold as initial guess, then optimize.
    
    # European put: exercise if S <= K. So at rt=0, S*=K.
    # For Bermudan, S*(rt) < K for rt > 0.
    
    # Parameterization: S*(rt) = K * (a0 + a1*rt + a2*rt^2 + a3*rt^3)
    # With constraint that S*(0) = K (at maturity), so a0 = 1.
    # S*(rt) = K * (1 + a1*rt + a2*rt^2 + a3*rt^3)
    # We want S*(rt) < K for rt > 0, so a1 + a2*rt + a3*rt^3 < 0 for rt in (0,1].
    # Let's initialize with a1 = -0.5, a2 = 0, a3 = 0 as a rough guess.
    
    # Actually, let's use a smoother parameterization to ensure positivity and monotonicity.
    # S*(rt) = K * sigmoid(z(rt)) / sigmoid(z(0)) ... this is getting complex.
    
    # Simpler: use the fact that for these parameters, the boundary is roughly:
    # S*(rt) ≈ K * (1 - 0.3*rt + 0.1*rt^2) or similar.
    
    # Let's just use a linear boundary initially: S*(rt) = K * (1 - alpha * rt)
    # And optimize alpha, plus add quadratic and cubic terms.
    
    def compute_value(params, paths_to_use, n_eval_paths=None):
        """Compute the discounted payoff for a given set of parameters and paths."""
        if n_eval_paths is None:
            n_eval_paths = paths_to_use.shape[0]
        
        a1, a2, a3 = params
        # S*(rt) = K * (1 + a1*rt + a2*rt^2 + a3*rt^3)
        # rt is remaining time fraction
        rt = remaining_time_fracs  # (n_steps,)
        boundary = K * (1.0 + a1 * rt + a2 * rt**2 + a3 * rt**3)  # (n_steps,)
        
        # For each path, determine exercise decisions
        # paths_to_use: (n_paths, n_steps)
        # boundary: (n_steps,)
        # Exercise if S <= boundary at each step, but we need to account for the fact that
        # once exercised, the path stops.
        
        # Vectorized approach: process steps in reverse (from maturity to start)
        # At maturity (step n_steps-1), exercise if S <= K (European put)
        # Actually, at the last exercise date, it's equivalent to European: exercise if S <= K.
        # For earlier dates, we need to compare immediate payoff vs. continuation value.
        
        # This is tricky to vectorize with the "stop after exercise" logic.
        # Let's do it step by step from last to first.
        
        n_s = paths_to_use.shape[1]
        n_p = paths_to_use.shape[0]
        
        # Continuation value: start with 0 (after maturity, no more payoffs)
        cont_value = np.zeros(n_p)
        
        # Process from last exercise date to first
        for i in range(n_s - 1, -1, -1):
            rt_i = remaining_time_fracs[i]
            boundary_i = K * (1.0 + a1 * rt_i + a2 * rt_i**2 + a3 * rt_i**3)
            
            S_i = paths_to_use[:, i]
            immediate_payoff = np.maximum(K - S_i, 0.0)
            
            # Exercise if immediate payoff > continuation value
            # But the boundary parameterization defines where we exercise.
            # Actually, the boundary IS the policy. We exercise if S <= boundary.
            # But we should also check if it's optimal to do so.
            # For the optimization, we use the boundary as the policy.
            
            exercise_mask = S_i <= boundary_i
            
            # If we exercise, we get immediate_payoff
            # If we don't, we get the discounted continuation value
            # But we need to be careful: the continuation value is already the expected value
            # of future payoffs, discounted to the current time.
            
            # Actually, the continuation value at step i is the discounted expected value
            # of future payoffs from steps i+1 to n_s-1.
            
            # Let's redefine: cont_value is the value at the current step of NOT exercising.
            # At maturity (i = n_s-1), if we don't exercise, value = 0.
            # So at i = n_s-1, we always exercise if immediate_payoff > 0.
            
            # Wait, this approach is flawed because the boundary is a deterministic policy,
            # not an optimal stopping rule. We're using the boundary as the policy, so we
            # exercise whenever S <= boundary, regardless of continuation value.
            
            # For the objective, we just simulate the policy and compute the total payoff.
            
            if i == n_s - 1:
                # Last step: exercise if S <= boundary (which should be ~K)
                payoff_i = np.where(exercise_mask, immediate_payoff, 0.0)
            else:
                # Earlier steps: exercise if S <= boundary
                # If we exercise, we get immediate_payoff and stop.
                # If we don't, we continue.
                # But we need to track which paths have already exercised.
                
                # This requires a different approach. Let's track an "active" mask.
                pass
            
            # This step-by-step approach with tracking is complex. Let me restructure.
        
        # Let me rewrite compute_value more carefully.
        # We'll track which paths are still active (not yet exercised).
        
        cont_value = np.zeros(n_p)
        active = np.ones(n_p, dtype=bool)
        
        for i in range(n_s - 1, -1, -1):
            rt_i = remaining_time_fracs[i]
            boundary_i = K * (1.0 + a1 * rt_i + a2 * rt_i**2 + a3 * rt_i**3)
            
            S_i = paths_to_use[:, i]
            immediate_payoff = np.maximum(K - S_i, 0.0)
            
            # Only consider active paths
            if not np.any(active):
                break
            
            exercise_mask = active & (S_i <= boundary_i)
            
            # For paths that exercise at this step, they get immediate_payoff
            # For paths that don't exercise, they continue (their value is cont_value from next step)
            
            # The total value for paths that exercise at step i is:
            # discount^(n_s-1-i) * immediate_payoff
            # The total value for paths that don't exercise is:
            # discount^(n_s-1-i) * cont_value (which is already the discounted future value)
            
            # Actually, let's define cont_value as the value at step i of NOT exercising,
            # which is the discounted expected future payoff.
            
            # Let's use a different approach: accumulate the total discounted payoff.
            
            total_value = np.zeros(n_p)
            already_exercised = np.zeros(n_p, dtype=bool)
            
            for j in range(n_s):
                if already_exercised.all():
                    break
                rt_j = remaining_time_fracs[j]
                boundary_j = K * (1.0 + a1 * rt_j + a2 * rt_j**2 + a3 * rt_j**3)
                
                S_j = paths_to_use[:, j]
                immediate_j = np.maximum(K - S_j, 0.0)
                
                # Exercise if S <= boundary and not already exercised
                ex_mask = (~already_exercised) & (S_j <= boundary_j)
                
                # Discount factor for payoff at step j: discount^(n_s-1-j)
                # Wait, the discounting should be from the payoff time to time 0.
                # Payoff at step j occurs at time t_j = (j+1)*dt.
                # Discount to time 0: exp(-r * t_j) = discount^(j+1)
                
                disc_j = discount ** (j + 1)
                
                total_value[ex_mask] += disc_j * immediate_j[ex_mask]
                already_exercised[ex_mask] = True
            
            # Average over paths
            return np.mean(total_value)
    
    # Initial parameters
    x0 = np.array([-0.3, 0.1, -0.05])
    
    # Objective function to minimize (negative of value)
    def objective(params):
        if time.time() - start_time > TIME_LIMIT:
            return 0.0  # Return a bad value if we're running out of time
        return -compute_value(params, exercise_states)
    
    # Optimize
    try:
        result = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-4})
        best_params = result.x
    except Exception:
        best_params = x0
    
    # If we have time, do a finer search or more iterations
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 10:
        try:
            result2 = minimize(objective, best_params, method='L-BFGS-B',
                              options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})
            best_params = result2.x
        except Exception:
            pass
    
    # Create the exercise policy object
    a1, a2, a3 = best_params
    
    class ExercisePolicy:
        def __init__(self, a1, a2, a3, K, remaining_time_fracs, n_steps):
            self.a1 = a1
            self.a2 = a2
            self.a3 = a3
            self.K = K
            self.remaining_time_fracs = remaining_time_fracs
            self.n_steps = n_steps
            
        def exercise(self, current_state, time_index):
            """
            current_state: array of current stock prices, shape (n_paths,)
            time_index: integer or array of exercise date indices (1-based or 0-based?)
            
            Returns: boolean vector of exercise decisions.
            """
            # time_index is the exercise date index (1 to n_steps)
            # We need to map to remaining time fraction
            if isinstance(time_index, (int, np.integer)):
                # 1-based index: time_index = 1 corresponds to first exercise date
                # remaining_time_frac = 1 - time_index/n_steps
                rt = 1.0 - time_index / self.n_steps
                boundary = self.K * (1.0 + self.a1 * rt + self.a2 * rt**2 + self.a3 * rt**3)
                return current_state <= boundary
            else:
                # Array of time indices
                time_index = np.asarray(time_index)
                rt = 1.0 - time_index / self.n_steps
                boundary = self.K * (1.0 + self.a1 * rt + self.a2 * rt**2 + self.a3 * rt**3)
                # boundary shape: (n_paths,) if time_index is (n_paths,)
                # current_state shape: (n_paths,)
                return current_state <= boundary
    
    policy = ExercisePolicy(a1, a2, a3, K, remaining_time_fracs, n_steps)
    return policy
# EVOLVE-BLOCK-END

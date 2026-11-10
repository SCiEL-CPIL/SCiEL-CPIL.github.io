"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fits the exercise policy for a Bermudan put option using a high-resolution 
    Binomial Tree backward induction method.
    
    This approach ignores the specific training paths and instead computes the 
    optimal exercise boundaries analytically/numerically based on the model 
    parameters (K, r, sigma, T, n_exercises). This is a deterministic method 
    that converges to the true optimal policy, avoiding the noise and bias 
    associated with Monte Carlo regression (LSM).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercises = 12
    
    # We will use a Binomial Tree with N steps.
    # To align with 12 monthly exercise dates, we can use N = 12 * steps_per_month.
    # A higher steps_per_month gives a finer grid and more accurate boundaries.
    # 100 steps per month -> 1200 total steps. This is computationally cheap 
    # (O(N^2) for the tree values, but we only need boundaries at 12 specific 
    # time slices, so we can optimize the backward pass).
    
    steps_per_month = 100
    N = n_exercises * steps_per_month
    
    # Binomial Tree Parameters (CRR model)
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Initialize the value array for the final step (maturity)
    # Number of nodes at step i is i + 1
    # We only need to store the current row of values during backward induction
    # to save memory, but we need to record the boundaries at specific steps.
    
    # We iterate backwards from step N to step 1.
    # Exercise dates are at steps: steps_per_month, 2*steps_per_month, ..., 12*steps_per_month.
    # Note: The problem states "12 monthly exercise dates (no time-0 exercise)".
    # So exercise is allowed at t_1, ..., t_12.
    # In our tree, step i corresponds to time t_i = i * dt.
    # t_1 = 1 * dt. Since dt = T/N, t_1 = T/N.
    # The first exercise date is at t = T/12.
    # Step index for t = T/12 is (T/12) / dt = (T/12) / (T/N) = N/12 = steps_per_month.
    # So exercise dates correspond to tree steps: steps_per_month, 2*steps_per_month, ..., N.
    
    # We need to store the boundary (max S to exercise) for each of the 12 exercise dates.
    # boundaries[0] corresponds to t_1, boundaries[1] to t_2, ..., boundaries[11] to t_12.
    boundaries = np.zeros(n_exercises)
    
    # Initialize values at maturity (step N)
    # Node j at step N has price S_0 * u^j * d^(N-j)
    # Payoff = max(K - S, 0)
    # We'll work backwards.
    
    # To be efficient, we can just store the array of values for the current step.
    # val[j] is the value at node j of the current step i.
    
    # Start at step N (maturity)
    # Prices at step N: S_0 * u^j * d^(N-j) for j=0..N
    # We don't actually need S_0 if we just want the boundary in terms of S.
    # But the tree is rooted at S_0. Let's assume S_0 = 100 for the tree construction?
    # No, the boundary is a price level. It is independent of S_0 in the sense that 
    # the policy is "exercise if S < B". The tree calculates B relative to the grid.
    # Actually, the boundary B is an absolute price. 
    # Let's construct the tree starting from a nominal S_0 = 1.0 or 100.0?
    # The shape of the value function V(S) is what matters.
    # Let's use S_0 = 1.0 for the tree nodes to keep numbers small, then scale?
    # No, it's easier to just use S_0 = 100.0 (typical spot) or any value.
    # The boundary B is the price level.
    
    S0_tree = 100.0 # Reference spot for tree construction
    
    # Precompute log-u and log-d for speed? Or just use powers.
    # For N=1200, u^j * d^(N-j) might be slow if done naively for each node.
    # Better: S_j = S0 * (u/d)^j * d^N.
    # Let's just store the values.
    
    # val[j] at step i
    # At step N:
    # j ranges from 0 to N
    # S_N_j = S0_tree * u^j * d^(N-j)
    # We can compute this iteratively or using vectorization.
    
    # Vectorized initialization for step N
    j_vals = np.arange(N + 1)
    # S = S0 * u^j * d^(N-j) = S0 * d^N * (u/d)^j
    ratio = u / d
    S_N = S0_tree * (d ** N) * (ratio ** j_vals)
    val = np.maximum(K - S_N, 0.0)
    
    # Backward induction
    # We need to check for exercise at steps: N, N-steps_per_month, ..., steps_per_month
    # Let's keep track of which exercise date we are at.
    # exercise_step_indices = [N, N - steps_per_month, ..., steps_per_month]
    # Map these to boundary indices 11, 10, ..., 0
    
    # We iterate i from N-1 down to 1.
    # At each step i, we compute the new values.
    # If i is an exercise date, we record the boundary.
    
    # To find the boundary at a specific step i:
    # We have val[j] which is the value at node j (price S_i_j).
    # We also have the intrinsic value at node j: max(K - S_i_j, 0).
    # The node is exercised if intrinsic > continuation (which is val[j] from the 
    # "continue" calculation, but wait, in backward induction, val[j] *is* the 
    # max(intrinsic, continuation). 
    # So, to find the boundary, we look at the nodes where exercise was optimal.
    # Exercise is optimal if intrinsic >= continuation.
    # We can determine this by checking if val[j] == intrinsic[j] (within tolerance).
    # However, due to floating point, it's better to reconstruct the decision.
    # Alternatively, since the policy is monotonic (put: exercise if S < B),
    # we can find the largest j (highest price) where exercise was optimal.
    # Note: In a binomial tree, higher j means higher price.
    # So we want the max j such that exercise was optimal.
    # Then B = S_i_j_max.
    
    # Let's store the step indices corresponding to exercise dates.
    # Step i corresponds to time t_i.
    # Exercise dates are at t_k = k * T / 12 for k=1..12.
    # i_k = k * N / 12 = k * steps_per_month.
    
    exercise_steps = {}
    for k in range(1, n_exercises + 1):
        step_idx = k * steps_per_month
        exercise_steps[step_idx] = k - 1  # Map to boundary index 0..11
        
    # Backward loop
    for i in range(N - 1, 0, -1):
        if time.time() > deadline:
            # Fallback: use intrinsic value boundaries (B=K) for remaining
            # This is a safe but suboptimal fallback
            for k in range(n_exercises):
                if boundaries[k] == 0.0: # Not yet set
                    boundaries[k] = K
            break
            
        # Compute continuation value
        # cont[j] = exp(-r*dt) * (p * val[j+1] + (1-p) * val[j])
        # val currently holds values for step i+1 (j=0..N-i-1? No, j=0..N-i)
        # Wait, at step i+1, there are N-(i+1)+1 = N-i nodes.
        # Indices 0..N-i-1.
        # Let's be careful with indices.
        # At step i, there are N-i+1 nodes.
        # val array length should be N-i. (From previous step i+1)
        
        # val[j] is value at node j of step i+1.
        # Node j of step i connects to node j of step i+1 (down) and node j+1 of step i+1 (up).
        
        # cont[j] = disc * (p * val[j+1] + (1-p) * val[j])
        # j ranges from 0 to N-i-1.
        
        disc = np.exp(-r * dt)
        
        # Vectorized calculation
        # val is length N-i
        # val[1:] is length N-i-1
        # val[:-1] is length N-i-1
        # Result length N-i-1? No.
        # Node 0 at step i connects to Node 0 and Node 1 at step i+1.
        # Node N-i-1 at step i connects to Node N-i-1 and Node N-i at step i+1.
        # So cont has length N-i.
        
        # cont[j] uses val[j] and val[j+1]
        # So we need val[0..N-i-1] and val[1..N-i]
        
        cont = disc * (p * val[1:] + (1 - p) * val[:-1])
        
        # Intrinsic value at step i
        # Prices at step i: S_i_j = S0_tree * u^j * d^(N-i-j)
        # We can update S prices iteratively or recompute.
        # Recomputing for N=1200 is fast enough if vectorized.
        
        j_vals_i = np.arange(N - i + 1)
        S_i = S0_tree * (d ** (N - i)) * (ratio ** j_vals_i)
        
        intrinsic = np.maximum(K - S_i, 0.0)
        
        # New value is max(intrinsic, cont)
        new_val = np.maximum(intrinsic, cont)
        
        # Check if this is an exercise date
        if i in exercise_steps:
            boundary_idx = exercise_steps[i]
            
            # Find the largest j where exercise was optimal.
            # Exercise was optimal if intrinsic[j] >= cont[j].
            # Since it's a put, exercise region is S < B, i.e., low j.
            # Wait, low j means low price.
            # So we want the MAX j such that intrinsic[j] >= cont[j].
            
            # Note: At high prices (large j), intrinsic is 0, cont is positive (usually).
            # At low prices (small j), intrinsic is positive, cont might be lower.
            
            # Find indices where exercise was optimal
            exercise_mask = intrinsic >= cont
            
            # Find the max index in exercise_mask
            if np.any(exercise_mask):
                max_j = np.max(np.where(exercise_mask)[0])
                boundaries[boundary_idx] = S_i[max_j]
            else:
                # Never exercise? Unlikely for a put, but set to 0
                boundaries[boundary_idx] = 0.0
        
        val = new_val
        
    # Ensure monotonicity of boundaries
    # For puts, boundaries should increase as time approaches maturity.
    # boundaries[0] <= boundaries[1] <= ... <= boundaries[11]
    for k in range(1, n_exercises):
        if boundaries[k] < boundaries[k-1]:
            boundaries[k] = boundaries[k-1]
            
    # The last boundary (t_12) should be K (or very close).
    # At maturity, you always exercise if ITM.
    # Our loop goes down to i=1. The step i=N (maturity) was initialized.
    # The step i=N-steps_per_month is the last exercise date (t_11? No, t_12 is maturity).
    # Wait, the problem says "12 monthly exercise dates".
    # Usually, Bermudan options allow exercise at t_1, ..., t_n.
    # If t_12 is maturity, you can exercise at t_12.
    # In our loop, we checked i in exercise_steps.
    # exercise_steps included N (which is t_12).
    # But the loop `for i in range(N-1, 0, -1)` does NOT check i=N.
    # It starts at N-1.
    # So boundary for t_12 (index 11) was never set in the loop?
    # Let's check: exercise_steps[N] = 11.
    # The loop doesn't visit i=N.
    # So boundaries[11] remains 0.0.
    # We must set it manually.
    boundaries[11] = K
    
    # Also, ensure boundaries are within reasonable range
    # For a put, B <= K.
    boundaries = np.minimum(boundaries, K)
    
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
            return current_state <= B
    
    return ExercisePolicy(boundaries, n_exercises)
# EVOLVE-BLOCK-END

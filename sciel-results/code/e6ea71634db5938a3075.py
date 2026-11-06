"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using backward induction on a binomial tree.
    
    Args:
        training_paths: array of shape (n_paths, n_times+1) with simulated stock price paths
        contract: dict with contract parameters
        
    Returns:
        An object with exercise(current_state, time) method returning boolean vector.
    """
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s limit
    
    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    
    # Exercise times: 12 monthly dates, no time-0 exercise
    # T=1, so monthly dates at 1/12, 2/12, ..., 12/12
    exercise_times = np.linspace(1.0/n_exercise, T, n_exercise)
    
    # Build binomial tree with sufficient resolution
    # Use a fine grid for accurate threshold computation
    # We'll use backward induction on a log-space binomial tree
    
    # Number of steps for the tree - use a reasonably fine grid
    # For 12 exercise dates, we can use a tree with steps aligned to exercise dates
    # Use a sub-step approach: multiple tree steps between exercise dates
    
    # Let's use a total number of time steps that gives good resolution
    # With T=1 and sigma=0.2, we need enough steps for accuracy
    # Use 1200 steps as mentioned in the reference (100 steps per month)
    n_steps_per_period = 100
    total_steps = n_exercise * n_steps_per_period
    
    dt = T / total_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    
    # We'll do backward induction on the tree
    # State at each node is the stock price
    # For a put: payoff at expiry if not exercised = max(K - S, 0)
    # At exercise dates, we can choose to exercise
    
    # Store the continuation value at each exercise date
    # and the critical threshold (exercise if S <= threshold)
    
    # For efficiency, we'll track the value function at exercise dates
    # using a 1D array representing stock prices at each node
    
    # Initialize at the last exercise date (which is T)
    # At T, if not exercised, the option expires with payoff max(K-S, 0)
    # But at the last exercise date, we can exercise, so value = max(K-S, 0)
    
    # We need to track the range of stock prices
    # S0 from the training data
    S0 = training_paths[0, 0]
    
    # Build the price grid at each exercise date
    # At time t_j, the stock price can be S0 * u^k * d^(j*n_steps_per_period - k)
    # for k = 0, 1, ..., j*n_steps_per_period
    
    # For memory efficiency, we'll only store values at exercise dates
    # and do the backward induction between them
    
    # Let's precompute the up/down factors for one period (between exercise dates)
    dt_ex = T / n_exercise
    u_ex = np.exp(sigma * np.sqrt(dt_ex))
    d_ex = 1.0 / u_ex
    p_ex = (np.exp(r * dt_ex) - d_ex) / (u_ex - d_ex)
    disc_ex = np.exp(-r * dt_ex)
    
    # For the backward induction, we'll use a finer tree internally
    # but only store results at exercise dates
    
    # Actually, let's use a direct approach with a fine binomial tree
    # and track the value at each node, but only keep the exercise date values
    
    # Memory-efficient approach: store value as a function of log-price
    # Use interpolation
    
    # Let's use a simpler approach: compute the critical thresholds directly
    # using backward induction on a binomial tree with the full resolution
    
    # We'll represent the state by the log-price: x = ln(S)
    # At each time step, x evolves as x -> x + ln(u) or x -> x + ln(d)
    
    # For the put, the payoff is max(K - S, 0) = max(K - exp(x), 0)
    
    # Let's do the backward induction with arrays
    # At each step, we have an array of values indexed by the number of up moves
    
    # Start from the end
    # At time T (last exercise date), the value is max(K - S, 0)
    
    # We'll work backwards from the last exercise date to the first
    
    # For each exercise date j (from last to first), we compute:
    # 1. The continuation value (discounted expected value from next period)
    # 2. The exercise value = max(K - S, 0)
    # 3. The option value = max(continuation, exercise)
    # 4. The critical threshold where exercise = continuation
    
    # Let's use a vectorized approach
    # At exercise date j, the possible log-prices are:
    # x_k = ln(S0) + k*ln(u_ex) - (j)*ln(u_ex)  for k=0,...,j
    # Wait, that's not quite right. Let me think more carefully.
    
    # In a binomial tree, after j periods, the stock price is:
    # S_j(k) = S0 * u_ex^k * d_ex^(j-k) = S0 * u_ex^(2k-j) * (u_ex*d_ex)^((j-k)/2)
    # Since u_ex * d_ex = 1, S_j(k) = S0 * u_ex^(2k-j)
    
    # So log-price: x_j(k) = ln(S0) + (2k-j) * ln(u_ex)
    
    # The number of nodes at step j is j+1
    
    # For the backward induction, we need to handle the transition between
    # exercise dates. Between exercise dates, there are n_steps_per_period
    # smaller steps.
    
    # Let me use a different approach: use the full fine-grained tree
    # but only store values at exercise dates using aggregation
    
    # Actually, for a put with these parameters, the optimal exercise boundary
    # is monotone. Let's compute it using a standard binomial tree approach.
    
    # Use a tree with N total steps
    N = 1200  # 100 steps per month for 12 months
    dt_fine = T / N
    
    u_fine = np.exp(sigma * np.sqrt(dt_fine))
    d_fine = 1.0 / u_fine
    p_fine = (np.exp(r * dt_fine) - d_fine) / (u_fine - d_fine)
    disc_fine = np.exp(-r * dt_fine)
    
    # Steps at which we can exercise (0-indexed from the end)
    # Exercise dates are at times 1/12, 2/12, ..., 12/12
    # In terms of fine steps: step i corresponds to time i*dt_fine
    # Exercise at time t_j = j/12 corresponds to step i_j = j * N / 12 = j * 100
    
    exercise_steps = [j * N // n_exercise for j in range(1, n_exercise + 1)]
    
    # Initialize the value array at the final time
    # At time T, the value is the payoff: max(K - S, 0)
    # S at step N with k up moves: S = S0 * u_fine^k * d_fine^(N-k) = S0 * u_fine^(2k-N)
    
    # We'll store the value as a function of k (number of up moves)
    # At step N, k ranges from 0 to N
    
    # For memory, we only need to store one row at a time
    # Let's work backwards from step N to step 0
    
    # Value at step N:
    # k = 0, 1, ..., N
    # S_k = S0 * u_fine^(2k - N)
    # V_k = max(K - S_k, 0)
    
    # We'll use arrays of size N+1 initially, shrinking as we go back
    
    # Actually, at step i, k ranges from 0 to i
    # So at step N, we have N+1 values
    
    # Let's allocate and compute
    
    # For the fine tree, N=1200, so we have up to 1201 nodes
    # This is manageable
    
    # Initialize at step N
    k_vals = np.arange(N + 1)
    S_N = S0 * u_fine ** (2 * k_vals - N)
    V = np.maximum(K - S_N, 0.0)
    
    # Store the value at each exercise date for threshold computation
    # We'll track the critical threshold at each exercise date
    
    # Work backwards from step N-1 to step 0
    # At each step, check if it's an exercise date
    
    # Map from step index to whether it's an exercise date
    # exercise_steps contains the step indices where we can exercise
    # (these are the step indices at the END of each period, i.e., the time points)
    
    # Wait, I need to be careful. In backward induction:
    # - At step i, we're at time i*dt_fine
    # - If step i is an exercise date, we can exercise
    # - The value at step i is max(exercise_value, continuation_value)
    # - continuation_value = disc * (p * V_up + (1-p) * V_down)
    
    # Let me redo this properly
    
    # At step i (time = i*dt_fine), the stock price depends on k (number of up moves)
    # S_i(k) = S0 * u_fine^k * d_fine^(i-k) = S0 * u_fine^(2k-i)
    
    # The value array at step i has length i+1 (k=0 to i)
    
    # Start at step N:
    k_N = np.arange(N + 1)
    S_at_N = S0 * u_fine ** (2 * k_N - N)
    V_next = np.maximum(K - S_at_N, 0.0)  # Value at step N
    
    # Store values at exercise dates
    # We'll collect the critical thresholds
    
    # For each exercise date, we want to find the threshold stock price
    # above which we don't exercise (continue) and below which we do exercise
    
    # Let's track the value function at each exercise date
    
    # Exercise dates in step indices (the time points where we can exercise)
    # Time t_j = j/12, step index = j * N / 12 = j * 100
    # These are steps 100, 200, ..., 1200
    
    # When we reach step i in backward induction, if i is an exercise step,
    # we record the value function
    
    # Let's create a set of exercise steps for quick lookup
    exercise_step_set = set(exercise_steps)
    
    # We'll store the value arrays at exercise dates
    # For threshold finding, we need S_i(k) and V_i(k)
    
    # Let's store: for each exercise date j, the array of (S, V) pairs
    
    # Actually, for the policy, we just need the critical threshold at each date
    # The threshold is the stock price where exercise value = continuation value
    
    # Let's compute it on the fly
    
    # Critical thresholds: threshold[j] for exercise date j (j=1,...,12)
    # At exercise date j, threshold[j] is the S such that max(K-S,0) = continuation_value
    
    # Since for a put, we exercise when S is low, the threshold is the max S
    # where we still exercise
    
    # Let's store the value arrays at exercise dates to compute thresholds later
    
    # For memory efficiency, let's just store the threshold directly
    
    # At an exercise date, given the value array V_i(k) for k=0,...,i:
    # S_i(k) = S0 * u_fine^(2k-i)
    # Exercise value: E_k = max(K - S_i(k), 0)
    # Continuation value: C_k = disc * (p * V_{i+1}(k+1) + (1-p) * V_{i+1}(k))
    # But wait, at step i, the continuation uses V_{i+1} which we've already computed
    
    # Hmm, I'm getting confused with the indexing. Let me restart the backward induction
    # more carefully.
    
    # Let me use a cleaner implementation
    
    # Reset
    # V_next will hold the value at step i+1, indexed by k=0,...,i+1
    # We compute V_current at step i, indexed by k=0,...,i
    
    # At step i:
    # S_i(k) = S0 * u_fine^(2k - i)
    # Exercise value: E_k = max(K - S_i(k), 0)
    # Continuation: C_k = disc_fine * (p_fine * V_next[k+1] + (1-p_fine) * V_next[k])
    #   where V_next has length i+2 (indices 0 to i+1)
    #   V_next[k] corresponds to k up moves at step i+1
    #   V_next[k+1] corresponds to k+1 up moves at step i+1
    # Option value: V_current[k] = max(E_k, C_k)
    
    # If step i is an exercise date, we can also record the threshold
    
    # Let's implement this
    
    # Initialize at step N
    i = N
    k_range = np.arange(i + 1)
    S_i = S0 * u_fine ** (2 * k_range - i)
    V_next = np.maximum(K - S_i, 0.0)
    
    # Store for the last exercise date (step N = 1200)
    # At step N, the value is just the payoff (no continuation since it's the end)
    # But we can still exercise, so the value is max(K-S, 0)
    # The threshold at the last date: exercise if K - S > 0, i.e., S < K
    # So threshold[12] = K
    
    # Let's store thresholds
    thresholds = np.zeros(n_exercise)
    
    # For the last exercise date (j=12, step=1200):
    # Exercise if S < K, so threshold = K
    thresholds[n_exercise - 1] = K
    
    # Now work backwards from step N-1 down to step 1
    # (we don't need step 0 since there's no exercise at time 0)
    
    for i in range(N - 1, 0, -1):
        if time.time() - start_time > DEADLINE:
            break
        
        # S_i(k) for k=0,...,i
        k_range = np.arange(i + 1)
        S_i = S0 * u_fine ** (2 * k_range - i)
        
        # Exercise value
        E = np.maximum(K - S_i, 0.0)
        
        # Continuation value
        # V_next has length i+2 (indices 0 to i+1)
        # C_k = disc * (p * V_next[k+1] + (1-p) * V_next[k])
        C = disc_fine * (p_fine * V_next[1:i+2] + (1.0 - p_fine) * V_next[0:i+1])
        
        # Option value
        V_current = np.maximum(E, C)
        
        # Check if this is an exercise date
        if i in exercise_step_set:
            # Find which exercise date this is
            # i = j * N // n_exercise, so j = i * n_exercise // N
            j = i * n_exercise // N
            if 1 <= j <= n_exercise:
                # Find the threshold: the largest S where E >= C
                # i.e., the largest k where max(K - S_i(k), 0) >= C_k
                # Since S_i(k) increases with k, and E decreases with k (for S < K)
                # while C is roughly decreasing with k (for a put),
                # the exercise region is for small k (low S)
                
                # Find the boundary: the largest k where E_k >= C_k
                exercise_mask = E >= C
                if np.any(exercise_mask):
                    # Find the largest k where we exercise
                    k_exercise = np.where(exercise_mask)[0]
                    k_max_exercise = k_exercise[-1]
                    # Threshold is S_i(k_max_exercise)
                    # But we should interpolate for better accuracy
                    # The threshold is where E = C
                    # For simplicity, use the stock price at the boundary node
                    thresholds[j - 1] = S_i[k_max_exercise]
                else:
                    # Never exercise at this date (unlikely for a put)
                    thresholds[j - 1] = 0.0
        
        V_next = V_current
    
    # Now we have the thresholds for each exercise date
    # thresholds[j] for j=0,...,11 corresponds to exercise dates 1,...,12
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, thresholds, exercise_times):
            self.thresholds = thresholds
            self.exercise_times = exercise_times
        
        def exercise(self, current_state, time_val):
            """
            Determine whether to exercise at the given time.
            
            Args:
                current_state: array of stock prices
                time_val: current time (scalar or array)
            
            Returns:
                Boolean array indicating whether to exercise
            """
            if isinstance(current_state, (list, tuple)):
                current_state = np.array(current_state)
            
            # Determine which exercise date we're at
            # time_val should be one of the exercise times
            # Find the closest exercise date
            
            if np.isscalar(time_val):
                # Find the index of the closest exercise time
                idx = np.argmin(np.abs(self.exercise_times - time_val))
                threshold = self.thresholds[idx]
                return current_state <= threshold
            else:
                # time_val is an array
                idxs = np.argmin(np.abs(self.exercise_times[None, :] - time_val[:, None]), axis=1)
                thresholds_at_time = self.thresholds[idxs]
                return current_state <= thresholds_at_time
    
    return ExercisePolicy(thresholds, exercise_times)

# EVOLVE-BLOCK-END

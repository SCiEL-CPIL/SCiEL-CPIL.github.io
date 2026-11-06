"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_dates = 12
    dt = T / n_dates
    discount = np.exp(-r * dt)
    
    n_paths = training_paths.shape[0]
    
    # Extract S values at each exercise date (dates 1 through 12)
    # training_paths shape: (n_paths, n_dates+1) - S at t=0, t=1/12, ..., t=1
    S_all = training_paths[:, 1:n_dates+1]  # shape (n_paths, 12)
    
    # Backward induction to compute continuation values
    # V[i] = continuation value at date i (before exercise decision)
    # At the last date (i=11, t=1), continuation = 0, so exercise if K-S > 0
    # V[11] = max(K - S, 0) but since we can only exercise, V[11] = K - S (if we exercise)
    # Actually, at the last date, we must exercise if in the money
    # V[i] = max(K - S_i, discount * E[V[i+1] | S_i])
    
    # For the last date (index 11, t=1):
    # We exercise if K - S > 0, so payoff = max(K - S, 0)
    # But since it's the last date, continuation after exercise is 0
    # So V[11] = max(K - S[11], 0)
    
    # Let's compute backward
    # V[i] for i = 11, 10, ..., 0 (but we only exercise at dates 1-12, so indices 0-11 in S_all)
    # Wait, S_all has 12 columns: dates 1 through 12
    # At date 12 (last), we exercise if in the money
    # At date i (0-indexed in S_all, corresponding to exercise date i+1), 
    # we compare K - S[i] with discount * E[continuation from i+1]
    
    # Let's define:
    # cont[i] = continuation value at exercise date i+1 (0-indexed), i.e., the value of NOT exercising
    # At the last date (i=11), cont[11] = 0 (no more dates)
    # At date i, we exercise if K - S[i] > discount * E[cont[i+1] | S[i]]
    
    # Actually, let me think more carefully:
    # V_i(S) = max(K - S, exp(-r*dt) * E[V_{i+1}(S_{i+1}) | S_i = S])
    # where V_{12}(S) = max(K - S, 0) (at the last date, we can only exercise or let it expire worthless)
    
    # Wait, at the last date (t=1), it's European-like: payoff = max(K - S, 0)
    # So V_12(S) = max(K - S, 0)
    # V_i(S) = max(K - S, exp(-r*dt) * E[V_{i+1}(S_{i+1}) | S_i = S]) for i = 1,...,11
    
    # Let's compute V backwards using the training paths
    
    # V at the last date (index 11 in S_all, which is date 12)
    V_next = np.maximum(K - S_all[:, 11], 0.0)  # shape (n_paths,)
    
    # Store exercise decisions for the policy
    # exercise_decision[i, j] = True if path j exercises at date i+1
    exercise_decisions = np.zeros((n_dates, n_paths), dtype=bool)
    
    # For the last date, exercise if in the money
    exercise_decisions[11, :] = S_all[:, 11] < K
    
    # Now go backwards from date 11 to date 1
    # For each date i (0-indexed, corresponding to exercise date i+1):
    # We need E[V_next | S_i]
    # V_next is the value at date i+2 (the next exercise date's value)
    
    # To compute E[V_next | S_i], we can use the training paths:
    # For each path, we know S_i and V_next (which is V at the next date for that path)
    # We can bin S_i and compute average V_next per bin
    
    # But this is path-dependent, so a better approach:
    # Use the fact that under GBM, S_{i+1} = S_i * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
    # So given S_i, S_{i+1} is lognormal
    # But V_next is a function of S_{i+1} and the future exercise policy
    
    # Alternative: use the training paths directly
    # For each path j, at date i, we have S_i[j] and the realized V_next[j] (value at date i+1 under optimal exercise)
    # We can estimate E[V_next | S_i = s] by binning
    
    # Let's use a simple approach: for each date, create bins of S values and compute average V_next per bin
    
    # We'll store the continuation value function for each date
    # cont_value[i](s) = exp(-r*dt) * E[V_{i+1} | S_i = s]
    
    # For interpolation, we'll use the training paths to create a lookup table
    
    # Let's compute backward, storing the continuation values
    # cont[i] = array of continuation values for each path at date i
    # This is exp(-r*dt) * E[V_{i+1} | S_i]
    
    # Actually, let's just store the threshold for each date
    # At date i, exercise if S < threshold[i]
    # threshold[i] = K - cont_value[i](S) where cont_value[i](S) is the continuation value
    
    # Simpler: for each date, find the optimal exercise boundary
    # By backward induction, the optimal policy is: exercise if S <= b_i(t)
    
    # Let's compute the continuation value for each path at each date
    # cont[i, j] = exp(-r*dt) * E[V_{i+1} | S_i = S_i[j]]
    
    # To estimate this, we can use the training paths:
    # For path j at date i, we want E[V_{i+1} | S_i = S_i[j]]
    # Since the paths are from GBM, we can use the fact that given S_i, the distribution of S_{i+1} is known
    # But V_{i+1} depends on the future exercise policy, which we're computing
    
    # Let's use a different approach: use the training paths to directly estimate the conditional expectation
    # For each date i, create a grid of S values, and for each S, compute the average V_next over paths with S_i close to S
    
    # This is essentially a nonparametric regression
    
    # Let's implement this:
    
    # First, let's precompute the V at each date for each path
    # V[11, j] = max(K - S_all[:, 11][j], 0)
    # V[i, j] = max(K - S_all[:, i][j], discount * E[V[i+1] | S_i = S_all[:, i][j]])
    
    # To compute E[V[i+1] | S_i = S_all[:, i][j]], we can:
    # 1. For each path j, find other paths with similar S_i values
    # 2. Average their V[i+1] values
    
    # But this is expensive. Let's use binning instead.
    
    # Create a grid of S values for each date
    # For simplicity, use the same grid for all dates
    
    # Let's use quantile-based binning
    
    n_bins = 50
    # Create bins based on the range of S values
    S_min = np.min(S_all)
    S_max = np.max(S_all)
    bin_edges = np.linspace(S_min, S_max, n_bins + 1)
    
    # For each date, compute the average V_next per bin
    # V_next is the value at the next date
    
    # Let's store the continuation value function as a piecewise constant function
    # cont_func[i](s) = average of V_next for paths with S_i in the bin containing s
    
    # Initialize
    V = np.zeros((n_dates, n_paths))
    V[11, :] = np.maximum(K - S_all[:, 11], 0.0)
    
    # For the last date, exercise if in the money
    exercise_decisions[11, :] = S_all[:, 11] < K
    
    # Go backwards
    for i in range(10, -1, -1):
        if time.time() > deadline:
            break
        
        # Compute E[V[i+1] | S_i] for each path
        # Bin S_i and compute average V[i+1] per bin
        
        # Assign each path to a bin
        bin_indices = np.digitize(S_all[:, i], bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Compute sum and count per bin
        bin_sums = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        np.add.at(bin_sums, bin_indices, V[i+1, :])
        np.add.at(bin_counts, bin_indices, 1)
        
        # Average per bin (handle zero counts)
        with np.errstate(divide='ignore', invalid='ignore'):
            bin_avg = np.where(bin_counts > 0, bin_sums / bin_counts, 0.0)
        
        # For each path, get the continuation value
        cont_val = bin_avg[bin_indices]
        
        # Discounted continuation value
        disc_cont = discount * cont_val
        
        # Exercise if K - S > disc_cont
        exercise_here = (K - S_all[:, i]) > disc_cont
        
        # V[i] = max(K - S, disc_cont)
        V[i, :] = np.maximum(K - S_all[:, i], disc_cont)
        
        # Store exercise decision
        exercise_decisions[i, :] = exercise_here
    
    # Now we have the exercise policy from the training data
    # But we need to create a smooth policy that generalizes
    
    # For the exercise function, we need to return a boolean for each state
    # The policy: at date i, exercise if S < threshold[i]
    # Let's estimate the threshold for each date
    
    # From the training data, for each date i, find the boundary
    # threshold[i] = the S value where K - S = disc_cont
    
    # We can estimate this by finding, for each date, the S values where exercise happens
    # and taking a percentile or using the bin structure
    
    # Let's use the bin structure to create a smooth threshold
    # For each date, the threshold is where K - S = discount * E[V[i+1] | S]
    
    # Let's compute the threshold for each bin edge
    # Actually, let's just use the exercise_decisions to create a simple threshold policy
    
    # For each date, find the maximum S where exercise happens (with some smoothing)
    thresholds = np.zeros(n_dates)
    for i in range(n_dates):
        if time.time() > deadline:
            break
        # Find paths that exercise at date i
        ex_mask = exercise_decisions[i, :]
        if np.any(ex_mask):
            # Use the 95th percentile of S values where exercise happens
            # This provides some smoothing
            ex_S = S_all[:, i][ex_mask]
            # Use a high percentile to be conservative
            thresholds[i] = np.percentile(ex_S, 95)
        else:
            thresholds[i] = 0.0  # Never exercise
    
    # But this might not be optimal. Let's use a better approach:
    # For each date, create a lookup table of S -> exercise decision
    # and interpolate
    
    # Actually, let's use the bin-based continuation value to create a smooth policy
    # For each date, we have bin_avg which gives E[V[i+1] | S_i in bin]
    # The threshold for bin b is where K - S = discount * bin_avg[b]
    # So S_threshold[b] = K - discount * bin_avg[b]
    
    # Let's recompute more carefully
    
    # Store the continuation value per bin for each date
    cont_per_bin = np.zeros((n_dates, n_bins))
    
    V[11, :] = np.maximum(K - S_all[:, 11], 0.0)
    exercise_decisions[11, :] = S_all[:, 11] < K
    
    # For the last date, the "continuation" is 0, so we always exercise if in the money
    cont_per_bin[11, :] = 0.0
    
    for i in range(10, -1, -1):
        if time.time() > deadline:
            break
        
        bin_indices = np.digitize(S_all[:, i], bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        bin_sums = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        np.add.at(bin_sums, bin_indices, V[i+1, :])
        np.add.at(bin_counts, bin_indices, 1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            bin_avg = np.where(bin_counts > 0, bin_sums / bin_counts, 0.0)
        
        cont_per_bin[i, :] = bin_avg
        
        cont_val = bin_avg[bin_indices]
        disc_cont = discount * cont_val
        
        exercise_here = (K - S_all[:, i]) > disc_cont
        V[i, :] = np.maximum(K - S_all[:, i], disc_cont)
        exercise_decisions[i, :] = exercise_here
    
    # Now create the exercise policy
    # For each date, we have cont_per_bin[i, b] = E[V[i+1] | S_i in bin b]
    # The threshold for bin b is S_th[b] = K - discount * cont_per_bin[i, b]
    # If S < S_th[b], exercise
    
    # Let's create a function that, given S and date, returns whether to exercise
    
    # Precompute the threshold for each bin for each date
    # threshold_per_bin[i, b] = K - discount * cont_per_bin[i, b]
    threshold_per_bin = K - discount * cont_per_bin
    
    # For the last date, threshold is K (exercise if S < K)
    threshold_per_bin[11, :] = K
    
    # Now create the policy object
    class ExercisePolicy:
        def __init__(self, threshold_per_bin, bin_edges, n_dates):
            self.threshold_per_bin = threshold_per_bin
            self.bin_edges = bin_edges
            self.n_dates = n_dates
            self.n_bins = len(bin_edges) - 1
        
        def exercise(self, current_state, time_val):
            # current_state: array of S values for all paths
            # time_val: the current time (to determine which date we're at)
            # Return: boolean array, True if exercise
            
            # Determine which exercise date we're at
            # time_val is in [0, 1], exercise dates are at 1/12, 2/12, ..., 12/12
            # Find the closest date index
            date_idx = int(round(time_val * 12)) - 1
            date_idx = np.clip(date_idx, 0, self.n_dates - 1)
            
            # Assign each S value to a bin
            S_vals = np.asarray(current_state)
            bin_indices = np.digitize(S_vals, self.bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
            
            # Get the threshold for each S value
            thresholds = self.threshold_per_bin[date_idx, bin_indices]
            
            # Exercise if S < threshold
            return S_vals < thresholds
    
    policy = ExercisePolicy(threshold_per_bin, bin_edges, n_dates)
    return policy
# EVOLVE-BLOCK-END

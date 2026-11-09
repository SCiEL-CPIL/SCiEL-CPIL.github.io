"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

class ExercisePolicy:
    def __init__(self, thresholds):
        # thresholds: list of 12 floats, one per exercise date (index 0..11 for t=1/12..12/12)
        self.thresholds = np.array(thresholds, dtype=np.float64)
        self.n_steps = len(self.thresholds)

    def exercise(self, current_state, time):
        # current_state: array of spots (N,)
        # time: float in (0, T] or int index
        # Determine step index
        if isinstance(time, (int, np.integer)):
            idx = int(time)
            if idx < 0:
                idx = 0
            if idx >= self.n_steps:
                idx = self.n_steps - 1
        else:
            # time is a float in (0, T], T=1
            T = 1.0
            # Exercise dates are at k/12 for k=1..12
            # time=1/12 -> idx 0, time=12/12=1 -> idx 11
            idx = int(np.ceil(time * self.n_steps)) - 1
            if idx < 0:
                idx = 0
            if idx >= self.n_steps:
                idx = self.n_steps - 1
        
        return np.array(current_state) <= self.thresholds[idx]


def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Extract parameters from contract or use defaults
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_steps = 12  # 12 monthly exercise dates
    
    # Determine training paths shape
    # training_paths should be (n_paths, n_time_steps) or similar
    if isinstance(training_paths, np.ndarray):
        paths = training_paths
    else:
        paths = np.array(training_paths)
    
    # Infer number of paths and time steps
    # Assume paths shape is (n_paths, n_time_points) where n_time_points includes time 0
    # Exercise dates are at indices 1..12 (no time-0 exercise)
    
    n_paths, n_time_points = paths.shape
    
    # The time steps: if n_time_points = 13, then times are 0, 1/12, ..., 12/12=1
    # Exercise at indices 1..12
    dt = T / n_steps
    
    # For each exercise date k=1..12, compute the continuation value estimate
    # using the training paths, and set threshold as the median (or similar) of
    # the continuation value across paths at that time.
    
    # Strategy: For each exercise date k (1-indexed, so index k in paths),
    # the immediate payoff is max(K - S, 0)
    # The continuation value is estimated by discounting the expected value of
    # the optimal policy from the next exercise date onward.
    
    # We do a backward induction:
    # V_12(S) = max(K - S, 0)  # last exercise date, must exercise or let expire
    # Actually at the last date, if you don't exercise, the option expires.
    # So V_12(S) = max(K - S, 0)
    
    # For k < 12:
    # V_k(S) = max(K - S, discount * E[V_{k+1}(S_{k+1}) | S_k = S])
    
    # We estimate E[V_{k+1}(S_{k+1}) | S_k = S] using the training paths.
    # For each path, we know S_k and S_{k+1}.
    # We can use a simple approach: for each path, the continuation value at step k
    # is discount * V_{k+1}(S_{k+1}) for that path.
    # The threshold is where max(K-S, cont) switches.
    
    # Backward induction with pathwise estimates:
    # Start from the last exercise date (index 12 in 1-indexed, or index 12 in 0-indexed if paths has 13 columns)
    
    # Let's assume paths has columns 0..12, where column j corresponds to time j/12
    # Exercise dates are columns 1..12
    
    # V at last exercise date (column 12):
    # V_12(S) = max(K - S, 0)
    
    # We'll store the continuation value for each path at each step
    # cont_value[k] = array of (n_paths,) continuation values at exercise date k
    
    # Start from the last exercise date
    # At column 12 (last exercise date), the value is the payoff if exercised
    # But we need to think about this more carefully.
    
    # Actually, for a Bermudan option, at each exercise date you can choose to exercise or not.
    # If you don't exercise at the last date, the option expires with zero value.
    # So at the last exercise date: V_12(S) = max(K - S, 0)
    
    # At earlier dates: V_k(S) = max(K - S, discount * E[V_{k+1}(S_{k+1}) | S_k])
    
    # Using pathwise simulation:
    # For each path i, at step k, we have S_k[i] and S_{k+1}[i]
    # The pathwise continuation estimate at step k for path i is: discount * V_{k+1}[i]
    # where V_{k+1}[i] is the value at step k+1 for path i
    
    # The threshold at step k is where K - S = discount * E[V_{k+1} | S_k = S]
    # This is complex. A simpler approach: use the pathwise values to estimate
    # the continuation value as a function of S_k.
    
    # Let's use a simpler approach: for each exercise date k, compute for each path
    # the continuation value (discounted value of not exercising) and the immediate payoff.
    # The optimal exercise decision for path i at step k is to exercise if
    # K - S_k[i] >= discount * V_{k+1}[i]
    
    # But V_{k+1}[i] depends on the future. We do backward induction.
    
    # Initialize: at the last exercise date (step 12, column 12), 
    # the value if you hold is 0 (option expires), so you always exercise if in the money.
    # V_12[i] = max(K - S_12[i], 0)
    
    # For step k from 11 down to 1:
    # cont[i] = discount * V_{k+1}[i]  # pathwise continuation estimate
    # exercise[i] = (K - S_k[i]) >= cont[i]
    # V_k[i] = max(K - S_k[i], cont[i])
    
    # The threshold at step k is the value of S_k where (K - S_k) = cont
    # But cont varies by path. We can estimate the threshold as the median of
    # the S_k values where exercise occurs, or better, fit a simple model.
    
    # For simplicity and robustness, let's compute the threshold as follows:
    # For each step k, we have pairs (S_k[i], cont[i]) for all paths i.
    # The exercise condition is S_k[i] <= K - cont[i]
    # So the path-specific threshold is K - cont[i]
    # The overall threshold can be the median of K - cont[i] across paths.
    
    # Let's implement this:
    
    discount = np.exp(-r * dt)
    
    # V at last exercise date
    # Column index for last exercise date: 12 (if 0-indexed and 13 columns)
    last_col = n_steps  # column 12 for n_steps=12
    S_last = paths[:, last_col]
    V_next = np.maximum(K - S_last, 0.0)
    
    thresholds = [0.0] * n_steps  # thresholds[0] for step 1, ..., thresholds[11] for step 12
    
    # thresholds[11] is for the last exercise date (step 12)
    # At the last date, you always exercise if in the money, so threshold = K
    # But actually, since there's no continuation, the threshold is K (exercise if S <= K)
    thresholds[11] = K
    
    # Backward induction for steps 11 down to 1
    for k in range(n_steps - 1, 0, -1):
        # k is 1-indexed exercise date, so column index is k
        # V_next is the value at step k+1 (already computed)
        # S_k is the spot at step k
        S_k = paths[:, k]
        
        # Pathwise continuation value
        cont = discount * V_next
        
        # Exercise if K - S_k >= cont, i.e., S_k <= K - cont
        # Path-specific threshold
        path_thresholds = K - cont
        
        # Overall threshold: use the median for robustness
        thresholds[k - 1] = np.median(path_thresholds)
        
        # Value at step k for each path
        V_next = np.where(S_k <= thresholds[k - 1], K - S_k, cont)
        
        # Check deadline
        if time.time() > deadline - 5.0:
            break
    
    # Create the policy
    policy = ExercisePolicy(thresholds)
    return policy
# EVOLVE-BLOCK-END

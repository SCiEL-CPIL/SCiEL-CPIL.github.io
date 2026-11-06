"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Extract parameters from contract if available, else use defaults
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_exercise', 12)
    
    dt = T / n_dates
    discount = np.exp(-r * dt)
    
    paths = np.asarray(training_paths, dtype=np.float64)
    
    # Ensure shape is (n_paths, n_dates + 1) or (n_paths, n_dates)
    # The task says training_paths has 12 monthly exercise dates (no time-0).
    # Usually training_paths includes S_0..S_12 (13 cols) or S_1..S_12 (12 cols).
    # The current program assumes S_all = training_paths[:, 1:n_dates+1], implying 13 cols.
    # Let's be robust.
    if paths.ndim == 1:
        paths = paths.reshape(-1, n_dates)
        
    # If the second dimension is n_dates + 1, we assume column 0 is S_0.
    # If it is n_dates, we assume columns are S_1..S_n_dates.
    if paths.shape[1] > n_dates:
        # Likely includes S_0. Use cols 1..n_dates
        S_all = paths[:, 1:n_dates+1]
    else:
        # Likely S_1..S_n_dates
        S_all = paths[:, :n_dates]
        
    # If for some reason it's transposed (n_dates, n_paths), fix it
    if S_all.shape[0] < S_all.shape[1]:
        S_all = S_all.T

    n_paths = S_all.shape[0]
    
    # V[i] will store the value of the option at exercise date i (0-indexed i)
    # V[11] is the value at the last date (maturity)
    V = np.zeros((n_dates, n_paths))
    
    # At maturity (last date, index n_dates-1), the value is the European payoff
    V[-1, :] = np.maximum(K - S_all[:, -1], 0.0)
    
    # We will store the policy parameters
    # For each date i, we store:
    #   nodes[i]: array of S values (grid points)
    #   cont_vals[i]: array of continuation values at those nodes
    policy_nodes = []
    policy_cont_vals = []
    
    # Number of grid points for interpolation
    # 200 is a good balance. 400 might be better but slower.
    n_nodes = 200
    
    # Backward induction
    for i in range(n_dates - 2, -1, -1):
        if time.time() > deadline:
            # If we run out of time, break and fill remaining with heuristic
            pass
        
        S_i = S_all[:, i]
        V_next = V[i+1, :]
        
        # Estimate C_i(S) = E[V_next | S_i = S]
        # Use quantile binning for robust coverage
        
        try:
            # Sort S_i and V_next together
            order = np.argsort(S_i)
            S_sorted = S_i[order]
            V_sorted = V_next[order]
            
            # Define bin indices
            bin_size = n_paths // n_nodes
            if bin_size == 0:
                bin_size = 1
                n_nodes = n_paths
            
            nodes = np.zeros(n_nodes)
            cont_vals = np.zeros(n_nodes)
            
            for k in range(n_nodes):
                start_idx = k * bin_size
                end_idx = (k + 1) * bin_size if k < n_nodes - 1 else n_paths
                if start_idx >= n_paths:
                    break
                bin_S = S_sorted[start_idx:end_idx]
                bin_V = V_sorted[start_idx:end_idx]
                
                # Use median for S to be robust to outliers within bin
                # Use mean for V to estimate expectation
                nodes[k] = np.median(bin_S)
                cont_vals[k] = np.mean(bin_V)
            
            # Ensure nodes are strictly increasing for np.interp
            mask = np.concatenate(([True], np.diff(nodes) > 1e-9))
            nodes = nodes[mask]
            cont_vals = cont_vals[mask]
            
            # Store policy
            policy_nodes.append(nodes)
            policy_cont_vals.append(cont_vals)
            
            # Compute V[i] for these training paths
            C_i = np.interp(S_i, nodes, cont_vals, left=cont_vals[0], right=cont_vals[-1])
            
            # Discounted continuation value
            disc_cont = discount * C_i
            
            # Exercise if K - S_i > disc_cont
            # V[i] is the max of exercise payoff and continuation
            V[i, :] = np.maximum(K - S_i, disc_cont)
            
        except Exception as e:
            # Fallback if quantile/binning fails
            # Use a simple threshold: Exercise if S < K
            # This is a safe default for puts
            policy_nodes.append(np.array([0.0, K, 2*K]))
            policy_cont_vals.append(np.array([K, 0.0, 0.0]))
            C_i = np.interp(S_i, [0.0, K, 2*K], [K, 0.0, 0.0])
            disc_cont = discount * C_i
            V[i, :] = np.maximum(K - S_i, disc_cont)

    # If we broke out of the loop early due to deadline, 
    # policy_nodes might have fewer than n_dates entries.
    # We need to fill the missing earlier dates.
    # The list is built from i=n_dates-2 down to 0.
    # So policy_nodes[0] is for i=n_dates-2. policy_nodes[-1] is for i=0 (if complete).
    # If we stopped at i=5, we have nodes for 10,9,8,7,6,5 (assuming n_dates=12).
    # We are missing 4,3,2,1,0.
    # We should prepend copies of the *last* calculated one (which is the earliest time we have, i=5)
    
    while len(policy_nodes) < n_dates:
        # Prepend a copy of the last available (earliest time calculated)
        policy_nodes.insert(0, policy_nodes[-1])
        policy_cont_vals.insert(0, policy_cont_vals[-1])

    # Now policy_nodes has n_dates entries.
    # policy_nodes[0] corresponds to i=0 (Date 1).
    # policy_nodes[-1] corresponds to i=n_dates-1 (Date 12).
    # Wait, the loop filled from back to front.
    # If complete:
    # i=10 -> policy_nodes[0]
    # i=9  -> policy_nodes[1]
    # ...
    # i=0  -> policy_nodes[11]
    # So policy_nodes[0] is for Date 11 (index 10).
    # policy_nodes[11] is for Date 1 (index 0).
    # This is REVERSED relative to the date index!
    
    # Let's check the order again.
    # Loop: for i in range(10, -1, -1):
    # i=10: append nodes_10. List: [nodes_10]
    # i=9: append nodes_9. List: [nodes_10, nodes_9]
    # ...
    # i=0: append nodes_0. List: [nodes_10, nodes_9, ..., nodes_0]
    
    # In ExercisePolicy.exercise, we use date_idx to index this list.
    # date_idx 0 corresponds to Date 1 (i=0).
    # But policy_nodes[0] is nodes_10 (Date 11).
    # So we need to REVERSE the list before passing to the class, or index correctly.
    
    # Let's reverse the lists so that index 0 corresponds to Date 1 (i=0).
    policy_nodes.reverse()
    policy_cont_vals.reverse()
    
    # Now:
    # policy_nodes[0] is for i=0 (Date 1)
    # policy_nodes[1] is for i=1 (Date 2)
    # ...
    # policy_nodes[11] is for i=11 (Date 12)
    
    class ExercisePolicy:
        def __init__(self, nodes_list, vals_list, n_dates, K, discount):
            self.nodes_list = nodes_list
            self.vals_list = vals_list
            self.n_dates = n_dates
            self.discount = discount
            self.K = K
            
        def exercise(self, current_state, time_val):
            # time_val is the current time.
            # Exercise dates are at t = 1/12, 2/12, ..., 12/12.
            # Determine the index of the current exercise date.
            # time_val * 12 gives the date number (1 to 12).
            # Index in our arrays (0 to 11) is date_number - 1.
            
            try:
                date_idx = int(round(time_val * 12)) - 1
                if date_idx < 0:
                    date_idx = 0
                if date_idx >= self.n_dates:
                    date_idx = self.n_dates - 1
            except:
                date_idx = 0
                
            S_vals = np.asarray(current_state, dtype=float)
            
            nodes = self.nodes_list[date_idx]
            vals = self.vals_list[date_idx]
            
            # Get continuation value for each S
            C = np.interp(S_vals, nodes, vals, left=vals[0], right=vals[-1])
            
            # Discounted continuation
            disc_C = self.discount * C
            
            # Exercise if intrinsic value > continuation
            intrinsic = self.K - S_vals
            
            return intrinsic > disc_C

    policy = ExercisePolicy(policy_nodes, policy_cont_vals, n_dates, K, discount)
    return policy
# EVOLVE-BLOCK-END

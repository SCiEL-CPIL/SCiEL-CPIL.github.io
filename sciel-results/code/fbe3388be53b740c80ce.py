import numpy as np

class BermudanPolicy:
    def __init__(self, training_paths, contract):
        self.K = contract.get('K', 100.0)
        self.r = contract.get('r', 0.05)
        self.sigma = contract.get('sigma', 0.2)
        self.T = contract.get('T', 1.0)
        self.num_ex_steps = 12
        
        # Tree parameters
        # Use 10 sub-steps per exercise date for smoothness
        self.sub_steps = 10
        self.N = self.num_ex_steps * self.sub_steps
        self.dt = self.T / self.N
        
        u = np.exp(self.sigma * np.sqrt(self.dt))
        d = 1.0 / u
        p = (np.exp(self.r * self.dt) - d) / (u - d)
        
        # Backward induction
        # V is an array of size N+1 at each step
        # We only need to store Continuation Values at exercise dates
        # Exercise dates are at steps: 10, 20, ..., 120
        # Note: Step 0 is t=0 (no exercise). Step 120 is t=T (maturity).
        
        # Initialize V at step N (maturity)
        # S_j at step N: S0 * u^j * d^(N-j)
        # We need to know S0. The contract might not provide S0 explicitly?
        # The training paths contain S0.
        # Let's extract S0 from training_paths.
        if training_paths.ndim == 1:
            training_paths = training_paths.reshape(1, -1)
        
        # Assume first column is t=0
        if training_paths.shape[1] > 0:
            self.S0 = np.mean(training_paths[:, 0])
        else:
            self.S0 = 100.0 # Fallback

        # Precompute log-spacings
        log_u = np.log(u)
        log_d = np.log(d)
        
        # Store continuation values for each exercise date index 1..11
        # cont_vals[k] will be an array of size (m_k + 1) where m_k = k * sub_steps
        # Actually, we can just store the whole V array at each exercise step?
        # No, we only need C = discount * E[V_next].
        
        # V_next starts as terminal payoff at step N
        # We iterate backwards.
        
        # To save memory and time, we can just compute C at the exercise steps and store them.
        # But we need V at all intermediate steps to compute the expectation.
        # So we keep a full V array of size N+1.
        
        V = np.zeros(self.N + 1)
        
        # Initialize V at step N
        # S_j = S0 * u^j * d^(N-j)
        # j = 0..N
        js = np.arange(self.N + 1)
        S_N = self.S0 * (u ** js) * (d ** (self.N - js))
        V[:] = np.maximum(self.K - S_N, 0.0)
        
        self.cont_vals = {} # key: exercise step index (1..11), value: array of C for that step
        
        # Iterate from step N-1 down to 1
        for m in range(self.N - 1, 0, -1):
            # Compute E[V_next]
            # E[V_{m+1}] = p * V_{j+1} + (1-p) * V_{j-1} ?
            # Careful with indices.
            # At step m, node j goes to j+1 (up) with prob p and j-1 (down) with prob 1-p?
            # Standard binomial:
            # S_{j} at step m.
            # Up: S_{j+1} at step m+1.
            # Down: S_{j-1} at step m+1.
            # So V[m, j] = exp(-r dt) * (p * V[m+1, j+1] + (1-p) * V[m+1, j-1])
            
            # V is currently V[m+1]
            # We need to compute V[m]
            V_curr = np.zeros(m + 1)
            
            # For j=0..m
            # V[m, j] = disc * (p * V[m+1, j+1] + (1-p) * V[m+1, j])
            # Wait, the down move from j goes to j-1?
            # If we start at j, up is j+1, down is j-1.
            # But at step m+1, the indices are 0..m+1.
            # From node j at step m:
            # Up to node j+1 at step m+1.
            # Down to node j at step m+1? No.
            # Let's check standard binomial indexing.
            # Usually, S_{i,j} where i is time, j is number of up moves.
            # S_{i,j} = S_0 u^j d^{i-j}.
            # From (i, j):
            # Up -> (i+1, j+1)
            # Down -> (i+1, j)
            # So V[i, j] = disc * (p * V[i+1, j+1] + (1-p) * V[i+1, j])
            
            # So V_curr[j] = disc * (p * V[j+1] + (1-p) * V[j])
            disc = np.exp(-self.r * self.dt)
            
            V_curr[:] = disc * (p * V[1:m+1] + (1-p) * V[0:m])
            
            # Check if step m is an exercise date
            # Exercise dates are at steps 10, 20, ..., 120
            # m is the current step index (1..N-1)
            if m % self.sub_steps == 0:
                # This is an exercise date
                # Compute S at this step
                js_m = np.arange(m + 1)
                S_m = self.S0 * (u ** js_m) * (d ** (m - js_m))
                
                # Immediate payoff
                P = np.maximum(self.K - S_m, 0.0)
                
                # Continuation value is V_curr (which is disc * E[V_next])
                C = V_curr
                
                # Store C for this exercise step
                # We need to map this step to the exercise index (1..11)
                ex_idx = m // self.sub_steps
                if ex_idx < self.num_ex_steps: # 1..11
                    self.cont_vals[ex_idx] = C
                    # Also store S_m for interpolation later?
                    # Or just store the log-S values?
                    # We can recompute S_m during evaluation if we store u, d, S0.
                    # But storing the log-S values is safer for interpolation.
                    self.cont_vals[ex_idx + 'S'] = np.log(S_m)
                
                # Update V for next iteration (backwards)
                # V = max(P, C)
                V_curr[:] = np.maximum(P, C)
            
            # Update V for next iteration
            V = V_curr
            
        # We don't need to store step 12 (maturity) because we always exercise if ITM.
        # The policy at step 12 is simply S < K.

    def exercise(self, current_state, time):
        S = np.asarray(current_state, dtype=float)
        t_idx = int(round(time))
        
        if t_idx < 1 or t_idx > self.num_ex_steps:
            return np.zeros_like(S, dtype=bool)
            
        if t_idx == self.num_ex_steps:
            return S < self.K
            
        # Get continuation values for this step
        C_vals = self.cont_vals.get(t_idx)
        S_logs = self.cont_vals.get(t_idx + 'S')
        
        if C_vals is None or S_logs is None:
            return S < self.K
            
        # Interpolate C for each S in current_state
        # S_logs is the log-S of the nodes at this step
        # C_vals is the continuation value at those nodes
        
        # For each s in S:
        # 1. log_s = log(s)
        # 2. Find index j such that S_logs[j] <= log_s <= S_logs[j+1]
        # 3. Interpolate C between C_vals[j] and C_vals[j+1]
        
        C_interp = np.zeros_like(S)
        
        # Vectorized search?
        # np.searchsorted(S_logs, log_S, side='right') - 1
        log_S = np.log(S)
        
        # Handle boundaries
        # If log_S < S_logs[0], use C_vals[0]
        # If log_S > S_logs[-1], use C_vals[-1]
        
        indices = np.searchsorted(S_logs, log_S, side='right') - 1
        indices = np.clip(indices, 0, len(S_logs) - 2)
        
        # We need to interpolate between indices[j] and indices[j] + 1
        j = indices
        j1 = j + 1
        
        # Weights
        # w = (log_S - S_logs[j]) / (S_logs[j1] - S_logs[j])
        denom = S_logs[j1] - S_logs[j]
        # Avoid division by zero (should not happen with distinct nodes)
        w = (log_S - S_logs[j]) / denom
        
        # Interpolate
        C_interp = (1 - w) * C_vals[j] + w * C_vals[j1]
        
        # Exercise if Payoff >= Continuation
        P = np.maximum(self.K - S, 0.0)
        
        return P >= C_interp

def fit_exercise_policy(training_paths, contract):
    return BermudanPolicy(training_paths, contract)

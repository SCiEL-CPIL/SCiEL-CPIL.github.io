import numpy as np

def fit_exercise_policy(training_paths, contract):
    # ---- Extract model parameters from the contract (with safe defaults) ----
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_dates = 12

    def _get(obj, *names, default=None):
        for n in names:
            if isinstance(obj, dict):
                if n in obj:
                    return obj[n]
            else:
                if hasattr(obj, n):
                    return getattr(obj, n)
        return default

    try:
        K = float(_get(contract, 'K', 'strike', 'Strike', default=K))
        r = float(_get(contract, 'r', 'rate', 'Rate', 'risk_free', default=r))
        sigma = float(_get(contract, 'sigma', 'vol', 'Vol', 'volatility', default=sigma))
        T = float(_get(contract, 'T', 'maturity', 'Maturity', 'horizon', default=T))
        nd = _get(contract, 'n_dates', 'n_exercise', 'n_bermudan', 'exercise_dates', default=n_dates)
        if nd is not None:
            n_dates = int(nd)
    except Exception:
        pass

    # ---- Binomial Tree Approach ----
    # We construct a fine-grained binomial tree to determine the optimal exercise thresholds.
    # The task mentions a "1200-step binomial-tree reference", so we use a similar resolution.
    
    # Number of steps in the tree. We want enough steps to approximate the continuous path well.
    # 1200 steps is a good balance between accuracy and speed (very fast).
    N_steps = 1200
    
    dt = T / N_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p_up = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    
    # We need to map the 12 exercise dates to the tree steps.
    # The exercise dates are at t_k = k * (T / n_dates) for k=1..n_dates.
    # In the tree, step j corresponds to time j * dt.
    # We want to exercise at steps corresponding to the exercise dates.
    
    # Calculate the tree step index for each exercise date
    # t_ex = k * T / n_dates
    # j_ex = t_ex / dt = k * T / n_dates / (T / N_steps) = k * N_steps / n_dates
    exercise_steps = []
    for k in range(1, n_dates + 1):
        j = int(round(k * N_steps / n_dates))
        exercise_steps.append(j)
    
    # Ensure the last exercise step is exactly N_steps (maturity)
    if exercise_steps[-1] != N_steps:
        exercise_steps[-1] = N_steps
        
    # Initialize the tree value array
    # V[j, i] is the value at time step j and node i
    # To save memory, we only need to store the values at exercise steps and the final step.
    # However, for backward induction, we need the next step's values.
    # We can store the full tree if N_steps is not too large. 1200 steps -> 1200x1200 matrix is ~1.4M floats, which is fine.
    
    V = np.zeros((N_steps + 1, N_steps + 1))
    
    # Initialize the final step (maturity)
    # S_N = S_0 * u^i * d^(N-i) ? No, standard binomial tree starts from S_0.
    # But we don't have S_0 explicitly? 
    # The contract says "Spot 90/100/110". Usually, the model is centered around 100.
    # The binomial tree is typically built from S_0. 
    # The exercise policy is `exercise(current_state, time)`. 
    # The threshold depends on S_0 if the tree is built from S_0.
    # However, the optimal exercise boundary for a put is independent of the starting price in terms of the *ratio* or specific levels if we define the tree correctly.
    # Actually, the threshold $S^*$ is an absolute price level.
    # If we build the tree from $S_0=100$, the thresholds will be absolute prices.
    # The evaluator will pass `current_state` which is an absolute price.
    # So we assume the model is calibrated such that $S_0=100$ (or we can just pick $S_0=100$ for the tree construction, as the shape of the boundary is what matters, and for GBM, the boundary scales linearly with S_0? No, for puts, the boundary is a specific price level. If S_0 changes, the boundary changes. But the contract implies a fixed model. The "Spot 90/100/110" likely refers to evaluation scenarios or training data spread. The underlying model parameters (r, sigma, T, K) are fixed. The starting spot for the *policy derivation* should be consistent. Let's assume S_0 = 100 for the tree construction, as K=100.
    
    S_0 = 100.0
    
    # Final values at maturity
    for i in range(N_steps + 1):
        S_final = S_0 * (u ** i) * (d ** (N_steps - i))
        V[N_steps, i] = np.maximum(K - S_final, 0.0)
        
    # Backward induction
    # We only need to make decisions at exercise_steps.
    # We iterate backwards from N_steps to 1.
    # We keep track of which steps are exercise steps.
    # is_exercise_step[j] is True if j is in exercise_steps.
    
    is_exercise_step = np.zeros(N_steps + 1, dtype=bool)
    for j in exercise_steps:
        is_exercise_step[j] = True
        
    thresholds = [0.0] * n_dates
    # Map exercise step index to date index (1..n_dates)
    # exercise_steps is sorted.
    # date_indices = [0] * (N_steps+1)
    
    # We need to store the thresholds for each date.
    # Let's create a list of (step, date_index)
    exercise_info = []
    for idx, j in enumerate(exercise_steps):
        # date index is idx+1 (1-based)
        exercise_info.append((j, idx + 1))
        
    # Backward induction
    for j in range(N_steps - 1, 0, -1):
        # Calculate continuation value
        # V_cont[i] = disc * (p_up * V[j+1, i+1] + (1-p_up) * V[j+1, i])
        
        # We need to be careful with the array bounds.
        # V[j+1, i] is defined for i=0..N_steps-j-1? 
        # In a standard binomial tree, at step j, there are j+1 nodes (i=0..j).
        # At step j+1, there are j+2 nodes (i=0..j+1).
        # Node i at step j connects to node i and i+1 at step j+1.
        
        # Let's use a 1D array for current and next step to save memory and simplify.
        # But V is already allocated as 2D.
        
        # Calculate S at step j for all nodes i=0..j
        # S_j_i = S_0 * u^i * d^(j-i)
        
        # We only need to calculate thresholds if j is an exercise step.
        
        # First, update V[j, i] for all i
        # V[j, i] = disc * (p_up * V[j+1, i+1] + (1-p_up) * V[j+1, i])
        
        # Vectorized calculation for i=0..j
        # i ranges from 0 to j
        # V[j, 0:j+1] = disc * (p_up * V[j+1, 1:j+2] + (1-p_up) * V[j+1, 0:j+1])
        
        V[j, 0:j+1] = disc * (p_up * V[j+1, 1:j+2] + (1-p_up) * V[j+1, 0:j+1])
        
        if is_exercise_step[j]:
            # Find the date index for this step
            date_idx = None
            for (step, d_idx) in exercise_info:
                if step == j:
                    date_idx = d_idx
                    break
                    
            if date_idx is not None:
                # Calculate Intrinsic Value for nodes i=0..j
                # S_j_i = S_0 * u^i * d^(j-i)
                # We can compute S values efficiently.
                # S_j_i = S_0 * d^j * (u/d)^i
                # Let ratio = u/d = u^2
                # S_j_0 = S_0 * d^j
                # S_j_i = S_j_0 * (u^2)^i
                
                # We want to find the threshold S* such that Intrinsic(S*) = V_cont(S*)
                # Intrinsic(S) = K - S (if S < K, else 0)
                # V_cont(S) is the value we just calculated in V[j, i]
                
                # We have discrete points (S_j_i, V[j, i]).
                # We find the largest S where Intrinsic >= V_cont.
                # Since S increases with i, and Intrinsic decreases, V_cont generally increases (for puts, lower S -> higher value).
                # Wait, for a put:
                # Low S -> High Intrinsic, High Continuation.
                # High S -> Low Intrinsic, Low Continuation.
                # Exercise is optimal when Intrinsic >= Continuation.
                # This happens for low S.
                # So we look for the crossover from "Exercise" to "Hold".
                
                S_vals = np.zeros(j + 1)
                # Calculate S values
                # S_j_i = S_0 * (u ** i) * (d ** (j - i))
                # This might be slow for large j.
                # Use logarithms or recurrence.
                # S_j_0 = S_0 * d**j
                # S_j_i = S_j_{i-1} * (u/d)
                
                S_j_0 = S_0 * (d ** j)
                ratio = u / d
                S_vals[0] = S_j_0
                if j > 0:
                    S_vals[1:] = S_vals[:-1] * ratio
                    
                Intrinsic = np.maximum(K - S_vals, 0.0)
                Cont = V[j, 0:j+1]
                
                # Find where Intrinsic >= Cont
                # We want the max S where this is true.
                # Since S is increasing, Intrinsic is decreasing, Cont is increasing (typically).
                # The condition Intrinsic >= Cont will be True for small i, False for large i.
                
                mask = Intrinsic >= Cont
                if np.any(mask):
                    # Find the last index where mask is True
                    last_true_idx = np.where(mask)[0][-1]
                    # The threshold is between S_vals[last_true_idx] and S_vals[last_true_idx+1]
                    # If last_true_idx is the last node, threshold is S_vals[last_true_idx] (or higher)
                    if last_true_idx == j:
                        star = S_vals[last_true_idx]
                    else:
                        # Interpolate
                        s1 = S_vals[last_true_idx]
                        s2 = S_vals[last_true_idx + 1]
                        v1 = Intrinsic[last_true_idx] - Cont[last_true_idx]
                        v2 = Intrinsic[last_true_idx + 1] - Cont[last_true_idx + 1]
                        
                        if abs(v1 - v2) < 1e-10:
                            star = s1
                        else:
                            frac = v1 / (v1 - v2)
                            star = s1 + frac * (s2 - s1)
                else:
                    # Never exercise
                    star = 0.0
                    
                thresholds[date_idx - 1] = star

    class _Policy:
        __slots__ = ('date_thresholds', 'n_dates', 'T')

        def __init__(self, date_thresholds, n_dates, T):
            self.date_thresholds = date_thresholds
            self.n_dates = n_dates
            self.T = T

        def exercise(self, current_state, time):
            cs = np.asarray(current_state, dtype=float)
            t = time
            if isinstance(t, (list, tuple, np.ndarray)):
                t = float(np.asarray(t, dtype=float).mean())
            else:
                t = float(t)
            # Primary: time is a fraction of the horizon in (0, T].
            # Fallback: if a value > 1 is passed, treat it as a 1..n_dates index.
            if t > 1.0 + 1e-9:
                i = int(round(t))
            else:
                i = int(round(t / self.T * self.n_dates))
            if i < 1:
                i = 1
            elif i > self.n_dates:
                i = self.n_dates
            star = self.date_thresholds[i - 1]
            return cs <= star

    return _Policy(thresholds, n_dates, T)

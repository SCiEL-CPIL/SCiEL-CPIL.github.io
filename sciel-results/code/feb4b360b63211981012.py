import numpy as np
from scipy.optimize import minimize, SLSQP
import time

N = 50

# EVOLVE-BLOCK-START
def _score(x):
    """Compute the ratio of squared L2 norm to (L1 norm * sup norm of autoconvolution)."""
    if len(x) == 0:
        return 0.0
    x = np.asarray(x, dtype=float)
    if np.any(x < -1e-15):
        return 0.0
    x = np.maximum(x, 0.0)
    l2_sq = np.sum(x**2)
    l1 = np.sum(x)
    if l1 < 1e-15:
        return 0.0
    # Autoconvolution: (x * x)[k] = sum_i x[i] * x[k-i] for valid indices
    # Using np.convolve
    ac = np.convolve(x, x, mode='full')
    sup_ac = np.max(ac)
    if sup_ac < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_ac)

def _negative_score(x):
    """Negative score for minimization."""
    return -_score(x)

def construct():
    deadline = time.time() + 140.0  # 140s deadline, 10s margin
    
    best_x = None
    best_score = 0.0
    
    # Initial candidates
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones(N))
    
    # 2. Single spike at center
    x_sp = np.zeros(N)
    x_sp[N//2] = 1.0
    candidates.append(x_sp)
    
    # 3. Single spike at various positions
    for i in range(N):
        x_i = np.zeros(N)
        x_i[i] = 1.0
        candidates.append(x_i)
    
    # 4. Two symmetric spikes around center
    for offset in range(1, N//2 + 1):
        x_2 = np.zeros(N)
        x_2[N//2 - offset] = 1.0
        x_2[N//2 + offset] = 1.0
        candidates.append(x_2)
    
    # 5. Block of consecutive ones
    for length in range(1, N+1):
        for start in range(N - length + 1):
            x_b = np.zeros(N)
            x_b[start:start+length] = 1.0
            candidates.append(x_b)
    
    # 6. Random nonnegative vectors
    rng = np.random.RandomState(42)
    for _ in range(200):
        x_r = rng.random(N)
        candidates.append(x_r)
    
    # 7. Sparse random
    for _ in range(200):
        x_r = np.zeros(N)
        k = rng.randint(1, 10)
        idx = rng.choice(N, size=k, replace=False)
        x_r[idx] = rng.random(k)
        candidates.append(x_r)
    
    # Evaluate all candidates
    for x in candidates:
        if time.time() > deadline - 5.0:
            break
        x = np.asarray(x, dtype=float)
        x = np.maximum(x, 0.0)
        s = _score(x)
        if s > best_score:
            best_score = s
            best_x = x.copy()
    
    # Now do local optimization from the best candidate and some top candidates
    # Sort candidates by score
    scored = []
    for x in candidates:
        x = np.asarray(x, dtype=float)
        x = np.maximum(x, 0.0)
        s = _score(x)
        scored.append((s, x))
    scored.sort(key=lambda t: -t[0])
    
    # Take top 10 for local refinement
    top_candidates = [x for _, x in scored[:10]]
    
    for x0 in top_candidates:
        if time.time() > deadline - 10.0:
            break
        
        x0 = np.asarray(x0, dtype=float)
        x0 = np.maximum(x0, 0.0)
        
        # Try SLSQP with epigraph formulation
        # Maximize l2_sq / (l1 * sup_ac)
        # Equivalent to: maximize l2_sq subject to l1 * sup_ac <= 1, x >= 0
        # Or: minimize -l2_sq subject to l1 * sup_ac <= 1, x >= 0
        
        # Let's use a simpler approach: direct minimization of negative score
        # But the max makes it non-smooth. Let's try SLSQP with constraints.
        
        # Alternative: use the epigraph formulation
        # Maximize t subject to:
        #   t * l1 * sup_ac <= l2_sq  (this is the ratio constraint)
        #   x >= 0
        
        # Actually, let's just do block coordinate descent with the strategy described.
        
        # Two-stage block-coordinate descent:
        # 1. Identify dominant autoconvolution peak index k*
        # 2. Solve 1D subproblem for symmetric pairs around k*
        
        x = x0.copy()
        for iteration in range(50):
            if time.time() > deadline - 10.0:
                break
            
            # Compute autoconvolution
            ac = np.convolve(x, x, mode='full')
            k_star = np.argmax(ac)
            
            # For each symmetric pair (i, k_star - i) where both are in [0, N-1],
            # try to optimize the weights
            improved = False
            for i in range(N):
                j = k_star - i
                if j < 0 or j >= N or j == i:
                    continue
                
                # Current values
                a = x[i]
                b = x[j]
                
                # Try a few candidate values for the pair
                # We want to maximize the contribution to L2 while controlling the peak
                # Simple: try scaling both up or down
                for scale in [0.5, 0.7, 0.8, 0.9, 1.1, 1.3, 1.5, 2.0]:
                    x_trial = x.copy()
                    x_trial[i] = a * scale
                    x_trial[j] = b * scale
                    if np.any(x_trial < -1e-15):
                        continue
                    x_trial = np.maximum(x_trial, 0.0)
                    s_trial = _score(x_trial)
                    if s_trial > _score(x) + 1e-10:
                        x = x_trial
                        improved = True
                        break
                
                # Also try asymmetric updates
                for scale_a in [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]:
                    for scale_b in [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]:
                        x_trial = x.copy()
                        x_trial[i] = a * scale_a
                        x_trial[j] = b * scale_b
                        if np.any(x_trial < -1e-15):
                            continue
                        x_trial = np.maximum(x_trial, 0.0)
                        s_trial = _score(x_trial)
                        s_curr = _score(x)
                        if s_trial > s_curr + 1e-10:
                            x = x_trial
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        s_final = _score(x)
        if s_final > best_score:
            best_score = s_final
            best_x = x.copy()
    
    # Final polish: try SLSQP on the best solution
    if best_x is not None and time.time() < deadline - 15.0:
        x0 = best_x.copy()
        
        def objective(v):
            return -_score(v)
        
        constraints = []
        # No explicit constraints needed since we clip to nonnegative
        bounds = [(0, None)] * N
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                          options={'maxiter': 100, 'ftol': 1e-8})
            if res.success or res.fun < objective(x0):
                x_res = np.maximum(res.x, 0.0)
                s_res = _score(x_res)
                if s_res > best_score:
                    best_score = s_res
                    best_x = x_res.copy()
        except Exception:
            pass
    
    # Return as list of floats
    if best_x is None:
        return [1.0] * N
    return [float(v) for v in best_x]
# EVOLVE-BLOCK-END

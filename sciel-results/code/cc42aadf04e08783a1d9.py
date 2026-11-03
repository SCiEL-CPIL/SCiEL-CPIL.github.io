"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

def _score_sequence(s):
    """Compute the true score: max|autoconv| / (sum s)^2."""
    sq_int = (np.sum(s)) ** 2
    if sq_int < 1e-10:
        return np.inf
    ac = fftconvolve(s, s, mode='full')
    max_ac = np.max(np.abs(ac))
    return max_ac / sq_int

def _soft_max_ac(s, temperature=1.0):
    """
    Differentiable surrogate for max |autoconvolution|.
    Uses log-sum-exp of (ac/temperature)^2 to approximate max ac^2.
    """
    ac = fftconvolve(s, s, mode='full')
    ac_sq = ac ** 2
    max_val = np.max(ac_sq)
    if max_val < 1e-10:
        return 0.0
    lse = max_val + np.log(np.sum(np.exp(ac_sq - max_val)))
    return np.sqrt(lse)

def _soft_objective(x, temperature=1.0):
    """
    Soft objective: use log-sum-exp surrogate for max |ac|.
    """
    s = np.tanh(x)
    sq_int = (np.sum(s)) ** 2
    if sq_int < 1e-10:
        return 1e10
    soft_max_ac = _soft_max_ac(s, temperature)
    return soft_max_ac / sq_int

def _fast_flip_eval(s, idx, ac):
    """
    Compute new autoconvolution max after flipping element idx.
    Only O(N) elements of ac change.
    Returns (new_ac, new_max_abs, new_sum_sq).
    """
    new_s = s.copy()
    new_s[idx] = -new_s[idx]
    
    # Update autoconvolution: ac_new[k] = ac_old[k] + 2*delta*s[j] for affected k
    # Flipping s[idx] changes s[idx] by delta = -2*s[idx]
    delta = -2.0 * s[idx]
    
    # ac[k] = sum_j s[j]*s[k-j]
    # After flip, ac_new[k] = ac[k] + delta*s[k-idx] for k in [idx, 2N-1-idx]
    # But we need to be careful: the full autoconvolution has length 2N-1
    # ac[k] for k=0..2N-2
    # When we flip s[idx], for each j, the term s[j]*s[k-j] changes if j==idx or k-j==idx
    # So ac_new[k] = ac[k] + delta*s[k-idx] + delta*s[idx] - delta*s[k-idx]*s[idx]... 
    # Actually simpler: ac_new = fftconvolve(new_s, new_s)
    # But that's O(N log N). Let's do the O(N) update properly.
    
    # ac_new[k] = sum_j new_s[j] * new_s[k-j]
    # new_s[j] = s[j] + delta*delta_j where delta_j = 1 if j==idx else 0
    # = sum_j (s[j] + delta*delta_j)(s[k-j] + delta*delta_{k-j})
    # = ac[k] + delta*sum_j delta_j*s[k-j] + delta*sum_j s[j]*delta_{k-j} + delta^2*sum_j delta_j*delta_{k-j}
    # = ac[k] + delta*s[k-idx] + delta*s[idx] + delta^2*(1 if k==2*idx else 0)
    
    new_ac = ac.copy()
    # For k in [idx, 2N-1-idx], k-idx is in [0, N-1]
    k_min = idx
    k_max = 2 * N - 1 - idx
    for k in range(k_min, k_max + 1):
        j = k - idx
        if 0 <= j < N:
            new_ac[k] += delta * s[j] + delta * s[idx]
            if k == 2 * idx:
                new_ac[k] += delta * delta  # delta^2 term
    # Also handle the case where k-idx is out of bounds but we already bounded k
    
    # Actually, let me just recompute properly using the formula:
    # new_ac[k] = ac[k] + delta*(s[k-idx] if 0<=k-idx<N else 0) + delta*(s[idx] if 0<=k-idx<N else 0)
    # Wait, that's not right either. Let me think again.
    
    # The cleanest way: for each k, new_ac[k] = old_ac[k] + delta * (s[k-idx] + s[idx]) when valid indices
    # But this double-counts. Let me just use the direct O(N) update:
    # 
    # new_ac[k] = sum_{j} new_s[j] * new_s[k-j]
    # where new_s differs from s only at index idx by delta
    # = sum_j s[j]*s[k-j] + delta * s[k-idx] + delta * s[idx] + delta^2 * [k==2idx]
    #   (the middle terms: one from j=idx, one from k-j=idx, but if both, it's delta^2)
    # Actually: sum_j (s[j]+delta*1_{j=idx})(s[k-j]+delta*1_{k-j=idx})
    # = sum_j s[j]s[k-j] + delta*sum_j 1_{j=idx}s[k-j] + delta*sum_j s[j]1_{k-j=idx} + delta^2*sum_j 1_{j=idx}1_{k-j=idx}
    # = ac[k] + delta*s[k-idx] + delta*s[idx] + delta^2*[k==2idx]
    # BUT: if k-idx is out of [0,N-1], s[k-idx] is 0. Similarly s[idx] is always valid.
    # Wait, s[idx] is a scalar, always valid. But s[k-idx] is only valid if 0 <= k-idx < N.
    # And the delta*s[idx] term: sum_j s[j]*1_{k-j=idx} = s[k-idx] if 0<=k-idx<N, else 0.
    # So actually both middle terms are delta*s[k-idx] (when valid)!
    # No wait: delta*sum_j 1_{j=idx}s[k-j] = delta*s[k-idx] if valid
    #          delta*sum_j s[j]1_{k-j=idx} = delta*s[idx] if k-idx in [0,N-1], i.e., same condition
    # So both give delta*s[k-idx] when k-idx is valid, and 0 otherwise.
    # Hmm, that's not right. Let me re-derive.
    #
    # Term A: delta * sum_j 1_{j=idx} * s[k-j] = delta * s[k-idx] if 0<=k-idx<N
    # Term B: delta * sum_j s[j] * 1_{k-j=idx} = delta * s[k-idx] if 0<=k-idx<N  [since j=k-idx must be in [0,N-1]]
    # Term C: delta^2 * sum_j 1_{j=idx} * 1_{k-j=idx} = delta^2 if k=2idx and idx in [0,N-1]
    #
    # So new_ac[k] = ac[k] + 2*delta*s[k-idx] + delta^2*[k==2idx] for k where 0<=k-idx<N
    # And new_ac[k] = ac[k] for other k.
    
    # Redo with correct formula
    new_ac = ac.copy()
    delta = -2.0 * s[idx]
    for k in range(max(0, idx), min(2*N-1, idx+N)):
        j = k - idx
        if 0 <= j < N:
            new_ac[k] += 2 * delta * s[j]
            if k == 2 * idx:
                new_ac[k] += delta * delta
    
    new_max_abs = np.max(np.abs(new_ac))
    new_sum = np.sum(new_s)
    new_sum_sq = new_sum ** 2
    
    return new_ac, new_max_abs, new_sum_sq

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin
    
    best_score = np.inf
    best_s = None
    
    np.random.seed(42)
    
    # Phase 1: Gradient-based optimization with adaptive temperature
    for init_iter in range(15):
        if time.time() > deadline:
            break
        
        # Diverse initializations
        if init_iter < 5:
            x0 = np.random.randn(N) * 0.1
        elif init_iter < 10:
            # Chirp-like initialization
            t = np.arange(N)
            x0 = 0.5 * np.sin(2 * np.pi * t * np.random.uniform(0.1, 0.5)) + np.random.randn(N) * 0.05
        else:
            x0 = np.random.randn(N) * 0.3
        
        # Adaptive temperature: start high, decrease
        temp = 2.0 * (0.5 ** (init_iter / 10.0))
        
        try:
            result = minimize(
                _soft_objective,
                x0,
                method='L-BFGS-B',
                args=(temp,),
                options={'maxiter': 80, 'ftol': 1e-9, 'gtol': 1e-7}
            )
            
            s_continuous = np.tanh(result.x)
            s_binary = np.sign(s_continuous)
            s_binary[s_binary == 0] = 1.0
            
            score_bin = _score_sequence(s_binary)
            if score_bin < best_score:
                best_score = score_bin
                best_s = s_binary.copy()
            
            score_cont = _score_sequence(s_continuous)
            if score_cont < best_score:
                best_score = score_cont
                best_s = s_continuous.copy()
                
        except Exception:
            continue
    
    # Phase 2: Fast local search with O(N) flip evaluation
    if best_s is not None:
        current_s = best_s.copy()
        current_score = best_score
        current_ac = fftconvolve(current_s, current_s, mode='full')
        
        # First: greedy local search - try all single flips, keep best
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_flip_score = current_score
            best_flip_idx = -1
            best_flip_ac = None
            
            for idx in range(N):
                new_ac, new_max_abs, new_sum_sq = _fast_flip_eval(current_s, idx, current_ac)
                if new_sum_sq < 1e-10:
                    continue
                new_score = new_max_abs / new_sum_sq
                if new_score < best_flip_score - 1e-12:
                    best_flip_score = new_score
                    best_flip_idx = idx
                    best_flip_ac = new_ac
            
            if best_flip_idx >= 0:
                current_s[best_flip_idx] = -current_s[best_flip_idx]
                current_ac = best_flip_ac
                current_score = best_flip_score
                if current_score < best_score:
                    best_score = current_score
                    best_s = current_s.copy()
                improved = True
        
        # Second: simulated annealing with fast evaluation
        sa_iters = 0
        max_sa = 300
        T_start = 0.05
        for sa_iter in range(max_sa):
            if time.time() > deadline:
                break
            sa_iters += 1
            
            idx = np.random.randint(N)
            new_ac, new_max_abs, new_sum_sq = _fast_flip_eval(current_s, idx, current_ac)
            if new_sum_sq < 1e-10:
                continue
            new_score = new_max_abs / new_sum_sq
            
            delta = new_score - current_score
            T = T_start * (0.95 ** sa_iter)
            if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-6)):
                current_s[idx] = -current_s[idx]
                current_ac = new_ac
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_s = current_s.copy()
    
    # Phase 3: Final polish with tighter gradient optimization
    if best_s is not None and time.time() < deadline - 10:
        # Use best solution as starting point for a final tight optimization
        x_start = np.arctanh(np.clip(best_s, -0.999, 0.999))
        try:
            result = minimize(
                _soft_objective,
                x_start,
                method='L-BFGS-B',
                args=(0.5,),
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            s_final = np.tanh(result.x)
            s_final_bin = np.sign(s_final)
            s_final_bin[s_final_bin == 0] = 1.0
            
            score_f = _score_sequence(s_final)
            if score_f < best_score:
                best_score = score_f
                best_s = s_final.copy()
            
            score_fb = _score_sequence(s_final_bin)
            if score_fb < best_score:
                best_score = score_fb
                best_s = s_final_bin.copy()
        except Exception:
            pass
    
    if best_s is None:
        return np.ones(N)
    
    return best_s
# EVOLVE-BLOCK-END

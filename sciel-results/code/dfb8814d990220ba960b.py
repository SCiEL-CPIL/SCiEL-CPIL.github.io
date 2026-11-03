import numpy as np
import time
import random
from scipy.optimize import minimize

N = 50

def _score_raw(h):
    """Compute the raw score: sum(h^2) / (sum(h) * max(autoconv(h)))"""
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-15):
        h = np.clip(h, 0, None)
    s = np.sum(h)
    if s < 1e-15:
        return 0.0
    autoconv = np.convolve(h, h, mode='full')
    max_ac = np.max(autoconv)
    if max_ac < 1e-15:
        return 0.0
    return np.sum(h**2) / (s * max_ac)

def _log_sum_exp(x, temperature=1.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    if x_max < 0:
        x_max = 0
    return x_max + np.log(np.sum(np.exp((x - x_max) / temperature))) * temperature

def _smoothed_ratio_obj(h, temperature=1.0):
    """
    Negative of the smoothed ratio objective.
    We maximize sum(h^2) / (sum(h) * smoothed_max_autoconv)
    which is equivalent to minimizing -log(sum(h^2)) + log(sum(h)) + log_smoothed_max_autoconv
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-10):
        h = np.clip(h, 0, None)
    s = np.sum(h)
    if s < 1e-15:
        return 1e6
    sq_sum = np.sum(h**2)
    if sq_sum < 1e-15:
        return 1e6
    autoconv = np.convolve(h, h, mode='full')
    # Use log-sum-exp as a smooth approximation of the max
    # We want to approximate max(autoconv)
    # log_sum_exp gives us log(sum(exp(x/T)))*T which approximates max for small T
    # But we need the actual max value, not log of it.
    # Let's use a different approach: the ratio is sum(h^2)/(sum(h)*max_ac)
    # We can minimize -log(ratio) = -log(sum(h^2)) + log(sum(h)) + log(max_ac)
    # But log(max_ac) is not differentiable.
    # Instead, let's just minimize the negative of the smoothed ratio directly.
    # Smoothed max: use the fact that for convex f, max is approximated by log-sum-exp.
    # Actually, let's use a penalty method or just use the log-sum-exp approximation of max.
    
    # Better approach: minimize -log(sum(h^2)) + log(sum(h)) + (1/T)*log_sum_exp(T*autoconv)
    # This approximates -log(ratio) when T is small.
    
    # For simplicity, let's just use the raw ratio with a smooth max approximation.
    # The issue is that max is not differentiable. Let's use the log-sum-exp trick.
    
    # Actually, let's try a different formulation.
    # We want to maximize f(h) = sum(h^2) / (sum(h) * max_ac)
    # Let g(h) = -log(f(h)) = -log(sum(h^2)) + log(sum(h)) + log(max_ac)
    # The term log(max_ac) is the problem.
    
    # Alternative: use the fact that max_ac = max_k sum_i h_i h_{i+k-N+1} (with appropriate indexing)
    # This is a max of quadratic forms.
    
    # Let's just use L-BFGS-B on the smoothed objective.
    # Smoothed max: M_smooth = log(sum(exp(ac/T)))*T
    # Then ratio_smooth = sum(h^2)/(sum(h)*M_smooth)
    # We minimize -log(ratio_smooth)
    
    ac = autoconv
    M_smooth = _log_sum_exp(ac, temperature)
    if M_smooth < 1e-15:
        return 1e6
    
    ratio_smooth = sq_sum / (s * M_smooth)
    if ratio_smooth < 1e-15:
        return 1e6
    
    return -np.log(ratio_smooth)

def _smoothed_ratio_grad(h, temperature=1.0):
    """Gradient of the smoothed ratio objective."""
    h = np.asarray(h, dtype=float).copy()
    if np.any(h < 0):
        h = np.clip(h, 0, None)
    s = np.sum(h)
    if s < 1e-15:
        return np.zeros_like(h)
    sq_sum = np.sum(h**2)
    if sq_sum < 1e-15:
        return np.zeros_like(h)
    
    autoconv = np.convolve(h, h, mode='full')
    ac = autoconv
    
    # M_smooth = log(sum(exp(ac/T)))*T
    ac_max = np.max(ac)
    if ac_max < 0:
        ac_max = 0
    exp_terms = np.exp((ac - ac_max) / temperature)
    sum_exp = np.sum(exp_terms)
    M_smooth = ac_max + temperature * np.log(sum_exp)
    
    if M_smooth < 1e-15:
        return np.zeros_like(h)
    
    # ratio = sq_sum / (s * M_smooth)
    # We minimize -log(ratio) = -log(sq_sum) + log(s) + log(M_smooth)
    
    # Gradient of -log(sq_sum): -2*h/sq_sum
    grad_sq = -2.0 * h / sq_sum
    
    # Gradient of log(s): 1/s
    grad_s = 1.0 / s * np.ones_like(h)
    
    # Gradient of log(M_smooth):
    # M_smooth = ac_max + T*log(sum(exp((ac-ac_max)/T)))
    # dM_smooth/dac_j = (1/T) * exp((ac_j - ac_max)/T) / sum(exp((ac-ac_max)/T)) * T = exp((ac_j-ac_max)/T)/sum_exp
    # So dM_smooth/dac_j = p_j where p_j = exp((ac_j-ac_max)/T)/sum_exp
    # dac_j/dh_i = sum_k (delta_{i,k} h_{j-i} + delta_{j-i,k} h_{i}) ... this is getting complex.
    
    # Let's use numerical gradient for simplicity given time constraints.
    # Actually, let's just use a simple approach: minimize the negative ratio directly with BFGS.
    
    # For now, return a simple gradient approximation.
    # The gradient of log(M_smooth) with respect to h is complex.
    # Let's use finite differences for the gradient if needed, but that's slow.
    
    # Alternative: just use the raw objective with a smoothed max and let L-BFGS-B handle it.
    # But L-BFGS-B needs gradients.
    
    # Let's compute the gradient numerically.
    eps = 1e-7
    grad = np.zeros_like(h)
    for i in range(len(h)):
        h_plus = h.copy()
        h_plus[i] += eps
        grad[i] = (_smoothed_ratio_obj(h_plus, temperature) - _smoothed_ratio_obj(h, temperature)) / eps
    
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Start with uniform distribution
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    # Try to find a good initial point using Sidon-like sets
    # A Sidon set has all pairwise sums distinct, which minimizes autoconvolution peaks.
    
    def find_sidon_set(N, K, attempts=200):
        for _ in range(attempts):
            candidates = list(range(N))
            random.shuffle(candidates)
            S = []
            sums = set()
            for cand in candidates:
                if len(S) == K:
                    break
                conflict = False
                new_sums = set()
                if cand + cand in sums:
                    conflict = True
                else:
                    new_sums.add(cand + cand)
                if not conflict:
                    for s in S:
                        sm = cand + s
                        if sm in sums or sm in new_sums:
                            conflict = True
                            break
                        new_sums.add(sm)
                if not conflict:
                    S.append(cand)
                    sums.update(new_sums)
            if len(S) == K:
                return S
        return None
    
    # Try different initial points
    initial_points = [best_h]
    
    # Try Sidon sets of various sizes
    for K in range(10, 4, -1):
        if time.time() > deadline - 60.0:
            break
        S = find_sidon_set(N, K, attempts=100)
        if S is not None:
            h_init = np.zeros(N)
            h_init[S] = 1.0 / K
            initial_points.append(h_init)
    
    # Also try some random initial points
    for _ in range(5):
        if time.time() > deadline - 60.0:
            break
        h_rand = np.random.exponential(1.0, size=N)
        h_rand /= np.sum(h_rand)
        initial_points.append(h_rand)
    
    # Optimize from each initial point using L-BFGS-B
    temperatures = [0.5, 1.0, 2.0]
    
    for h_init in initial_points:
        if time.time() > deadline - 30.0:
            break
        
        for temp in temperatures:
            if time.time() > deadline - 10.0:
                break
            
            try:
                result = minimize(
                    _smoothed_ratio_obj,
                    h_init,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                h_opt = result.x
                h_opt = np.clip(h_opt, 0, None)
                score = _score_raw(h_opt)
                
                if score > best_score:
                    best_score = score
                    best_h = h_opt.copy()
            except Exception as e:
                pass
    
    # Final refinement: try to improve the best solution found
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                _smoothed_ratio_obj,
                best_h,
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_opt = result.x
            h_opt = np.clip(h_opt, 0, None)
            score = _score_raw(h_opt)
            if score > best_score:
                best_score = score
                best_h = h_opt.copy()
        except Exception as e:
            pass
    
    return list(best_h)

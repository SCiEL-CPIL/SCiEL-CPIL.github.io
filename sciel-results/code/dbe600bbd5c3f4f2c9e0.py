import numpy as np
import time
import sys
from scipy.optimize import minimize

# ---- Task constants (interface) ----
N = 50
DELTA = 0.5 / N
_XC = -0.25 + (np.arange(N) + 0.5) * DELTA

_DEADLINE = 145.0  # Safety margin for 150s limit
_START = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _ratio(h):
    """Raw objective: L2^2 / (L1 * max_autoconv). Returns -inf/0 if degenerate."""
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < 0):
        return -np.inf
    l1 = np.sum(h)
    if l1 <= 0.0:
        return 0.0
    l2_sq = np.dot(h, h)
    if l2_sq <= 0.0:
        return 0.0
    conv = np.convolve(h, h, mode="full")
    m = np.max(conv)
    if m <= 0.0:
        return 0.0
    return l2_sq / (l1 * m)


def _smooth_log_ratio_and_grad(h, T=1.0, lam=1.0, eps=1e-9):
    """
    Compute the smooth objective and gradient for L-BFGS-B.
    Objective: -log(L2^2) + log(L1) + log(softmax(conv, T)) + lam * sum(log(h_i + eps))
    
    We minimize this. The log-barrier term enforces h_i > 0.
    The softmax term smooths the max of the autoconvolution.
    """
    h = np.asarray(h, dtype=np.float64)
    
    # Safety clip for numerical stability in logs
    h_safe = np.clip(h, 1e-15, None)
    
    l1 = np.sum(h_safe)
    l2_sq = np.dot(h_safe, h_safe)
    
    if l1 <= 0 or l2_sq <= 0:
        return 1e10, np.zeros(N)
    
    # Compute autoconvolution
    conv = np.convolve(h_safe, h_safe, mode="full")
    
    # Softmax approximation of max
    # To avoid overflow, subtract max
    c_max = np.max(conv)
    if c_max <= 0:
        return 1e10, np.zeros(N)
    
    # log(sum(exp(conv/T))) = log(sum(exp((conv - c_max)/T))) + c_max/T
    # This is log(softmax_sum) + c_max/T
    # We want log(max) approx.
    # log(softmax_sum) = log(sum(exp((conv-c_max)/T)))
    # As T -> 0, this approaches 0. So log(max) ~ c_max/T + log(sum(...))
    # Wait, the term in the objective is log(max_conv).
    # Using log-sum-exp: LSE(x/T) = log(sum(exp(x/T))) = c_max/T + log(sum(exp((x-c_max)/T)))
    # So log(max) is approximated by LSE(conv/T) - log(N_conv) ? No.
    # Actually, min-max inequality: max(x) <= LSE(x/T) <= max(x) + T*log(N).
    # We want to minimize log(max). Since LSE is an upper bound on log(max) (scaled),
    # minimizing LSE will push the distribution to be "peaky" in a way that reduces the effective max.
    # Let's use: obj_term = LSE(conv/T) / T ? No.
    # Let's stick to the gradient of log(max) approximated by softmax.
    # grad log(sum(exp(c/T))) w.r.t h_i = (1/T) * sum_j (p_j * 2*h_{c_j-i})
    # where p_j = exp(c_j/T) / sum(exp(c_k/T))
    
    scaled_conv = conv / T
    scaled_conv -= np.max(scaled_conv) # for stability
    exp_conv = np.exp(scaled_conv)
    sum_exp = np.sum(exp_conv)
    
    if sum_exp <= 0:
        return 1e10, np.zeros(N)
    
    p = exp_conv / sum_exp
    
    # Value: -log(l2_sq) + log(l1) + (1/T) * log(sum_exp) + (c_max/T) + lam * sum(log(h+eps))
    # Note: log(sum(exp(conv/T))) = log(sum_exp) + max(conv)/T
    lse_val = np.log(sum_exp) + c_max / T
    
    val = -np.log(l2_sq) + np.log(l1) + lse_val + lam * np.sum(np.log(h_safe + eps))
    
    # Gradient
    grad = np.zeros(N)
    
    # d/dh_i [-log(l2_sq)] = -2*h_i / l2_sq
    grad -= 2.0 * h_safe / l2_sq
    
    # d/dh_i [log(l1)] = 1 / l1
    grad += 1.0 / l1
    
    # d/dh_i [lse_val] = (1/T) * sum_k p_k * d(conv_k)/dh_i
    # conv_k = sum_j h_j h_{k-j}
    # d(conv_k)/dh_i = h_{k-i} + h_{i-k} (if indices valid)
    # Since conv is symmetric, d(conv_k)/dh_i = 2 * h_{k-i} if k-i in [0, N-1] else 0?
    # Actually, conv_k = sum_{j} h_j h_{k-j}.
    # Term is h_i h_{k-i} + h_{k-i} h_i = 2 h_i h_{k-i} if k!=i?
    # If k=i, term is h_i^2, derivative 2 h_i.
    # Generally, d(conv_k)/dh_i = 2 * h_{k-i} if 0 <= k-i < N.
    # Let's compute efficiently.
    # grad_lse[i] = (1/T) * sum_k p_k * 2 * h_{k-i} (if valid)
    
    # We can compute this via convolution of p with h?
    # Let q_k = 2 * p_k.
    # grad_lse = convolve(p, h, mode='full')? No.
    # sum_k p_k h_{k-i}. This is (p * h)[i] in correlation sense?
    # Let's just loop or use fft. N=50 is small, loop is fine, but let's use np.convolve logic.
    # sum_k p_k h_{k-i} is the cross-correlation of p and h.
    # np.correlate(p, h, mode='same')?
    # p has length 2N-1. h has length N.
    # We need sum_{k=0}^{2N-2} p_k h_{k-i} where 0 <= k-i < N.
    
    # Efficient way:
    # grad_lse = np.zeros(N)
    # for i in range(N):
    #    # k-i in [0, N-1] => k in [i, i+N-1]
    #    # p indices: i to i+N-1
    #    # h indices: 0 to N-1
    #    # sum p[i+j] * h[j] for j=0..N-1
    #    grad_lse[i] = np.dot(p[i:i+N], h_safe)
    
    # This is O(N^2) which is 2500 ops. Very fast.
    grad_lse = np.zeros(N)
    for i in range(N):
        grad_lse[i] = np.dot(p[i:i+N], h_safe)
    
    grad += (2.0 / T) * grad_lse
    
    # d/dh_i [lam * sum(log(h_j + eps))] = lam / (h_i + eps)
    grad += lam / (h_safe + eps)
    
    return val, grad


def _polish_lbfgsb(h_init, T_start=1.0, T_end=0.1, lam_start=1.0, lam_end=0.1, max_iter=1000):
    """
    Refine h using L-BFGS-B with annealing of T and lam.
    """
    h = np.asarray(h_init, dtype=np.float64)
    if np.sum(h) <= 0:
        return h
    
    # Normalize to sum=1 initially to stabilize
    h = h / np.sum(h)
    
    # We will minimize the smooth objective.
    # L-BFGS-B handles bounds, but we use log barrier for soft positivity.
    # We can set bounds to [0, None] to be safe, but log barrier is stronger.
    
    best_h = h.copy()
    best_val = _ratio(h)
    
    # Annealing schedule
    num_steps = 5
    Ts = np.linspace(T_start, T_end, num_steps)
    lams = np.linspace(lam_start, lam_end, num_steps)
    
    current_h = h.copy()
    
    for step in range(num_steps):
        if _time_left() < 5:
            break
            
        T = Ts[step]
        lam = lams[step]
        
        def obj_fun(x):
            val, _ = _smooth_log_ratio_and_grad(x, T=T, lam=lam)
            return val
        
        def grad_fun(x):
            _, g = _smooth_log_ratio_and_grad(x, T=T, lam=lam)
            return g
        
        try:
            res = minimize(
                obj_fun, 
                current_h, 
                jac=grad_fun, 
                method='L-BFGS-B', 
                bounds=[(0, None)]*N, # Hard bound to prevent negative
                options={'maxiter': max_iter//num_steps, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.x is not None:
                hx = np.clip(res.x, 0, None)
                if np.sum(hx) > 1e-10:
                    hx = hx / np.sum(hx)
                    r = _ratio(hx)
                    if r > best_val:
                        best_val = r
                        best_h = hx.copy()
                    current_h = hx
        except Exception:
            pass
            
    return best_h


def _generate_inits():
    """Generate diverse initializations."""
    inits = []
    rng = np.random.RandomState(42)
    
    # 1. Uniform
    inits.append(np.ones(N) / N)
    
    # 2. Random sparse (Sidon-like attempts)
    for _ in range(20):
        k = rng.randint(4, 12)
        idx = rng.choice(N, k, replace=False)
        h = np.zeros(N)
        h[idx] = 1.0
        h /= np.sum(h)
        inits.append(h)
        
    # 3. Random smooth
    for _ in range(20):
        h = rng.rand(N)
        # Smooth
        h = np.convolve(h, np.ones(5)/5, mode='same')
        h = np.clip(h, 0, None)
        if np.sum(h) > 0:
            h /= np.sum(h)
            inits.append(h)
            
    # 4. Peaky
    for _ in range(10):
        h = np.zeros(N)
        center = rng.randint(5, N-5)
        width = rng.uniform(1, 5)
        h = np.exp(-0.5 * ((np.arange(N) - center) / width)**2)
        h /= np.sum(h)
        inits.append(h)
        
    return inits


def construct():
    np.random.seed(12345)
    best_h = np.ones(N) / N
    best_r = _ratio(best_h)

    def upd(h):
        nonlocal best_h, best_r
        h = np.asarray(h, dtype=np.float64)
        if np.sum(h) <= 0 or not np.all(np.isfinite(h)):
            return
        hh = np.clip(h, 0.0, None)
        r = _ratio(hh)
        if np.isfinite(r) and r > best_r:
            best_r = r
            best_h = hh.copy()

    inits = _generate_inits()
    
    # Process inits with L-BFGS-B polish
    for h_init in inits:
        if _time_left() < 15:
            break
            
        h_opt = _polish_lbfgsb(h_init)
        upd(h_opt)
        
        # If we found a significantly better solution, use it as a new init
        if best_r > _ratio(np.ones(N)/N) * 1.05:
            inits.append(best_h.copy())
            inits.append(best_h.copy()) # Duplicate to exploit basin
            
    # Final tight polish on the best found
    if _time_left() > 10:
        # Use very low T and lam for final squeeze
        h_final = _polish_lbfgsb(best_h, T_start=0.5, T_end=0.05, lam_start=0.5, lam_end=0.05, max_iter=2000)
        upd(h_final)
        
    return list(best_h)

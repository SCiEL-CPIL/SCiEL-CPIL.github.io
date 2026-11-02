import numpy as np
import time

N = 50

def _compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * sup|h*h|)"""
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15 or l2_sq < 1e-15:
        return 0.0
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def _soft_sup_conv(h, beta):
    """Smooth approximation of sup autoconvolution via log-sum-exp."""
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    if m < 1e-15:
        return 1e-15
    shifted = conv - m
    return m * np.exp(np.logaddexp.reduce(shifted / beta) * beta) / (len(conv) ** 0) if beta > 0 else m

def _soft_sup_and_grad(h, beta):
    """Return (soft_sup, grad wrt h) using log-sum-exp trick."""
    conv = np.convolve(h, h, mode='full')
    m = np.max(conv)
    if m < 1e-15:
        return 1e-15, np.zeros_like(h)
    
    # Log-sum-exp for stability
    shifted = conv - m
    log_sum = np.log(np.sum(np.exp(shifted)))
    # soft_sup = exp(m + log_sum - log(N_conv)) but we want weighted sum
    # Actually: soft_sup = sum_j exp(conv_j / beta) / beta * ... 
    # Let's use: S = sum_j exp(conv_j / beta), then dS/dh_i = sum_j exp(conv_j/beta) * (1/beta) * h_{k*-i}
    # soft_sup = S / sum(exp) * m? No.
    # Standard softmax: p_j = exp(conv_j/beta) / sum(exp(conv_k/beta))
    # soft_sup = sum_j p_j * conv_j
    # d(soft_sup)/dh_i = sum_j p_j * h_{k*-i} / ... 
    
    # Simpler: use the exponential weighting directly
    # F(h) = (1/beta) * log(sum_j exp(conv_j / beta))
    # dF/dh_i = (1/beta) * sum_j exp(conv_j/beta) * h_{k*-i} / sum_k exp(conv_k/beta)
    # This is the expected value of conv gradient weighted by softmax probabilities
    
    exp_conv = np.exp(conv / beta - m / beta)  # stable
    weights = exp_conv / np.sum(exp_conv)
    soft_sup_val = np.sum(weights * conv)
    
    # Gradient: d(soft_sup_val)/dh_i = sum_j weights[j] * h[k*-i] where k* corresponds to j
    grad = np.zeros_like(h)
    for i in range(N):
        # For each j in conv, contribution is weights[j] * h[j-i] if valid
        # j-i must be in [0, N-1]
        # j ranges from 0 to 2N-2
        # For fixed i, j-i in [0,N-1] means j in [i, i+N-1]
        j_start = max(0, i)
        j_end = min(2 * N - 2, i + N - 1)
        if j_start <= j_end:
            grad[i] = np.sum(weights[j_start:j_end+1] * h[j_start - i : j_end - i + 1])
    
    return soft_sup_val, grad

def _gradient_log_ratio_smooth(h, beta):
    """Gradient of log(||h||_2^2) - log(||h||_1) - log(soft_sup_conv)."""
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    
    if l1 < 1e-15 or l2_sq < 1e-15:
        return np.zeros_like(h)
        
    soft_sup, grad_conv = _soft_sup_and_grad(h, beta)
    if soft_sup < 1e-15:
        return np.zeros_like(h)
        
    grad_l2 = 2.0 * h / l2_sq
    grad_l1 = np.zeros_like(h)
    mask_pos = h > 1e-12
    grad_l1[mask_pos] = 1.0 / l1
    grad_sup = -grad_conv / soft_sup
    
    return grad_l2 - grad_l1 + grad_sup

def _optimize_from(h0, deadline, beta_start=5.0, beta_end=0.5):
    """Gradient ascent on log ratio with annealed beta."""
    h = h0.copy()
    best_h = h.copy()
    best_ratio = _compute_ratio(h)
    n_steps = 2000
    
    for step in range(n_steps):
        if time.time() > deadline - 5.0:
            break
        
        # Anneal beta
        t = step / max(n_steps - 1, 1)
        beta = beta_start * (beta_end / beta_start) ** t
        
        grad = _gradient_log_ratio_smooth(h, beta)
        
        # Adaptive step size with line search
        lr = 0.1 / (1.0 + 0.01 * step)
        for trial in range(10):
            h_new = h + lr * grad
            h_new[h_new < 0] = 0
            l1_new = np.sum(h_new)
            if l1_new < 1e-15:
                break
            h_new = h_new / l1_new
            
            ratio_new = _compute_ratio(h_new)
            if ratio_new > best_ratio:
                best_ratio = ratio_new
                best_h = h_new.copy()
            if ratio_new > _compute_ratio(h) * 0.999:
                h = h_new
                break
            lr *= 0.5
        else:
            h = h + lr * grad
            h[h < 0] = 0
            l1_new = np.sum(h)
            if l1_new > 1e-15:
                h = h / l1_new
    
    return best_h, best_ratio

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_h = np.zeros(N)
    best_ratio = 0.0
    
    # Known good starting points
    sidon_S = [0, 1, 4, 10, 18, 23, 30]
    h_sidon = np.zeros(N)
    for idx in sidon_S:
        h_sidon[idx] = 1.0 / 7.0
    r = _compute_ratio(h_sidon)
    if r > best_ratio:
        best_ratio = r
        best_h = h_sidon.copy()
    
    # Also try uniform
    h_uni = np.ones(N) / N
    r = _compute_ratio(h_uni)
    if r > best_ratio:
        best_ratio = r
        best_h = h_uni.copy()
    
    # Gradient optimization from multiple starts
    starts = [h_sidon.copy(), h_uni.copy()]
    
    # Random sparse starts
    np.random.seed(42)
    for _ in range(5):
        k = np.random.randint(3, 12)
        idxs = np.random.choice(N, k, replace=False)
        h_r = np.zeros(N)
        h_r[idxs] = 1.0 / k
        h_r += np.random.rand(N) * 1e-4
        h_r[h_r < 0] = 0
        starts.append(h_r)
    
    for h_start in starts:
        if time.time() > deadline - 10.0:
            break
        
        h_opt, r_opt = _optimize_from(h_start, deadline)
        if r_opt > best_ratio:
            best_ratio = r_opt
            best_h = h_opt.copy()
    
    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(_compute_ratio(h))

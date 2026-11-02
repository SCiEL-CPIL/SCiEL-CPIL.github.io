import time
import numpy as np
from scipy.fft import fft, ifft

N = 200
DEADLINE_S = 145.0

def _time_remaining():
    return DEADLINE_S - time.time()

def _max_overlap(h):
    """Calculate the maximum circular overlap O(s) = max_s sum_i h_i (1-h)_{i-s}."""
    H = fft(h)
    power = np.abs(H) ** 2
    corr = np.real(ifft(power))
    # O(s) = N/2 - corr[s]
    return float(np.max(N / 2.0 - corr))

def _compute_all_corrs_fft(h):
    """Compute all circular autocorrelations efficiently using FFT."""
    H = fft(h)
    power = np.abs(H) ** 2
    corr = np.real(ifft(power))
    return corr

def _objective(h):
    """The objective is the max overlap, which we want to minimize."""
    return _max_overlap(h)

def _soft_min_obj(h, beta=100.0):
    """
    Soft approximation of the max-overlap objective using the log-sum-exp trick.
    O(s) = N/2 - corr[s]. We want to minimize max(O(s)) = N/2 - min(corr(s)).
    Minimizing max(O) is equivalent to maximizing min(corr).
    So we want to maximize a soft-min of corr.
    soft_max(corr) = (1/beta) * log(sum(exp(beta * corr)))
    We want to maximize this. So the objective to minimize is -soft_max(corr).
    """
    corr = _compute_all_corrs_fft(h)
    max_corr = np.max(corr)
    # Shift for numerical stability
    shifted = beta * (corr - max_corr)
    log_sum_exp = max_corr * beta + np.log(np.sum(np.exp(shifted)))
    soft_max = log_sum_exp / beta
    return -soft_max

def _soft_min_grad(h, beta=100.0):
    """
    Gradient of the soft-max objective (to be minimized).
    f(h) = - (1/beta) * log(sum(exp(beta * corr_s)))
    df/dh_i = - (1/beta) * (sum_s beta * exp(beta * corr_s) * dcorr_s/dh_i) / (sum_s exp(beta * corr_s))
    = - sum_s exp(beta * corr_s) * dcorr_s/dh_i / sum_s exp(beta * corr_s)
    
    corr_s = sum_i h_i h_{i-s}
    dcorr_s/dh_i = h_{i-s}
    
    So grad_i = - sum_s exp(beta * corr_s) * h_{i-s} / Z
    where Z = sum_s exp(beta * corr_s)
    
    Let w_s = exp(beta * (corr_s - max_corr)) / Z_normalized
    Actually, let p_s = exp(beta * corr_s) / sum(exp(beta * corr_s))
    Then grad_i = - sum_s p_s * h_{i-s}
    
    This is - (circular cross-correlation of p with h) at index i.
    Let's compute it using FFT.
    P = fft(p)
    H = fft(h)
    cross_corr = real(ifft(P * conj(H)))
    grad = -cross_corr
    """
    corr = _compute_all_corrs_fft(h)
    max_corr = np.max(corr)
    shifted = beta * (corr - max_corr)
    p = np.exp(shifted)
    p_sum = np.sum(p)
    p_norm = p / p_sum
    
    P = fft(p_norm)
    H = fft(h)
    # Circular cross-correlation of p_norm and h
    # c_k = sum_i p_norm_i h_{i-k}
    # DFT of c is P * conj(H)
    cross_corr = np.real(ifft(P * np.conj(H)))
    grad = -cross_corr
    
    return grad

def _project_to_binary(h):
    """
    Project a real vector h (length N, sum N/2) to a binary vector {0,1} with sum N/2.
    Simplest method: threshold at 0.5 and adjust to ensure exactly N/2 ones.
    """
    h = h.copy()
    # Threshold
    b = (h >= 0.5).astype(np.float64)
    num_ones = np.sum(b)
    target = N / 2.0
    
    if num_ones > target:
        # Need to turn some 1s to 0s. Pick the smallest h values among 1s.
        ones_indices = np.where(b == 1.0)[0]
        num_to_flip = int(num_ones - target)
        if num_to_flip > 0:
            # Sort by h value, pick the smallest
            order = np.argsort(h[ones_indices])
            flip_indices = ones_indices[order[:num_to_flip]]
            b[flip_indices] = 0.0
    elif num_ones < target:
        # Need to turn some 0s to 1s. Pick the largest h values among 0s.
        zeros_indices = np.where(b == 0.0)[0]
        num_to_flip = int(target - num_ones)
        if num_to_flip > 0:
            order = np.argsort(-h[zeros_indices])
            flip_indices = zeros_indices[order[:num_to_flip]]
            b[flip_indices] = 1.0
    
    return b

def construct():
    t0 = time.time()
    best_h = None
    best_val = float('inf')
    
    # Initialize with a good heuristic: balanced alternating or Kronecker sequence
    # Try a few initial candidates
    candidates = []
    
    # 1. Alternating
    h_alt = np.zeros(N)
    h_alt[::2] = 1.0
    candidates.append(h_alt)
    
    # 2. Kronecker sequence with golden ratio
    alpha = (np.sqrt(5) - 1) / 2.0
    h_kron = np.zeros(N)
    for i in range(N):
        # Indicator that {i * alpha} in [0.5, 1) ? No, we need sum N/2.
        # Let's use {i * alpha} < 0.5
        val = (i * alpha) % 1.0
        if val < 0.5:
            h_kron[i] = 1.0
    # Adjust sum if necessary
    if np.sum(h_kron) != N / 2.0:
        # Simple adjustment
        ones = np.where(h_kron == 1.0)[0]
        zeros = np.where(h_kron == 0.0)[0]
        curr_sum = np.sum(h_kron)
        if curr_sum > N / 2.0:
            for idx in ones[:int(curr_sum - N/2.0)]:
                h_kron[idx] = 0.0
        elif curr_sum < N / 2.0:
            for idx in zeros[:int(N/2.0 - curr_sum)]:
                h_kron[idx] = 1.0
    candidates.append(h_kron)
    
    # 3. Random
    h_rand = np.zeros(N)
    idx = np.random.choice(N, N//2, replace=False)
    h_rand[idx] = 1.0
    candidates.append(h_rand)
    
    # Evaluate initial candidates
    for h in candidates:
        val = _max_overlap(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Gradient descent with projection (Projected Gradient Descent on the soft objective)
    # We will alternate between soft gradient steps and projection to binary.
    # Actually, the soft objective is smooth but the projection is non-differentiable.
    # A common approach is to optimize the soft objective over the hypercube [0,1]^N
    # with sum constraint, then round at the end. But the task requires binary.
    # Let's try optimizing a relaxed problem where h in [0,1] and sum=N/2,
    # using the soft-max objective, then project to binary at the end.
    # However, the max overlap of the projected binary might be worse than the relaxed optimum.
    # Given the time limit, we might be better off with Simulated Annealing on binary sequences,
    # using the soft objective as a guide or just the hard objective.
    
    # Let's stick with Simulated Annealing but use the soft objective for faster evaluation?
    # No, soft objective is expensive (full FFT + exp). Hard objective is also full FFT.
    # The hard objective is O(N log N).
    
    # Let's run SA on the binary sequences.
    # To speed up, we can precompute the FFT of the current h?
    # When we swap i and j, the change in H is local.
    # H_new = H_old + delta_i * e^{-i 2 pi k i/N} + delta_j * e^{-i 2 pi k j/N}
    # This is complex to update |H|^2 efficiently without recomputing.
    # For N=200, recomputing FFT is very fast (< 0.1 ms).
    
    # Let's use SA with restarts.
    
    for restart in range(50):
        if _time_remaining() < 5.0:
            break
        
        # Start from best found or random
        if best_h is not None and np.random.random() < 0.5:
            h = best_h.copy()
        else:
            h = np.zeros(N)
            idx = np.random.choice(N, N//2, replace=False)
            h[idx] = 1.0
        
        current_val = _max_overlap(h)
        if current_val < best_val:
            best_val = current_val
            best_h = h.copy()
        
        T = 1.0
        T_min = 0.01
        cooling = 0.9995
        max_steps = 10000
        
        for step in range(max_steps):
            if _time_remaining() < 5.0:
                break
            
            # Pick random i, j
            i = np.random.randint(N)
            j = np.random.randint(N)
            if i == j or h[i] == h[j]:
                continue
            
            # Swap
            h[i], h[j] = h[j], h[i]
            new_val = _max_overlap(h)
            
            delta = new_val - current_val
            if delta < 0 or (T > T_min and np.random.random() < np.exp(-delta / T)):
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_h = h.copy()
            else:
                h[i], h[j] = h[j], h[i]
            
            T *= cooling
            if T < T_min:
                T = T_min
    
    return best_h

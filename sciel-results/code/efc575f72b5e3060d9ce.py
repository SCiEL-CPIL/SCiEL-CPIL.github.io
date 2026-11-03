import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    # Initial guess: Alternating 0.5 or random
    # A good start for low autocorrelation is often a binary sequence like Barker,
    # but we are in [0,1]. Let's start with a random uniform in [0,1] and project.
    # Or better, start with a "flat" sequence and perturb.
    # Actually, starting with a random permutation of 0s and 1s (sum N/2) is a good discrete start.
    # Let's create a binary sequence with N/2 ones and N/2 zeros.
    h = np.zeros(N)
    half = N // 2
    h[:half] = 1.0
    
    # Shuffle to break periodicity
    np.random.shuffle(h)
    
    # We will use PGD to optimize continuous h
    # Objective: Minimize max_k sum_i h_i (1 - h_{i+k})
    
    # Parameters
    beta = 10.0  # Sharpness of LogSumExp
    eta = 0.1    # Step size
    max_iters = 2000
    
    # Precompute shifts? No, FFT is faster.
    
    def project(h):
        # Project h onto {h | 0 <= h_i <= 1, sum(h) = N/2}
        # 1. Clip to [0, 1]
        h_clipped = np.clip(h, 0.0, 1.0)
        
        # 2. Adjust sum
        target_sum = N / 2.0
        current_sum = np.sum(h_clipped)
        
        if abs(current_sum - target_sum) < 1e-9:
            return h_clipped
            
        # We need to add or subtract a constant lambda to reach target_sum,
        # respecting [0, 1] bounds.
        # This is equivalent to finding lambda such that sum(clip(h_clipped - lambda, 0, 1)) = target_sum
        # Note: if we subtract lambda, we are reducing the sum.
        
        # Bisection for lambda
        # If current_sum > target_sum, we need to reduce h. lambda > 0.
        # If current_sum < target_sum, we need to increase h. lambda < 0.
        
        lo = -1.0
        hi = 1.0
        
        for _ in range(50):
            mid = (lo + hi) / 2.0
            h_trial = np.clip(h_clipped - mid, 0.0, 1.0)
            sum_trial = np.sum(h_trial)
            
            if sum_trial > target_sum:
                # We reduced too much (or increased too little).
                # If mid > 0 (reducing), sum decreased. If sum is still > target, we need to reduce more -> increase mid.
                # If mid < 0 (increasing), sum increased. If sum is still > target, we need to increase less -> decrease mid (make it more negative).
                # Let's analyze monotonicity.
                # f(lambda) = sum(clip(h - lambda, 0, 1)).
                # f is decreasing in lambda.
                # We want f(lambda) = target_sum.
                if sum_trial > target_sum:
                    lo = mid
                else:
                    hi = mid
            else:
                hi = mid
                
        h_proj = np.clip(h_clipped - (lo + hi) / 2.0, 0.0, 1.0)
        return h_proj

    def compute_grad(h):
        # Compute A_k = sum_i h_i (1 - h_{i+k}) for all k
        # A_k = N/2 - S_k, where S_k is circular autocorrelation.
        # S_k = sum_i h_i h_{i+k}
        
        # Use FFT for correlation
        # corr = ifft(fft(h) * conj(fft(h))) gives circular correlation?
        # Let H = fft(h).
        # Power spectrum P = H * conj(H).
        # S_k = ifft(P)_k.
        
        H = np.fft.fft(h)
        P = H * np.conj(H)
        S = np.fft.ifft(P).real
        
        # S[k] corresponds to sum_i h[i] * h[(i+k) % N]
        # A_k = N/2 - S_k
        A = N / 2.0 - S
        
        # Compute softmax weights p_k
        # p_k = exp(beta * A_k) / sum(exp(beta * A_k))
        # To avoid overflow, subtract max
        A_max = np.max(A)
        exp_A = np.exp(beta * (A - A_max))
        p = exp_A / np.sum(exp_A)
        
        # Gradient of A_k w.r.t h_j is 1 - h_{j+k} - h_{j-k}
        # grad_g_j = sum_k p_k (1 - h_{j+k} - h_{j-k})
        # This is a convolution of p with the "kernel" derived from h?
        # Let's compute it directly or via FFT.
        # term_k = p_k
        # grad = sum_k p_k (1 - h_{j+k} - h_{j-k})
        # grad = sum_k p_k - sum_k p_k h_{j+k} - sum_k p_k h_{j-k}
        # sum_k p_k = 1
        # sum_k p_k h_{j+k} is circular correlation of p and h?
        # Let's use FFT for speed.
        # corr(p, h)_j = sum_k p_k h_{j+k} ?
        # Standard correlation C(j) = sum_k p(k) h(j+k).
        
        P_fft = np.fft.fft(p)
        H_fft = np.fft.fft(h)
        
        # We want sum_k p_k h_{j+k}
        # This is cross-correlation.
        # In FFT domain, correlation is ifft(fft(p) * conj(fft(h)))?
        # Let's check.
        # x * y correlation = ifft(fft(x) * conj(fft(y)))
        # So sum_k p_k h_{j+k} = ifft(fft(p) * conj(fft(h)))_j ??
        # Actually, standard definition: (p * h)(j) = sum_k p(k) h(j-k).
        # We have h(j+k). Let m = -k. sum_m p(-m) h(j-m).
        # Since p is real and symmetric? No, p is not necessarily symmetric.
        # But A_k is symmetric? S_k = S_{-k}. So A_k = A_{-k}. So p_k = p_{-k}.
        # If p is symmetric, then sum_k p_k h_{j+k} = sum_k p_{-k} h_{j+k} = sum_m p_m h_{j-m} = (p * h)(j).
        
        # Let's assume p is symmetric (it is, because A is symmetric).
        # So we can compute conv/corr via FFT.
        
        # Let's just compute the two terms:
        # T1_j = sum_k p_k h_{j+k}
        # T2_j = sum_k p_k h_{j-k}
        
        # T2 is standard correlation p, h.
        # T1 is correlation p, h with reversed h? Or just T1_j = T2_{-j}?
        # Since p is symmetric, T1_j = sum_k p_k h_{j+k} = sum_k p_{-k} h_{j+k} = sum_m p_m h_{j-m} = T2_j.
        # So T1 = T2.
        
        # So grad_j = 1 - 2 * T2_j.
        
        # Compute T2 = ifft(fft(p) * conj(fft(h)))
        T2 = np.fft.ifft(P_fft * np.conj(H_fft)).real
        
        grad = 1.0 - 2.0 * T2
        
        return grad

    best_h = h.copy()
    best_score = np.inf
    
    for i in range(max_iters):
        if time.time() - start_time > timeout:
            break
            
        grad = compute_grad(h)
        
        # Gradient Descent Step
        h_new = h - eta * grad
        
        # Project
        h_proj = project(h_new)
        
        # Check score
        H = np.fft.fft(h_proj)
        P = H * np.conj(H)
        S = np.fft.ifft(P).real
        A = N / 2.0 - S
        current_max = np.max(A)
        
        if current_max < best_score:
            best_score = current_max
            best_h = h_proj.copy()
            
        h = h_proj
        
        # Adaptive step size?
        # If oscillating, reduce eta.
        if i > 100 and i % 50 == 0:
            eta *= 0.9

    return best_h

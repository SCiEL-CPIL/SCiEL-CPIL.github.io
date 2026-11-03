import numpy as np
import time
import random

N = 50

def _compute_ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * sup|h*h|)"""
    h = np.asarray(h, dtype=float)
    if np.all(h < 1e-15):
        return 0.0
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15 or l2_sq < 1e-15:
        return 0.0
    # Autoconvolution using FFT for speed
    size = 2 * N - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    h_fft = np.fft.fft(h, n=fft_size)
    conv_fft = h_fft * h_fft
    conv = np.fft.ifft(conv_fft).real[:size]
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def _project_simplex(v):
    """Project vector v onto the probability simplex."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    w = np.maximum(v - theta, 0)
    return w

def _generate_sidon_set(n_points, n_total, max_attempts=100):
    """Generate a random Sidon-like set of indices."""
    for _ in range(max_attempts):
        S = set()
        sums = set()
        while len(S) < n_points:
            x = random.randint(0, n_total - 1)
            if x in S:
                continue
            valid = True
            for s in S:
                if x + s in sums:
                    valid = False
                    break
            if valid:
                if 2*x in sums:
                    valid = False
                    break
                if valid:
                    for s in S:
                        sums.add(x + s)
                    sums.add(2*x)
                    S.add(x)
        if len(S) == n_points:
            return sorted(S)
    return random.sample(range(n_total), n_points)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_h = np.zeros(N)
    best_ratio = 0.0
    
    h_uniform = np.ones(N) / N
    r_uniform = _compute_ratio(h_uniform)
    if r_uniform > best_ratio:
        best_ratio = r_uniform
        best_h = h_uniform.copy()
        
    h_delta = np.zeros(N)
    h_delta[0] = 1.0
    r_delta = _compute_ratio(h_delta)
    if r_delta > best_ratio:
        best_ratio = r_delta
        best_h = h_delta.copy()

    num_restarts = 30
    iters_per_restart = 300
    eta_start = 0.05
    tau_start = 5.0
    
    size = 2 * N - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
        
    for restart in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        if restart == 0:
            h = h_uniform.copy()
        elif restart == 1:
            k = random.randint(5, 15)
            indices = random.sample(range(N), k)
            h = np.zeros(N)
            h[indices] = 1.0
            h /= np.sum(h)
        elif restart < 10:
            k = random.randint(5, 12)
            indices = _generate_sidon_set(k, N)
            h = np.zeros(N)
            h[indices] = 1.0
            h /= np.sum(h)
        else:
            k = random.randint(3, 20)
            indices = random.sample(range(N), k)
            h = np.zeros(N)
            weights = np.random.rand(k)
            h[indices] = weights
            h /= np.sum(h)

        current_ratio = _compute_ratio(h)
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_h = h.copy()

        eta = eta_start
        tau = tau_start
        
        for i in range(iters_per_restart):
            if time.time() > deadline - 5.0:
                break
                
            h_fft = np.fft.fft(h, n=fft_size)
            conv_fft = h_fft * h_fft
            conv = np.fft.ifft(conv_fft).real[:size]
            
            max_conv = np.max(conv)
            shifted = tau * (conv - max_conv)
            shifted = np.clip(shifted, -500, 500)
            exp_shifted = np.exp(shifted)
            sum_exp = np.sum(exp_shifted)
            
            if sum_exp < 1e-15:
                break
                
            M = (max_conv + np.log(sum_exp) / tau)
            w = exp_shifted / sum_exp
            
            # Compute grad_M using FFT
            w_fft = np.fft.fft(w, n=fft_size)
            h_fft = np.fft.fft(h, n=fft_size)
            
            # term2 = w * h
            term2 = np.fft.ifft(w_fft * h_fft).real[:N]
            
            # term1 = w \star h (cross-correlation)
            # ifft(fft(w) * conj(fft(h)))
            term1 = np.fft.ifft(w_fft * np.conj(h_fft)).real[:N]
            
            grad_M = (term1 + term2) / tau
            
            l2_sq = np.sum(h**2)
            l1 = np.sum(h)
            
            factor = 1.0 / (l1 * M)
            grad_R = factor * (2.0 * h - (l2_sq / M) * grad_M)
            
            h_new = h + eta * grad_R
            h_new = _project_simplex(h_new)
            
            r_new = _compute_ratio(h_new)
            
            if r_new > best_ratio:
                best_ratio = r_new
                best_h = h_new.copy()
                
            if r_new > current_ratio * (1 - 1e-6):
                h = h_new
                current_ratio = r_new
                eta = min(eta * 1.05, 0.5)
            else:
                eta *= 0.7
                if eta < 1e-5:
                    break

    final_ratio = _compute_ratio(best_h)
    if final_ratio > best_ratio:
        best_ratio = final_ratio

    return list(best_h)

if __name__ == "__main__":
    h = construct()
    print(_compute_ratio(h))

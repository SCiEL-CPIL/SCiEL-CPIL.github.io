"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize_scalar

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar

def compute_overlap(h):
    """
    Compute the maximum overlap of h with shifted 1-h.
    Overlap at shift k is sum_i h[i] * (1 - h[(i+k) % N]).
    We want to minimize the maximum over all shifts k.
    """
    # h is a circular sequence of length N
    # For each shift k, compute sum_i h[i] * (1 - h[(i+k)%N])
    # We can use FFT or direct computation. For N=100, direct is fine.
    
    # Let's compute all cross-correlations
    # overlap[k] = sum_i h[i] * (1 - h[(i+k)%N])
    #            = sum_i h[i] - sum_i h[i]*h[(i+k)%N]
    #            = sum(h) - autocorr_h[k]
    
    # sum(h) is fixed at N/2 = 50
    sum_h = np.sum(h)
    
    # Compute autocorrelation: autocorr[k] = sum_i h[i] * h[(i+k)%N]
    # Using FFT for efficiency
    # Zero-pad to 2N to avoid circular convolution issues? 
    # Actually, we want circular autocorrelation.
    # Circular autocorrelation via FFT:
    # autocorr = ifft(fft(h) * conj(fft(h)))
    
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    # autocorr[k] = sum_i h[i] * h[(i+k)%N] for circular
    # Note: numpy fft gives autocorr[0] = sum h[i]^2, autocorr[k] = sum_i h[i]*h[(i+k)%N]
    
    overlaps = sum_h - autocorr
    return np.max(overlaps)

def project_to_sum(h, target_sum):
    """
    Project h onto the set {x : 0 <= x[i] <= 1, sum(x) = target_sum}.
    Using bisection on the water-filling level.
    """
    target_sum = float(target_sum)
    if np.all(h >= 0) and np.all(h <= 1) and abs(np.sum(h) - target_sum) < 1e-12:
        return h
    
    def residual(lam):
        # x[i] = clip(h[i] - lam, 0, 1)
        # sum(x) - target_sum
        x = np.clip(h - lam, 0.0, 1.0)
        return np.sum(x) - target_sum
    
    # Find lam such that residual(lam) = 0
    # residual is decreasing in lam
    # If h[i] - lam > 1 => x[i] = 1
    # If h[i] - lam < 0 => x[i] = 0
    
    # Bounds for lam:
    # min_h = min(h), max_h = max(h)
    # If lam < min_h - 1, all x[i] = 1? No, if lam is very small, h[i]-lam is large, clip to 1.
    # If lam is very large, h[i]-lam is negative, clip to 0.
    
    lo = np.min(h) - 1.0
    hi = np.max(h) + 1.0
    
    # Check endpoints
    if residual(lo) < 0:
        # Sum is too small even with lam as low as possible?
        # This shouldn't happen if target_sum <= N
        pass
    if residual(hi) > 0:
        # Sum is too large even with lam as high as possible?
        # This shouldn't happen if target_sum >= 0
        pass
    
    for _ in range(100):
        mid = (lo + hi) / 2.0
        r = residual(mid)
        if abs(r) < 1e-12:
            break
        if r > 0:
            lo = mid
        else:
            hi = mid
    
    lam = (lo + hi) / 2.0
    return np.clip(h - lam, 0.0, 1.0)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Initialize h
    # Symmetric initialization might help
    h = np.full(N, 0.5)
    best_h = h.copy()
    best_overlap = compute_overlap(best_h)
    
    # Gradient Descent with Projection
    # We want to minimize max_k overlap[k]
    # This is non-smooth. We can use subgradient or smooth approximation.
    # Let f(h) = max_k (sum_h - autocorr[k]) = sum_h - min_k autocorr[k]
    # Since sum_h is constant, minimizing f is equivalent to maximizing min_k autocorr[k].
    
    # Let's try to maximize min_k autocorr[k].
    # g(h) = min_k autocorr[k]
    # Subgradient of min is -e_k where k is argmin.
    # So we want to increase autocorr[k_min].
    # d(autocorr[k])/d(h[i]) = h[(i+k)%N] + h[(i-k)%N] (by symmetry of autocorr, but let's be precise)
    # autocorr[k] = sum_j h[j] h[(j+k)%N]
    # d/dh[i] = h[(i+k)%N] + h[(i-k)%N]  (if i != (i+k)%N, which is always true for k!=0 mod N)
    # For k=0, d/dh[i] = 2*h[i]
    
    # We want to maximize min_k autocorr[k].
    # Use subgradient ascent: h_new = h + alpha * subgrad
    # subgrad[i] = sum over k where autocorr[k] is min? 
    # Actually, for max of min, the subgradient is the convex hull of gradients of the active constraints.
    # Simplest: pick one active k (e.g., the one with min autocorr) and use its gradient.
    # Or average over all active k's.
    
    # Let's use the gradient of the minimum autocorr.
    # To make it smoother, we can use a soft-min or just pick the argmin.
    
    # Let's try a different approach:
    # Minimize the sum of exponentials of overlaps (smooth max approximation).
    # f_beta(h) = (1/beta) * log(sum_k exp(beta * overlap[k]))
    # This is smooth and approximates max as beta -> inf.
    
    beta = 10.0
    alpha = 0.01
    
    for step in range(10000):
        if time.time() > deadline:
            break
        
        # Compute overlaps
        H = np.fft.fft(h)
        autocorr = np.real(np.fft.ifft(H * np.conj(H)))
        sum_h = np.sum(h)
        overlaps = sum_h - autocorr
        
        # Compute gradient of f_beta
        # f_beta = (1/beta) * log(sum_k exp(beta * overlap[k]))
        # d(f_beta)/d(h[i]) = (1/beta) * (sum_k exp(beta*overlap[k]) * beta * d(overlap[k])/d(h[i])) / sum_k exp(beta*overlap[k])
        #                    = sum_k [ exp(beta*overlap[k]) / sum_j exp(beta*overlap[j]) ] * d(overlap[k])/d(h[i])
        
        # d(overlap[k])/d(h[i]) = d(sum_h - autocorr[k])/d(h[i])
        #                        = 1 - (h[(i+k)%N] + h[(i-k)%N])  for k != 0
        #                        = 1 - 2*h[i]  for k = 0
        
        # Let's compute weights w_k = exp(beta * overlap[k]) / sum_j exp(beta * overlap[j])
        # To avoid overflow, subtract max
        max_overlap = np.max(overlaps)
        w = np.exp(beta * (overlaps - max_overlap))
        w /= np.sum(w)
        
        # Gradient accumulation
        grad = np.zeros(N)
        # For each k, add w[k] * d(overlap[k])/d(h[i]) to grad[i]
        # d(overlap[k])/d(h[i]) = 1 - (h[(i+k)%N] + h[(i-k)%N])
        # This is O(N^2) if done naively.
        # We can use FFT to compute the sum w[k] * h[(i+k)%N]
        # Let v[k] = w[k]
        # sum_k w[k] * h[(i+k)%N] is the circular cross-correlation of w and h.
        # cross_corr[i] = sum_k w[k] * h[(i+k)%N]
        # This can be computed as ifft(fft(w) * conj(fft(h)))? No.
        # Circular correlation: corr[i] = sum_k w[k] * h[(i+k)%N]
        # Let's check:
        # ifft(fft(w) * conj(fft(h)))[i] = sum_k w[k] * conj(h[(i-k)%N])
        # Since h is real, conj(h) = h.
        # So ifft(fft(w) * conj(fft(h)))[i] = sum_k w[k] * h[(i-k)%N]
        # We want sum_k w[k] * h[(i+k)%N].
        # Let j = -k. sum_j w[-j] * h[(i-j)%N] = corr_w_neg[i]
        # So we need to flip w.
        
        w_flipped = w.copy()
        w_flipped[1:] = w[:0:-1]  # w[1], w[2], ..., w[N-1] reversed?
        # w[0] stays, w[1] goes to N-1, etc.
        # Actually, w_flipped[k] = w[(-k)%N]
        w_flipped = np.roll(w, -1)  # No, roll shifts.
        # Let's just construct it explicitly for clarity, N is small.
        w_neg = np.zeros(N)
        w_neg[0] = w[0]
        w_neg[1:] = w[:0:-1]  # w[N-1], w[N-2], ..., w[1]
        
        # Compute sum_k w_neg[k] * h[(i+k)%N]
        # This is ifft(fft(w_neg) * conj(fft(h)))
        H = np.fft.fft(h)
        W_neg = np.fft.fft(w_neg)
        cross_corr = np.real(np.fft.ifft(W_neg * np.conj(H)))
        # cross_corr[i] = sum_k w_neg[k] * h[(i+k)%N] = sum_k w[(-k)%N] * h[(i+k)%N]
        # Let j = -k, sum_j w[j] * h[(i-j)%N]
        # This is the correlation of w and h at shift i.
        # We want sum_k w[k] * h[(i+k)%N].
        # Let's re-verify.
        # We want S[i] = sum_k w[k] * h[(i+k)%N]
        # This is the circular cross-correlation of w and h.
        # S = ifft(fft(w) * conj(fft(h)))?
        # ifft(fft(w) * conj(fft(h)))[i] = sum_k w[k] * h[(i-k)%N]
        # So S[i] = ifft(fft(w) * conj(fft(h)))[-i % N]?
        # Let's just use the w_neg approach.
        # S[i] = sum_k w[k] * h[(i+k)%N]
        # Let j = i+k => k = j-i
        # S[i] = sum_j w[j-i] * h[j]
        # This is the convolution of w and h?
        # Conv[i] = sum_j w[j] * h[i-j]
        # So S is the cross-correlation.
        
        # Let's stick to the explicit loop for gradient, N=100 is small.
        # O(N^2) = 10000 ops per step, 10000 steps = 100M ops. Might be slow in Python.
        # Let's use the FFT method properly.
        
        # S[i] = sum_k w[k] * h[(i+k)%N]
        # This is the circular cross-correlation of w and h.
        # corr[i] = sum_k w[k] * h[(i+k)%N]
        # In FFT: corr = ifft(conj(fft(w)) * fft(h))
        # Let's verify:
        # ifft(conj(fft(w)) * fft(h))[i] = sum_k conj(w[k]) * h[(i+k)%N]
        # Since w is real, conj(w) = w.
        # So corr = ifft(conj(fft(w)) * fft(h)) gives sum_k w[k] * h[(i+k)%N].
        
        W = np.fft.fft(w)
        H = np.fft.fft(h)
        S = np.real(np.fft.ifft(np.conj(W) * H))
        
        # grad[i] = sum_k w[k] * (1 - (h[(i+k)%N] + h[(i-k)%N]))
        #         = sum_k w[k] - sum_k w[k]*h[(i+k)%N] - sum_k w[k]*h[(i-k)%N]
        # sum_k w[k] = 1
        # sum_k w[k]*h[(i+k)%N] = S[i]
        # sum_k w[k]*h[(i-k)%N] = S[-i % N] (by symmetry, if w is symmetric, but w is not necessarily symmetric)
        # Actually, sum_k w[k]*h[(i-k)%N] is the correlation at -i.
        # Let T[i] = sum_k w[k]*h[(i-k)%N]
        # T = ifft(fft(w) * conj(fft(h)))
        
        T = np.real(np.fft.ifft(W * np.conj(H)))
        
        grad = 1.0 - S - T
        
        # Update h
        h = h + alpha * grad
        
        # Project
        h = project_to_sum(h, N/2.0)
        
        # Compute current overlap
        H_curr = np.fft.fft(h)
        autocorr_curr = np.real(np.fft.ifft(H_curr * np.conj(H_curr)))
        overlap_curr = np.max(np.sum(h) - autocorr_curr)
        
        if overlap_curr < best_overlap:
            best_overlap = overlap_curr
            best_h = h.copy()
        
        # Adaptive beta
        if step % 100 == 0:
            beta *= 1.1
            alpha *= 0.99
        
        if beta > 1000:
            break
    
    # Final projection
    best_h = project_to_sum(best_h, N/2.0)
    return best_h
# EVOLVE-BLOCK-END

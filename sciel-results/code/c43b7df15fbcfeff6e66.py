import numpy as np
import time
import sys

N = 600
TIME_LIMIT = 140  # Safety margin before 150s kill

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    # Use FFT for faster convolution calculation
    # We need the linear convolution of f with f
    # Length is 2N - 1
    n_conv = 2 * N - 1
    # Pad to next power of 2 for FFT efficiency? 
    # 1200 is not power of 2. 2048 is. 
    # However, for N=600, np.convolve is very fast. 
    # Let's use FFT for the optimization loop if we call it frequently, 
    # but for the final score, accuracy is key. 
    # Actually, let's stick to np.convolve for the score function as it's exact and fast enough for single calls.
    conv = np.convolve(f, f, mode='full')
    return np.max(conv) / (s * s)

def construct():
    start_time = time.time()
    
    # Strategy: Spectral Lifting / Dual Optimization
    # The autoconvolution peak is related to the maximum of the inverse Fourier transform of |F|^2.
    # Specifically, (f*f)(t) = IFFT(|FFT(f)|^2)(t).
    # We want to minimize max_t (f*f)(t) subject to f >= 0 and sum(f) = 1 (normalization doesn't change the ratio).
    
    # Let's work with the log-spectrum.
    # F[k] = A[k] * exp(i * phi[k])
    # |F[k]|^2 = A[k]^2.
    # The peak of the autoconvolution is determined by the sum of A[k]^2 * exp(i * phi[k] * t).
    # This is complex. 
    # 
    # Alternative simpler spectral approach:
    # Minimize the L_infinity norm of the autoconvolution.
    # This is equivalent to minimizing the peak of the inverse FFT of the power spectrum.
    # 
    # Let's use a projected gradient descent on the time-domain signal f, 
    # but using the FFT to compute the gradient of the "smooth max" of the autoconvolution.
    # The objective is J(f) = max_t (f*f)(t).
    # We can approximate max(x) with a soft-max: mu * log(sum(exp(x/mu))).
    # Gradient of soft-max autoconvolution w.r.t f:
    # d/d f_i [ max_t (sum_j f_j f_{t-j}) ]
    # Using soft max: S = mu * log(sum_t exp((f*f)(t)/mu))
    # dS/d(f*f)(t) = exp((f*f)(t)/mu) / sum_k exp((f*f)(k)/mu) = p_t
    # d(f*f)(t)/df_i = sum_j [delta(t - j - i) + delta(t - (N-1-j) - i)] ? No, convolution indices.
    # (f*f)(t) = sum_k f_k f_{t-k}.
    # d/d f_i (f*f)(t) = f_{t-i} + f_{t-i-N+1} (with wraparound logic for linear conv).
    # Actually, for linear convolution, if we treat it as circular convolution of padded f, 
    # the gradient is the circular correlation.
    
    # Let's use the circular convolution trick for gradient computation speed.
    # Pad f to length M >= 2N-1. Let M be next power of 2 >= 2N-1.
    # M = 2048.
    M = 1
    while M < 2 * N - 1:
        M *= 2
        
    def soft_max_autoconv(f, mu):
        """
        Compute soft-max of autoconvolution and its gradient w.r.t f.
        f: length N
        Returns: value, grad_f (length N)
        """
        # Pad f to M
        f_pad = np.zeros(M)
        f_pad[:N] = f
        
        # Compute circular convolution of f_pad with f_pad
        # This approximates linear convolution if M >= 2N-1
        F = np.fft.fft(f_pad)
        PowerSpectrum = np.abs(F)**2
        # Inverse FFT gives circular autoconvolution
        conv_circ = np.fft.ifft(PowerSpectrum).real
        
        # Extract the linear convolution part
        # The linear convolution of two length-N signals is contained in indices 0 to 2N-2 of the circular convolution (if M is large enough and no wrap-around aliasing into the valid region? 
        # Actually, circular conv of length M:
        # (f_pad * f_pad)(t) = sum_k f_pad[k] f_pad[(t-k) % M]
        # For t in 0..2N-2, and f_pad is zero for k>=N:
        # The terms are sum_{k=0}^{N-1} f[k] f[(t-k)%M].
        # If t-k < 0, (t-k)%M = M + t - k. Since f_pad is zero for index >= N, and M+t-k >= M >= 2N, f_pad[M+t-k] is 0.
        # So for t in 0..2N-2, conv_circ[t] is exactly the linear convolution.
        
        conv_lin = conv_circ[:2*N-1]
        
        # Soft max
        # Shift to avoid overflow
        max_val = np.max(conv_lin)
        shifted = (conv_lin - max_val) / mu
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        
        val = max_val + mu * np.log(sum_exp)
        
        # Probabilities
        p = exp_shifted / sum_exp
        
        # Gradient
        # d val / d f_i = (1/mu) * sum_t p_t * d(conv_lin[t])/d f_i
        # d(conv_lin[t])/d f_i = f[t-i] + f[t-i-N+1] (with appropriate bounds)
        # Let's compute this via correlation.
        # d/d f_i (f*f)(t) is the derivative of sum_k f[k] f[t-k] w.r.t f[i].
        # Term 1: k=i => f[t-i]
        # Term 2: t-k=i => k=t-i => f[t-i]
        # So d/d f_i (f*f)(t) = 2 * f[t-i] ? 
        # Wait. (f*f)(t) = sum_{k=0}^{N-1} f[k] f[t-k].
        # If we vary f[i]:
        # 1. k=i: f[i] f[t-i]. Deriv: f[t-i].
        # 2. t-k=i => k=t-i. If t-i is in [0, N-1], term is f[t-i] f[i]. Deriv: f[t-i].
        # So yes, it's 2 * f[t-i] if t-i is valid, else f[t-i] or f[i] depending on boundaries?
        # Actually, if t-i is out of bounds, the term doesn't exist.
        # So d/d f_i (f*f)(t) = f[t-i] (if 0<=t-i<N) + f[t-i] (if 0<=t-i<N) = 2 f[t-i]?
        # No.
        # Let's check.
        # t=0. f*f[0] = f[0]f[0]. d/d f[0] = 2 f[0]. Formula: 2 f[0-0] = 2 f[0]. Correct.
        # t=1. f*f[1] = f[0]f[1] + f[1]f[0] = 2 f[0]f[1]. d/d f[0] = 2 f[1]. Formula: 2 f[1-0] = 2 f[1]. Correct.
        # d/d f[1] = 2 f[0]. Formula: 2 f[1-1] = 2 f[0]. Correct.
        # t=N. f*f[N] = sum_{k=1}^{N-1} f[k] f[N-k].
        # d/d f[0] = 0. Formula: 2 f[N-0]. f[N] is 0. Correct.
        # So the gradient vector g_i = sum_t p_t * (f[t-i] + f[t-i]) ? 
        # Wait, the sum is over t.
        # g_i = sum_{t=0}^{2N-2} p_t * (f[t-i] + f[t-i] if valid?)
        # Actually, it's simpler to compute the gradient as the correlation of p with f, doubled?
        # Let h(t) = p_t.
        # g_i = sum_t h(t) * d/d f_i (f*f)(t)
        # d/d f_i (f*f)(t) = (f * delta_i)(t) + (delta_i * f)(t) ? No.
        # It is the circular correlation of f with the impulse response?
        # Let's use the property that gradient of ||A f||^2 is 2 A^T A f.
        # Here A is the convolution matrix.
        # The gradient is 2 * (p * f) ? No.
        # Let's compute it explicitly using FFT for speed.
        # grad_f = IFFT( FFT(p_pad) * conj(FFT(f_pad)) ) ?
        # Let's derive:
        # L = sum_t p_t (f*f)(t)
        # dL/d f_i = sum_t p_t (f[t-i] + f[t-i]) ... wait.
        # (f*f)(t) = sum_k f_k f_{t-k}.
        # d/d f_i = sum_t p_t ( delta_{k,i} f_{t-k} + f_k delta_{t-k,i} )
        # = sum_t p_t ( f_{t-i} + f_{t-i} )  (since k=i and t-k=i => k=t-i)
        # = 2 sum_t p_t f_{t-i}
        # This is 2 * (p * f)(i) ? 
        # (p * f)(i) = sum_t p_t f_{i-t}.
        # Let u = t-i. t = u+i.
        # sum_u p_{u+i} f_u.
        # This is the cross-correlation of p and f.
        
        # So grad_f = 2 * corr(p, f)
        # Where corr(p, f)_i = sum_t p_t f_{t-i}.
        
        # We can compute this using FFT.
        # corr(p, f) = IFFT( conj(FFT(p)) * FFT(f) )
        
        p_pad = np.zeros(M)
        p_pad[:2*N-1] = p
        
        P = np.fft.fft(p_pad)
        F = np.fft.fft(f_pad)
        
        corr = np.fft.ifft(np.conj(P) * F).real
        
        grad_f = 2.0 * corr[:N]
        
        return val, grad_f

    # Optimization Loop
    # Initialize f
    # Start with a slightly peaked distribution to encourage low peak autoconvolution?
    # Or uniform? Uniform has high peak relative to flat?
    # Uniform f=1/N. Autoconv is triangle. Peak is N/2 * (1/N)^2 * N? 
    # Let's just start with a random positive vector or a Gaussian.
    # A Gaussian in time domain has Gaussian spectrum.
    
    # Let's try a few initializations.
    best_f = None
    best_score = np.inf
    
    inits = []
    # 1. Uniform
    inits.append(np.ones(N) / N)
    # 2. Gaussian centered
    x = np.linspace(-1, 1, N)
    inits.append(np.exp(-5 * x**2))
    inits[-1] /= np.sum(inits[-1])
    # 3. Bump function (smooth step)
    inits.append(np.maximum(0, 1 - np.abs(x)**4))
    inits[-1] /= np.sum(inits[-1])
    
    for init in inits:
        if time.time() - start_time > TIME_LIMIT * 0.8:
            break
            
        f = init.copy()
        # Normalize
        f /= np.sum(f)
        
        # Projected Gradient Descent
        # We minimize J(f) = soft_max_autoconv(f, mu)
        # Subject to f >= 0 and sum(f) = 1.
        
        # Adaptive learning rate
        lr = 0.1
        mu = 0.1 # Soft max temperature. Smaller is sharper (closer to max).
        
        # We need to handle the non-smoothness of the max. 
        # Soft max is smooth. 
        # But the constraint f>=0 makes it non-smooth at boundaries.
        # We can use a barrier or projection.
        
        for step in range(1000):
            if time.time() - start_time > TIME_LIMIT * 0.95:
                break
                
            # Compute objective and gradient
            val, grad = soft_max_autoconv(f, mu)
            
            # Update
            f_new = f - lr * grad
            
            # Projection onto simplex (f >= 0, sum(f) = 1)
            # We can do this by clipping and renormalizing, but that's not strictly projection.
            # Proper projection onto simplex:
            # 1. Sort f_new in descending order.
            # 2. Find rho such that sum(max(f_new - rho, 0)) = 1.
            
            def project_simplex(v):
                v = np.asarray(v)
                n = v.size
                u = np.sort(v)[::-1]
                cssv = np.cumsum(u)
                rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
                theta = (cssv[rho] - 1) / (rho + 1.0)
                w = np.maximum(v - theta, 0)
                return w
            
            f_proj = project_simplex(f_new)
            
            # Check improvement
            # We need to compute the actual score (max autoconv) to ensure we are improving.
            # The soft max is an upper bound.
            # Let's compute the true score every few steps or if it looks good.
            
            # To save time, we only compute the true score if the soft max value decreased significantly
            # or periodically.
            
            # Actually, let's just use the soft max value for the optimization steps,
            # but track the best true score.
            
            # Compute true score for current f_proj
            # To avoid overhead, we can just use f_proj for the next iteration.
            # We will compute the true score at the end of the loop or if it's a candidate for best.
            
            # Let's compute true score every 10 steps to update best.
            if step % 10 == 0 or step == 999:
                true_score = compute_score(f_proj)
                if true_score < best_score:
                    best_score = true_score
                    best_f = f_proj.copy()
            
            # Adaptive learning rate
            # If val increased, reduce lr.
            # We need previous val.
            if step > 0:
                if val > prev_val:
                    lr *= 0.5
                else:
                    lr = min(lr * 1.05, 1.0)
            prev_val = val
            
            f = f_proj
            
            # Convergence check
            if step > 10 and abs(val - prev_val) < 1e-9:
                break
            
            # Reduce mu (anneal) to make soft max sharper
            if step % 50 == 0:
                mu *= 0.9

    # Final Polish: Coordinate Descent on the best_f found
    if best_f is not None:
        f = best_f.copy()
        current_score = best_score
        
        # Simple coordinate descent to polish
        # We can try adjusting individual elements.
        # To speed up, we can use the gradient information from the last soft max step?
        # Or just do a few rounds of local search.
        
        # Let's do a few rounds of "top-k" adjustment.
        # Identify the index of the peak in the autoconvolution.
        # Reduce f at indices that contribute most to that peak.
        
        for _ in range(20):
            if time.time() - start_time > TIME_LIMIT * 0.98:
                break
                
            conv = np.convolve(f, f, mode='full')
            peak_idx = np.argmax(conv)
            peak_val = conv[peak_idx]
            
            # Find which f[i] contribute to conv[peak_idx]
            # conv[peak_idx] = sum_k f[k] f[peak_idx - k]
            # We want to reduce f[k] or f[peak_idx-k].
            # Let's try reducing the largest contributor.
            
            contributors = []
            for k in range(N):
                j = peak_idx - k
                if 0 <= j < N:
                    contributors.append((k, f[k] * f[j]))
            
            if not contributors:
                break
                
            # Sort by contribution
            contributors.sort(key=lambda x: x[1], reverse=True)
            
            improved = False
            # Try reducing the top contributor
            k_idx, contrib_val = contributors[0]
            
            # Try reducing f[k_idx]
            orig = f[k_idx]
            for factor in [0.9, 0.8, 0.7]:
                f[k_idx] = orig * factor
                # Renormalize
                s = np.sum(f)
                if s < 1e-15:
                    f[k_idx] = orig
                    continue
                f /= s
                
                new_score = compute_score(f)
                if new_score < current_score - 1e-12:
                    current_score = new_score
                    improved = True
                    break
                else:
                    f[k_idx] = orig
                    f /= s # Restore normalization
            
            if not improved:
                break

        best_f = f

    # Ensure final output is valid
    if best_f is None:
        best_f = np.ones(N) / N
        
    best_f = np.maximum(best_f, 0.0)
    s = np.sum(best_f)
    if s > 0:
        best_f /= s
        
    return [float(x) for x in best_f]

if __name__ == "__main__":
    f = construct()
    print(f"Score: {compute_score(f)}")
    print(f"Length: {len(f)}")
    print(f"Min: {min(f)}, Max: {max(f)}")

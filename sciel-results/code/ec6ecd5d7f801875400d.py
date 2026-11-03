import numpy as np
import time
import scipy.optimize as opt

N = 600
TIME_LIMIT = 140

def fft_autoconv(f):
    """Compute full autoconvolution via FFT."""
    n = len(f)
    size = 2 * n - 1
    fft_size = 1
    while fft_size < size:
        fft_size *= 2
    F = np.fft.rfft(f, n=fft_size)
    conv = np.fft.irfft(F * F, n=fft_size)
    return conv[:size]

def compute_score(f):
    """Compute the normalized maximum of the autoconvolution."""
    f = np.asarray(f, dtype=np.float64)
    if np.any(f < -1e-15):
        return np.inf
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return np.inf
    conv = fft_autoconv(f)
    return np.max(conv) / (s * s)

def spectral_gradient_descent(f_init, start_time, max_time=120.0):
    """
    Gradient descent in the frequency domain with soft non-negativity penalty.
    """
    f = f_init.copy()
    f = np.maximum(f, 0.0)
    s = np.sum(f)
    if s < 1e-15:
        return f_init
    f = f / s # Normalize
    
    # Work with full FFT for gradient symmetry ease, or rFFT?
    # Let's use full complex FFT for easier gradient handling of conjugates.
    F = np.fft.fft(f)
    
    best_f = f.copy()
    best_val = compute_score(f)
    
    # Parameters
    lr = 0.01
    momentum = 0.9
    penalty_lambda = 10.0
    v = np.zeros_like(F) # Momentum for F
    
    for step in range(10000):
        if time.time() - start_time > max_time - 5:
            break
            
        # Compute current f from F
        f_curr = np.real(np.fft.ifft(F))
        
        # Enforce non-negativity softly? Or hard clip and re-FFT?
        # Hard clipping breaks the gradient flow.
        # Let's use a penalty term in the objective: L = Score + lambda * ||min(f,0)||^2
        # But Score involves max of conv.
        
        # Let's compute the score and the penalty
        f_nonneg = np.maximum(f_curr, 0)
        s_curr = np.sum(f_nonneg)
        if s_curr < 1e-15:
            break
        f_norm = f_nonneg / s_curr
        conv = fft_autoconv(f_norm)
        max_idx = np.argmax(conv)
        max_val = conv[max_idx]
        score = max_val / (s_curr * s_curr)
        
        # Penalty for negative parts
        neg_part = np.minimum(f_curr, 0)
        penalty = penalty_lambda * np.sum(neg_part * neg_part)
        
        total_loss = score + penalty
        
        if total_loss < best_val + penalty: # Note: best_val is raw score, comparing mixed metrics is tricky.
             # We should track best based on raw score of the projected f
             pass
        
        # Update best if the projected f is better
        if score < best_val - 1e-10:
            best_val = score
            best_f = f_norm.copy()

        # Calculate Gradient w.r.t F
        # d(score)/dF
        # score = max(conv) / s^2
        # conv = ifft(|F|^2)
        # d(conv_m)/dF_k = 2 * conj(F_k) * exp(i 2 pi m k / N)
        # d(score)/dF_k = (1/s^2) * d(conv_m*)/dF_k - 2*score/s * ds/dF_k
        # s = sum(f) = sum(ifft(F)) = F_0 (DC component)
        # ds/dF_k = 1 if k==0 else 0 (since f is real, F_0 is real)
        
        # Let's approximate the gradient of the max term.
        # grad_conv = 2 * conj(F) * ifft_phase(max_idx)
        # where ifft_phase(m) is the vector of exp(i 2 pi m k / N)
        
        phase_vec = np.exp(1j * 2 * np.pi * max_idx * np.arange(N) / N)
        grad_conv = 2 * np.conj(F) * phase_vec
        
        # Gradient of score = (grad_conv - 2 * score * f * ... ) / s^2 ?
        # Actually, it's easier to just use finite differences for the gradient?
        # No, N=600, finite diff is too slow.
        
        # Let's use the analytical gradient for the max term.
        # d(score)/dF_k = (1/s^2) * (d(conv_m*)/dF_k - 2*conv_m*/s * ds/dF_k)
        # ds/dF_k = delta_{k,0}
        
        ds_dF = np.zeros(N, dtype=complex)
        ds_dF[0] = 1.0
        
        grad_score = (grad_conv - 2 * max_val * ds_dF / s_curr) / (s_curr * s_curr)
        
        # Gradient of penalty: L_pen = lambda * sum(min(f,0)^2)
        # d(f)/dF_k = ifft_basis_k
        # d(L_pen)/dF_k = lambda * sum( 2 * min(f_j,0) * d(f_j)/dF_k )
        # d(f_j)/dF_k = (1/N) * exp(i 2 pi j k / N)
        # This is an FFT of the gradient in time domain.
        # grad_pen_time = 2 * lambda * min(f_curr, 0)
        # grad_pen_freq = fft(grad_pen_time) / N
        
        grad_pen_time = 2 * penalty_lambda * np.minimum(f_curr, 0)
        grad_pen_freq = np.fft.fft(grad_pen_time) / N
        
        grad_total = grad_score + grad_pen_freq
        
        # Update F with momentum
        v = momentum * v - lr * grad_total
        F = F + v
        
        # Ensure F is Hermitian for real f?
        # The update might break Hermitian symmetry slightly due to numerical errors or if we don't enforce it.
        # Enforce Hermitian symmetry: F[N-k] = conj(F[k])
        for k in range(1, N // 2):
            avg = (F[k] + np.conj(F[N-k])) / 2
            F[k] = avg
            F[N-k] = np.conj(avg)
        # F[0] and F[N/2] (if N even) should be real
        F[0] = np.real(F[0])
        if N % 2 == 0:
            F[N//2] = np.real(F[N//2])

    return best_f

def construct():
    start_time = time.time()
    
    x = np.linspace(-0.25, 0.25, N, endpoint=False) + (0.5 / N)
    candidates = []
    
    # Uniform
    f = np.ones(N)
    candidates.append(f / np.sum(f))
    
    # Gaussian
    for alpha in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        f = np.exp(-alpha * x ** 2)
        candidates.append(f / np.sum(f))
        
    # Triangle
    for w in [0.1, 0.2, 0.25, 0.4, 0.5]:
        f = np.zeros(N)
        mask = np.abs(x) <= w / 2.0
        f[mask] = 1.0 - 2.0 * np.abs(x[mask]) / w
        f = np.maximum(f, 0)
        if np.sum(f) > 1e-15:
            candidates.append(f / np.sum(f))

    # Score them
    scored = []
    for f in candidates:
        score = compute_score(f)
        scored.append((f, score))
    
    scored.sort(key=lambda x: x[1])
    best_f = scored[0][0].copy()
    best_score = scored[0][1]
    
    # Apply Spectral Gradient Descent to the best initial candidate
    if time.time() - start_time < TIME_LIMIT - 10:
        refined = spectral_gradient_descent(best_f, start_time, TIME_LIMIT - 5)
        refined_score = compute_score(refined)
        if refined_score < best_score:
            best_f = refined
            best_score = refined_score
            
    # Fallback/Polish with original coordinate descent if time permits?
    # The directive says structural change, so we stick to the new method primarily.
    # But we can use the old method as a polish if it helps?
    # Let's just return the best from the new method.
    
    # Final normalization
    best_f = np.maximum(best_f, 0.0)
    s = np.sum(best_f)
    if s > 1e-15:
        best_f = best_f / s
        
    return [float(x) for x in best_f]

import time
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE = 140.0  # Safety margin before 150s kill

def _overlaps_from_h(h):
    """Compute all circular overlaps using FFT-based autocorrelation."""
    # overlap[k] = sum_i h[i] * (1 - h[(i+k) % N]) = sum(h) - autocorr[k]
    # autocorr[k] = sum_i h[i] * h[(i+k) % N]
    # Use FFT: autocorr = ifft(fft(h) * conj(fft(h)))
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    return np.sum(h) - autocorr

def _h_from_coeffs(coeffs):
    """Reconstruct h from Fourier coefficients."""
    c0 = coeffs[0]
    c_complex = coeffs[1:]
    
    spectrum = np.zeros(N, dtype=complex)
    spectrum[0] = c0
    for k in range(1, N // 2):
        spectrum[k] = c_complex[k - 1]
        spectrum[N - k] = np.conj(c_complex[k - 1])
    
    if N % 2 == 0:
        spectrum[N // 2] = np.real(c_complex[N // 2 - 1])
    
    h = np.fft.ifft(spectrum).real
    return h

def _objective_and_grad(coeffs):
    """Objective with analytic gradient via chain rule."""
    h = _h_from_coeffs(coeffs)
    N_local = len(h)
    
    # Penalties
    sum_h = np.sum(h)
    sum_penalty = 100.0 * (sum_h - N_local / 2) ** 2
    
    below = np.maximum(0, -h)
    above = np.maximum(0, h - 1)
    bound_penalty = 100.0 * (np.sum(below ** 2) + np.sum(above ** 2))
    
    # Overlaps via FFT
    overlaps = _overlaps_from_h(h)
    
    # Soft-max surrogate
    tau = 0.05
    exps = np.exp(overlaps / tau)
    log_sum_exp = np.log(np.sum(exps)) * tau
    soft_max = log_sum_exp
    
    obj = soft_max + sum_penalty + bound_penalty
    
    # Gradient via chain rule
    # d(soft_max)/d(h[i]) = (1/N) * sum_k exp(overlap[k]/tau) * d(overlap[k])/d(h[i])
    # overlap[k] = sum(h) - autocorr[k]
    # d(overlap[k])/d(h[i]) = 1 - d(autocorr[k])/d(h[i]) = 1 - (h[(i-k)%N] + h[(i+k)%N])... 
    # Actually autocorr[k] = sum_j h[j]*h[(j+k)%N]
    # d(autocorr[k])/d(h[i]) = h[(i-k)%N] + h[(i+k)%N] (for k != 0 and k != N/2)
    # For k=0: d(autocorr[0])/d(h[i]) = 2*h[i]
    # For k=N/2 (if N even): d(autocorr[N/2])/d(h[i]) = h[(i+N/2)%N]
    
    weights = exps / np.sum(exps)  # normalized weights for soft-max
    
    # d(soft_max)/d(h[i]) = tau * sum_k weights[k] * d(overlap[k])/d(h[i])
    # d(overlap[k])/d(h[i]) = 1 - d(autocorr[k])/d(h[i])
    
    grad_h = np.zeros(N_local)
    for k in range(N_local):
        dk = tau * weights[k]
        if k == 0:
            # d(autocorr[0])/d(h[i]) = 2*h[i]
            grad_h += dk * (1 - 2 * h)
        elif N_local % 2 == 0 and k == N_local // 2:
            # d(autocorr[N/2])/d(h[i]) = h[(i+N/2)%N]
            shifted = np.roll(h, k)
            grad_h += dk * (1 - shifted)
        else:
            # d(autocorr[k])/d(h[i]) = h[(i-k)%N] + h[(i+k)%N]
            shifted_minus = np.roll(h, k)  # h[(i+k)%N]
            shifted_plus = np.roll(h, -k)  # h[(i-k)%N]
            grad_h += dk * (1 - shifted_minus - shifted_plus)
    
    # Gradient of penalties
    grad_h += 200.0 * (sum_h - N_local / 2)
    grad_h += 200.0 * (np.where(h < 0, -h, 0) * (-1) + np.where(h > 1, h - 1, 0))
    # Fix: d(bound_penalty)/d(h[i]) = 200*(-h[i]) if h[i]<0, 200*(h[i]-1) if h[i]>1
    grad_h += 200.0 * np.where(h < 0, -h, 0) * (-1)  # This is wrong, let me redo
    
    # Redo gradient of bound penalty properly
    # bound_penalty = 100 * (sum(max(0,-h)^2) + sum(max(0,h-1)^2))
    # d/dh[i] = 100 * (2*max(0,-h[i])*(-1) if h[i]<0 else 0 + 2*max(0,h[i]-1) if h[i]>1 else 0)
    grad_bound = np.zeros(N_local)
    mask_below = h < 0
    mask_above = h > 1
    grad_bound[mask_below] = 200.0 * (-h[mask_below]) * (-1)  # = 200*h[i] for h[i]<0... wait
    # d(100 * (-h[i])^2)/d(h[i]) = 100 * 2*(-h[i])*(-1) = 200*h[i] for h[i] < 0
    # d(100 * (h[i]-1)^2)/d(h[i]) = 100 * 2*(h[i]-1) = 200*(h[i]-1) for h[i] > 1
    grad_bound[mask_below] = 200.0 * h[mask_below]
    grad_bound[mask_above] = 200.0 * (h[mask_above] - 1)
    
    # Redo grad_h properly
    grad_h = np.zeros(N_local)
    for k in range(N_local):
        dk = tau * weights[k]
        if k == 0:
            grad_h += dk * (1 - 2 * h)
        elif N_local % 2 == 0 and k == N_local // 2:
            shifted = np.roll(h, k)
            grad_h += dk * (1 - shifted)
        else:
            shifted_minus = np.roll(h, k)
            shifted_plus = np.roll(h, -k)
            grad_h += dk * (1 - shifted_minus - shifted_plus)
    
    grad_h += 200.0 * (sum_h - N_local / 2)
    grad_h += grad_bound
    
    # Now chain rule: coeffs -> h via IFFT
    # h = IFFT(spectrum), spectrum depends on coeffs
    # d(h[i])/d(coeffs[0]) = 1/N for all i (DC term)
    # d(h[i])/d(coeffs[k]) for k>=1: spectrum[k] = coeffs[k], spectrum[N-k] = conj(coeffs[k])
    # h = IFFT(spectrum) = (1/N) * sum_k spectrum[k] * exp(2*pi*i*k*i/N)
    # d(h[i])/d(re(coeffs[m])) for m>=1: (1/N)*(exp(2*pi*i*m*i/N) + exp(2*pi*i*(N-m)*i/N)) = (2/N)*cos(2*pi*m*i/N)
    # d(h[i])/d(im(coeffs[m])) for m>=1: (1/N)*(i*exp(2*pi*i*m*i/N) - i*exp(2*pi*i*(N-m)*i/N)) = (2/N)*sin(2*pi*m*i/N) * (-1)... 
    # Let me just use: h = IFFT(spectrum), so dh/dc = IFFT(d spectrum/dc)
    # This is getting complex. Let me use a simpler approach: since we're in Fourier domain,
    # just compute gradient numerically but with the fast overlap computation.
    
    # Actually, let's just return obj and use a simpler optimization approach.
    # The gradient computation is error-prone. Let's use the fact that the problem
    # is well-behaved and use a different strategy.
    
    return obj, None  # Return None for gradient to use numerical differentiation

def construct():
    """
    Returns a sequence of step heights h in [0, 1] whose sum is exactly N/2.
    """
    start_time = time.time()
    
    # Start with h = 0.5 everywhere (theoretical minimum)
    best_h = np.full(N, 0.5)
    best_score = _overlaps_from_h(best_h).max()
    
    # Try to improve from various starting points
    for attempt in range(10):
        if time.time() - start_time > DEADLINE:
            break
        
        if attempt == 0:
            # Start from uniform
            h_init = np.full(N, 0.5)
        elif attempt == 1:
            # Start from a sinusoidal perturbation
            h_init = 0.5 + 0.1 * np.sin(2 * np.pi * np.arange(N) / N)
            h_init = np.clip(h_init, 0, 1)
            h_init += (N / 2.0 - np.sum(h_init)) / N
            h_init = np.clip(h_init, 0, 1)
        elif attempt == 2:
            # Start from a binary-like sequence
            h_init = np.zeros(N)
            h_init[::2] = 1.0
            h_init += (N / 2.0 - np.sum(h_init)) / N
        elif attempt == 3:
            # Random perturbation
            h_init = 0.5 + 0.2 * (np.random.rand(N) - 0.5)
            h_init = np.clip(h_init, 0, 1)
            h_init += (N / 2.0 - np.sum(h_init)) / N
            h_init = np.clip(h_init, 0, 1)
        else:
            # More random perturbations
            h_init = 0.5 + 0.3 * (np.random.rand(N) - 0.5)
            h_init = np.clip(h_init, 0, 1)
            h_init += (N / 2.0 - np.sum(h_init)) / N
            h_init = np.clip(h_init, 0, 1)
        
        # Convert to Fourier coefficients
        spectrum = np.fft.fft(h_init)
        coeffs = np.zeros(N // 2)
        coeffs[0] = np.real(spectrum[0])
        for k in range(1, N // 2):
            coeffs[k] = spectrum[k]
        if N % 2 == 0:
            coeffs[N // 2 - 1] = np.real(spectrum[N // 2])
        
        # Optimize using Nelder-Mead (no gradient needed, robust)
        def obj_func(c):
            h = _h_from_coeffs(c)
            h = np.clip(h, 0, 1)
            h += (N / 2.0 - np.sum(h)) / N
            h = np.clip(h, 0, 1)
            overlaps = _overlaps_from_h(h)
            return overlaps.max()
        
        try:
            result = minimize(
                obj_func,
                coeffs,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            
            h_opt = _h_from_coeffs(result.x)
            h_opt = np.clip(h_opt, 0, 1)
            h_opt += (N / 2.0 - np.sum(h_opt)) / N
            h_opt = np.clip(h_opt, 0, 1)
            
            score_opt = _overlaps_from_h(h_opt).max()
            if score_opt < best_score:
                best_h = h_opt
                best_score = score_opt
        except Exception:
            pass
    
    # Ensure exact sum
    diff_final = N / 2.0 - np.sum(best_h)
    best_h += diff_final / N
    best_h = np.clip(best_h, 0, 1)
    
    # Final adjustment to ensure exact sum after clipping
    diff2 = N / 2.0 - np.sum(best_h)
    if abs(diff2) > 1e-15:
        # Distribute the difference
        idx = np.argsort(best_h)
        if diff2 > 0:
            for i in idx[::-1]:
                add = min(diff2, 1.0 - best_h[i])
                best_h[i] += add
                diff2 -= add
                if abs(diff2) < 1e-15:
                    break
        else:
            for i in idx:
                sub = min(-diff2, best_h[i])
                best_h[i] -= sub
                diff2 += sub
                if abs(diff2) < 1e-15:
                    break
    
    return best_h

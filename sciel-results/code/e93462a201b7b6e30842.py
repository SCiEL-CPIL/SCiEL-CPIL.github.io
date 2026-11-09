"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _compute_sidelobe_energy(h):
    """
    Compute the sum of squares of the non-zero lags of the circular autocorrelation.
    This is proportional to the sum of the 4th powers of the DFT magnitudes.
    Lower is better.
    """
    f = np.fft.fft(h.astype(float))
    power = np.abs(f) ** 4
    # The total energy in autocorrelation domain is sum(|f|^4)/N
    # sum(c_k^2) = (1/N) * sum(|f_k|^4)
    # We want to minimize the off-center lags.
    # c_0 = N (for +/-1 sequence, sum h_i^2 = N).
    # sum_{k!=0} c_k^2 = (1/N) * sum(|f_k|^4) - N
    # Minimizing sum(|f_k|^4) is equivalent.
    return np.sum(power)

def _autocorr_peak_periodic(h):
    """
    Compute the maximum absolute autocorrelation over non-zero lags.
    This is the metric used for the score in the original task description context, 
    but the prompt suggests L2 norm might be a better proxy for C3 bound improvement 
    or at least a different search landscape. 
    However, the evaluator measures the "raw value". 
    Let's stick to the original metric for evaluation but use L2 for optimization gradient.
    """
    f = np.fft.fft(h.astype(float))
    power = f * np.conj(f)
    acorr = np.fft.ifft(power).real
    if N > 1:
        return np.max(np.abs(acorr[1:]))
    else:
        return 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    best_h = None
    best_score = float('inf')
    
    # Multiple restarts with different initializations
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize x with random smooth signal or sinusoid
        # Option 1: Random
        if r % 2 == 0:
            x = np.random.randn(N)
        else:
            # Option 2: Windowed sinusoid
            t = np.linspace(0, 1, N, endpoint=False)
            freq = 1 + np.random.rand() * 5
            window = np.hanning(N)
            x = window * np.sin(2 * np.pi * freq * t)
            
        # Projected Gradient Descent on continuous x to minimize sum(|FFT(x)|^4)
        # Gradient of sum(|FFT(x)|^4) w.r.t x:
        # Let F = FFT(x). J = sum(|F|^4) = sum(F * conj(F) * F * conj(F))
        # dJ/dx = 4 * IFFT( conj(FFT(x)) * |FFT(x)|^2 * 2 ) ? No.
        # J = sum_k F_k F_k^* F_k F_k^*
        # dJ/dF_k^* = 2 F_k |F_k|^2
        # dJ/dx = IFFT( 2 * conj(FFT(x)) * |FFT(x)|^2 ) * 2 ? 
        # Actually, for real x, the gradient of sum |F|^4 is 4 * real( IFFT( 2 * conj(F) * |F|^2 ) )?
        # Let's use the standard result: grad_x sum |FFT(x)|^4 = 4 * IFFT( conj(FFT(x)) * |FFT(x)|^2 ).real
        # Wait, scaling factors depend on FFT definition.
        # Let's just use finite differences or a robust numerical gradient if unsure, 
        # but analytic is faster.
        # For numpy FFT (unnormalized forward, 1/N backward):
        # f = fft(x)
        # grad_f = 2 * np.conj(f) * np.abs(f)**2
        # grad_x = np.fft.ifft(grad_f).real
        # J = sum(|f|^4). dJ/df = 2 f |f|^2 (for complex, treating f and f* independently, 
        # actually d/df* (f f* f f*) = 2 f |f|^2).
        # So grad_x = ifft(2 * conj(f) * |f|^2).real
        
        step_size = 0.1
        x = x / (np.linalg.norm(x) + 1e-9) * np.sqrt(N) # Normalize energy
        
        for step in range(1000):
            if time.time() > deadline:
                break
                
            f = np.fft.fft(x)
            grad_f = 2.0 * np.conj(f) * (np.abs(f) ** 2)
            grad_x = np.fft.ifft(grad_f).real
            
            # Normalize gradient to step size
            norm_grad = np.linalg.norm(grad_x)
            if norm_grad > 1e-9:
                grad_x = grad_x / norm_grad
            
            # Update x
            x = x - step_size * grad_x
            
            # Project to unit sphere (or just keep magnitude reasonable)
            # The sign function is the projection to {-1, 1}.
            # We keep x continuous but guide it.
            
            # Evaluate discrete metric on sign(x)
            h = np.sign(x)
            # Ensure no zeros (sign(0) is 0, map to 1)
            h[h == 0] = 1.0
            
            score = _autocorr_peak_periodic(h)
            
            if score < best_score:
                best_score = score
                best_h = h.copy()
                
            # Adaptive step size: if score improved, keep, else reduce?
            # Simple annealing of step size
            step_size *= 0.999

    # Final local search on the best found discrete solution
    if best_h is not None and time.time() < deadline:
        h = best_h.copy()
        current_score = best_score
        improved = True
        while improved and time.time() < deadline:
            improved = False
            perm = np.random.permutation(N)
            for i in perm:
                if time.time() > deadline:
                    break
                h[i] *= -1
                new_score = _autocorr_peak_periodic(h)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    best_score = current_score
                    best_h = h.copy()
                else:
                    h[i] *= -1

    if best_h is None:
        best_h = np.random.choice([-1, 1], size=N).astype(np.float64)
        
    return best_h
# EVOLVE-BLOCK-END

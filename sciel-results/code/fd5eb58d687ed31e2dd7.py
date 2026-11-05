import numpy as np

N = 400

def construct():
    # Target: flat autoconvolution
    # C_target is a constant vector of length 2*N - 1
    L = 2 * N - 1
    # We want h*h = K * ones(L)
    # FFT(h*h) = FFT(h) * FFT(h) (with zero padding)
    # Let H be the FFT of h zero-padded to L
    # H^2 = FFT(K * ones(L))
    # FFT(ones(L)) is a Dirichlet kernel
    # We compute H_target = sqrt(FFT(ones(L)))
    
    # Note: FFT of a real constant vector is real and non-negative (at DC) and oscillates?
    # No, FFT of a constant vector is a delta function at DC?
    # Wait. The vector is 1, 1, ..., 1 (length L).
    # FFT of this is L * delta[k] (discrete).
    # So H^2 would be a delta function.
    # Then H would be a delta function.
    # Then h would be a constant?
    # If h is constant, h*h is a triangle.
    # Contradiction.
    
    # Let's re-evaluate.
    # We want C_k = K for all k.
    # This is impossible for a finite sequence h?
    # If h has finite support, C is finite support.
    # C is non-zero only for k in [0, 2N-2].
    # So we want C_k = K for k in [0, 2N-2] and 0 elsewhere.
    # This is a box function in time.
    # So we want the FFT of C to be the spectrum of a box.
    # Spectrum of a box is a Dirichlet kernel (sinc-like).
    # So H^2 should be a Dirichlet kernel.
    # H = sqrt(Dirichlet).
    # This matches my previous thought.
    
    # The vector "ones(L)" in time domain corresponds to the target C.
    # So we take FFT of "ones(L)".
    
    ones_L = np.ones(L)
    C_spec = np.fft.fft(ones_L, n=L) # FFT of the target convolution
    # C_spec should be real? 
    # FFT of a real symmetric vector is real.
    # ones_L is symmetric.
    # So C_spec is real.
    # Is it non-negative?
    # Dirichlet kernel can be negative.
    # We need H^2 = C_spec.
    # If C_spec has negative parts, we can't take sqrt directly.
    # We can set H^2 = max(C_spec, 0) or similar.
    # Or we can use the fact that we just want to minimize max(C).
    # A better initialization might be to use the analytic solution for minimum peak autocorrelation.
    
    # However, let's try the sqrt of the magnitude.
    # H_target = sqrt(abs(C_spec))
    # Phase? We can choose phase to minimize max(C).
    # Zero phase (symmetric h) is a good start.
    
    H_target = np.sqrt(np.abs(C_spec))
    # Ensure H_target is consistent with real h?
    # For real h, H[k] = conj(H[L-k]).
    # abs(C_spec) is symmetric, so sqrt is symmetric.
    # So h will be real and symmetric.
    
    h = np.fft.ifft(H_target, n=L).real
    h = h[:N]
    
    # Normalize
    s = np.sum(h)
    if abs(s) < 1e-9:
        h[0] += 1e-9
        s = np.sum(h)
    h = h / s
    
    # Refine with gradient descent
    best_h = h.copy()
    c = np.convolve(h, h)
    best_value = 2.0 * N * np.max(c) / np.sum(h)**2
    
    eta = 0.01
    momentum = np.zeros(N)
    mu = 0.9
    
    for iteration in range(100):
        c = np.convolve(h, h)
        max_idx = np.argmax(c)
        
        grad = np.zeros(N)
        i_vals = np.arange(N)
        j_vals = max_idx - i_vals
        mask = (j_vals >= 0) & (j_vals < N)
        grad[i_vals[mask]] = 2.0 * h[j_vals[mask]]
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm
        
        momentum = mu * momentum - eta * grad
        h = h + momentum
        
        s = np.sum(h)
        if abs(s) < 1e-9:
            h[0] += 1e-9
            s = np.sum(h)
        h = h / s
        
        c = np.convolve(h, h)
        current_value = 2.0 * N * np.max(c) / np.sum(h)**2
        if current_value < best_value:
            best_value = current_value
            best_h = h.copy()
            
    return best_h

import numpy as np
import time

N = 100

def construct():
    # Strategy: Model h as a sum of Fourier modes.
    # h[n] = 0.5 + sum_{k=1}^{K} (a_k cos(2*pi*k*n/N) + b_k sin(2*pi*k*n/N))
    # This ensures sum(h) = N * 0.5 = N/2 automatically.
    # We optimize coefficients a_k, b_k to minimize max_overlap.
    
    K = 5  # Number of harmonics to use. Low dimensionality (2*K params).
    
    # Initialize coefficients with small random values
    rng = np.random.default_rng(42)
    coeffs = rng.normal(0, 0.1, 2*K)
    
    def get_h(coeffs):
        n = np.arange(N)
        h = np.full(N, 0.5)
        for k in range(1, K+1):
            a = coeffs[2*(k-1)]
            b = coeffs[2*(k-1)+1]
            h += a * np.cos(2*np.pi*k*n/N) + b * np.sin(2*np.pi*k*n/N)
        # Clip to [0, 1] and renormalize? 
        # The problem requires h in [0,1] and sum exactly N/2.
        # If we clip, sum changes. If we don't clip, values might go out of bounds.
        # For small coefficients, h stays near 0.5, so bounds are likely respected.
        # We will check bounds and sum.
        # To strictly satisfy sum=N/2, we rely on the construction (DC=0.5).
        # If clipping occurs, we need a projection. 
        # Let's first try without clipping, assuming small amplitudes.
        # If any h[i] < 0 or > 1, we return a penalty.
        return h

    def max_overlap(h):
        # h is real, length N
        # Overlap for shift s: sum(h[i] * (1 - h[(i+s)%N]))
        # = N/2 - sum(h[i] * h[(i+s)%N])  (since sum(1-h)=N/2)
        # We want to minimize max_s overlap.
        # This is equivalent to maximizing min_s sum(h[i]*h[(i+s)%N]).
        # Let's compute all autocorrelations.
        # FFT method for circular cross-correlation.
        # C[s] = sum(h[i] * h[(i+s)%N])
        # This is the circular autocorrelation.
        # Using FFT:
        # H = FFT(h)
        # C = IFFT(|H|^2) ? No, that's aconv.
        # Circular correlation c[s] = sum h[i] h[i+s].
        # In frequency: C = IFFT(FFT(h) * conj(FFT(h)))? 
        # Let's check indices.
        # If h is real, FFT(h) is conjugate symmetric.
        # Let's just do it directly for N=100, it's fast enough.
        # Or use np.fft.
        # Circular autocorrelation r[s] = sum_{i=0}^{N-1} h[i] h[(i+s)%N]
        # This is the same as the inverse FFT of |FFT(h)|^2.
        H = np.fft.fft(h)
        power = H * np.conj(H)
        r = np.fft.ifft(power).real
        # r[0] is sum h[i]^2. r[s] is sum h[i] h[i+s].
        # Note: np.fft.ifft returns unnormalized result? 
        # np.fft.ifft(fft(x)) = x.
        # So r is correct.
        
        overlaps = N/2 - r
        return np.max(overlaps)

    def score_and_grad(coeffs):
        h = get_h(coeffs)
        
        # Check bounds
        if np.any(h < -1e-9) or np.any(h > 1 + 1e-9):
            # Penalty for out of bounds
            penalty = 10.0 * (np.sum(np.maximum(0, -h)**2) + np.sum(np.maximum(0, h-1)**2))
            # Gradient of penalty
            grad_penalty = 20.0 * (-np.minimum(h, 0) + np.maximum(h-1, 0))
            # Approximate gradient of max_overlap is hard analytically with penalty.
            # We will use finite differences for the whole objective if bounds are violated.
            # But let's try to stay in bounds.
            pass
            
        # Compute overlap
        max_ov = max_overlap(h)
        
        # We need gradient of max_overlap.
        # max_ov = max_s (N/2 - r[s])
        # Let s* be the argmax.
        # grad max_ov = - grad r[s*]
        # r[s] = sum_i h[i] h[i+s]
        # d r[s] / d h[j] = h[j+s] + h[j-s] (circular)
        # So grad h = -(h[roll(j, s*)] + h[roll(j, -s*)])?
        # Wait, r[s] = sum_i h[i] h[(i+s)%N].
        # Let u[i] = h[i], v[i] = h[(i+s)%N].
        # r[s] = sum u[i] v[i].
        # d r[s] / d h[j] = v[j] + u[j-s] (since h[j] appears in u[j] and v[j-s] where (j-s+s)%N = j).
        # So d r[s] / d h = np.roll(h, -s) + np.roll(h, s).
        
        # Find s*
        H = np.fft.fft(h)
        power = H * np.conj(H)
        r = np.fft.ifft(power).real
        overlaps = N/2 - r
        s_star = np.argmax(overlaps)
        
        grad_h = -(np.roll(h, -s_star) + np.roll(h, s_star))
        
        # Map gradient from h-space to coeff-space
        # h[n] = 0.5 + sum (a_k cos... + b_k sin...)
        # d h[n] / d a_k = cos(2*pi*k*n/N)
        # d h[n] / d b_k = sin(2*pi*k*n/N)
        
        grad_coeffs = np.zeros(2*K)
        n = np.arange(N)
        for k in range(1, K+1):
            c = np.cos(2*np.pi*k*n/N)
            s = np.sin(2*np.pi*k*n/N)
            grad_coeffs[2*(k-1)] = np.dot(grad_h, c)
            grad_coeffs[2*(k-1)+1] = np.dot(grad_h, s)
            
        # Add penalty gradient if out of bounds
        if np.any(h < 0) or np.any(h > 1):
            # Simple penalty: push back towards [0,1]
            # Gradient of 0.5 * sum((max(0, -h))^2 + (max(0, h-1))^2)
            # = -min(h,0) + max(h-1,0)
            penalty_grad_h = -np.minimum(h, 0) + np.maximum(h-1, 0)
            for k in range(1, K+1):
                c = np.cos(2*np.pi*k*n/N)
                s = np.sin(2*np.pi*k*n/N)
                grad_coeffs[2*(k-1)] += 100.0 * np.dot(penalty_grad_h, c)
                grad_coeffs[2*(k-1)+1] += 100.0 * np.dot(penalty_grad_h, s)
                
        return max_ov, grad_coeffs

    # Gradient Descent
    lr = 0.01
    best_coeffs = coeffs.copy()
    best_score = float('inf')
    
    start_time = time.time()
    deadline = 140.0
    
    # Initial evaluation
    h_init = get_h(coeffs)
    if np.any(h_init < 0) or np.any(h_init > 1):
        # Try to scale down
        scale = 1.0
        while np.any(get_h(coeffs*scale) < 0) or np.any(get_h(coeffs*scale) > 1) and scale > 1e-4:
            scale *= 0.5
        coeffs *= scale
        best_coeffs = coeffs.copy()
        
    current_score, _ = score_and_grad(coeffs)
    best_score = current_score
    best_coeffs = coeffs.copy()
    
    for i in range(5000):
        if time.time() > deadline:
            break
            
        score, grad = score_and_grad(coeffs)
        
        if score < best_score:
            best_score = score
            best_coeffs = coeffs.copy()
            
        # Update
        coeffs -= lr * grad
        
        # Adaptive learning rate
        if i % 100 == 0:
            lr *= 0.95

    # Final check and cleanup
    h_final = get_h(best_coeffs)
    
    # Ensure strict [0,1] bounds and sum N/2
    # If bounds are slightly violated due to numerical issues, clip and adjust?
    # The sum constraint is hard to satisfy if we clip.
    # However, if we stay in bounds, sum is exactly N/2 by construction.
    # Let's verify.
    if np.any(h_final < 0) or np.any(h_final > 1):
        # Fallback to uniform if optimization failed to stay in bounds
        return np.full(N, 0.5)
        
    # Verify sum
    if not np.isclose(np.sum(h_final), N/2):
        # This should not happen with the Fourier construction
        return np.full(N, 0.5)

    return h_final

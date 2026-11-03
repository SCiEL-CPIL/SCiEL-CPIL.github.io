import numpy as np
import time

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initialize
    f = np.ones(N)
    
    # Parameters
    beta = 10.0  # Softmax sharpness
    eta = 0.01   # Learning rate
    top_k = 5    # Use top-k peaks for stability instead of full softmax initially?
                 # Full softmax is smoother. Let's stick to full softmax but maybe adapt beta.
    
    best_f = f.copy()
    best_val = np.inf
    
    # Pre-compute shifts? No, N is small enough.
    
    while time.time() < deadline:
        # Compute autoconvolution
        C = np.convolve(f, f, mode='full')
        
        # Calculate current max
        current_max = np.max(C)
        
        # Track best
        if current_max < best_val:
            best_val = current_max
            best_f = f.copy()
        
        # Compute weights for softmax
        # Subtract max for numerical stability
        C_shifted = C - np.max(C)
        exp_C = np.exp(beta * C_shifted)
        w = exp_C / np.sum(exp_C)
        
        # Compute gradient g = sum_k w_k * grad(C_k)
        # grad(C_k)[i] = 2 * f[k-i] if 0 <= k-i < N else 0
        # This is equivalent to:
        # g = 2 * sum_k w_k * shift(f, k)
        # Where shift(f, k) is f[k-i] at index i.
        # This is the cross-correlation of w and f?
        # Let's compute it efficiently.
        # g[i] = 2 * sum_k w[k] * f[k-i]
        # Let j = k-i => k = i+j
        # g[i] = 2 * sum_j w[i+j] * f[j]
        # This is the cross-correlation of w and f.
        
        # np.correlate(w, f, mode='same')?
        # np.correlate(a, v) computes sum_j a[j+k] * conj(v[j]).
        # We want sum_j w[i+j] f[j].
        # This is correlate(w, f, 'same')?
        # Let's check indices.
        # np.correlate(w, f, 'full')[i] = sum_j w[j] f[j - i + offset]?
        # It's easier to use FFT.
        
        # g = 2 * IFFT( FFT(w) * conj(FFT(f)) )?
        # Let's verify.
        # W = FFT(w), F = FFT(f).
        # IFFT(W * conj(F))[i] = sum_k W[k] conj(F[k]) e^{2pi i k i / N} / N
        # This is the circular cross-correlation.
        # Since N=600 and conv length is 1199, circular might wrap around.
        # But f is supported on [0, N-1].
        # Let's just use the direct sum if it's fast enough or FFT with zero padding.
        
        # Zero padding to 2*N to avoid circular wrap-around.
        size = 2 * N
        W = np.fft.fft(w, size=size)
        F = np.fft.fft(f, size=size)
        
        # Cross correlation
        # We want g[i] = sum_k w[k] f[k-i].
        # This is the cross-correlation of w and f.
        # In FFT domain: IFFT( W * conj(F) )?
        # Let's test with small N.
        # w=[1,0], f=[1,1].
        # g[0] = w[0]f[0] + w[1]f[1] = 1*1 + 0 = 1.
        # g[1] = w[1]f[0] + w[2]f[1] = 0.
        # g[2] = w[2]f[0] ... = 0.
        # IFFT(FFT([1,0]) * conj(FFT([1,1])))
        # FFT([1,0,0,0]) = [1,1,1,1]
        # FFT([1,1,0,0]) = [2, 0, -2, 0] -> conj [2, 0, -2, 0]
        # Prod = [2, 0, -2, 0]
        # IFFT = [0.5, 0, -0.5, 0]
        # This doesn't match.
        
        # The formula for cross-correlation c[i] = sum_j a[j+i] b[j] is IFFT( A * conj(B) ).
        # Here we want sum_k w[k] f[k-i].
        # Let j = k-i => k = i+j.
        # sum_j w[i+j] f[j].
        # This is exactly the cross-correlation of w and f at lag i.
        # So g = 2 * IFFT( W * conj(F) ).
        
        # Let's re-verify the FFT identity.
        # x[i] = sum_j a[j+i] b[j]
        # X = FFT(x).
        # It is known that FFT(corr(a,b)) = A * conj(B).
        # So corr(a,b) = IFFT(A * conj(B)).
        # Here a=w, b=f.
        # So g = 2 * IFFT(W * conj(F)).
        
        grad = 2 * np.real(np.fft.ifft(W * np.conj(F)))[:N]
        
        # Update f using Mirror Descent
        # f = f * exp(-eta * grad)
        # To avoid underflow/overflow, we can normalize.
        f = f * np.exp(-eta * grad)
        
        # Normalize to keep scale manageable (e.g., sum to 1)
        s = np.sum(f)
        if s > 0:
            f = f / s
        else:
            break
            
        # Check time
        if time.time() > deadline - 5.0:
            break

    # Return the best found
    return list(best_f)


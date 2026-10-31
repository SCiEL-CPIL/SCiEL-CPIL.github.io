import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def max_overlap(h):
        # O_k = N/2 - sum_i h[i]*h[(i+k) mod N]
        # min_k corr = min_k sum_i h[i]*h[(i+k) mod N]
        # overlap_max = N/2 - min_k corr[k]
        # Use FFT for circular autocorrelation
        fft_h = np.fft.fft(h, n=2*N)
        corr = np.fft.ifft(fft_h * np.conj(fft_h)).real[:N]
        return N/2 - np.min(corr)

    def autocorr_var(h):
        # Variance of autocorrelation as smooth proxy
        fft_h = np.fft.fft(h, n=2*N)
        corr = np.fft.ifft(fft_h * np.conj(fft_h)).real[:N]
        return np.var(corr)

    def project(h):
        # Project to sum = N/2 and [0,1]
        h = np.clip(h, 0, 1)
        s = h.sum()
        if s > 0:
            h = h / s * (N / 2)
        h = np.clip(h, 0, 1)
        return h

    def anneal(h0, budget):
        h = h0.copy()
        T = 0.05
        lr = 0.01
        best_h = h.copy()
        best_score = max_overlap(h)
        step = 0
        while time.time() < budget:
            step += 1
            # Compute autocorrelation
            fft_h = np.fft.fft(h, n=2*N)
            corr = np.fft.ifft(fft_h * np.conj(fft_h)).real[:N]
            
            # Find the shift with minimum correlation (max overlap)
            k_min = np.argmin(corr)
            
            # Gradient of -corr[k_min] w.r.t. h[j]
            # d/dh[j] corr[k_min] = h[(j+k_min) % N] + h[(j-k_min) % N]
            # So gradient of overlap = -(h[(j+k_min)%N] + h[(j-k_min)%N])
            idx = np.arange(N)
            grad = -(h[(idx + k_min) % N] + h[(idx - k_min) % N])
            
            # Also add a small term to flatten autocorrelation (reduce variance)
            # d/dh[j] var(corr) ~ 2 * (corr - mean(corr)) * d(corr)/dh[j]
            corr_dev = corr - np.mean(corr)
            # d(corr[k])/dh[j] = h[(j+k)%N] + h[(j-k)%N] for each k
            # This is expensive; use a simplified version
            # Instead, add a small regularization toward 0.5
            reg = 0.01 * (h - 0.5)
            
            # Annealing: random perturbation
            noise = T * (np.random.random(N) - 0.5)
            
            h_new = h - lr * (grad + reg) + noise
            h_new = project(h_new)
            
            score = max_overlap(h_new)
            if score < best_score:
                best_score = score
                best_h = h_new.copy()
            
            # Accept/reject with annealing
            if score < best_score or np.random.random() < np.exp(-(score - best_score) / max(T, 1e-6)):
                h = h_new
            else:
                h = project(h + 0.5 * (h - h_new))
            
            # Cool down
            T *= 0.999
            if T < 0.001:
                T = 0.001
                lr = max(0.002, lr * 0.95)
        
        return best_h, best_score

    best_h = None
    best_score = float('inf')

    # Generate diverse initializers
    initializers = []
    
    # 1. Constant
    initializers.append(np.full(N, 0.5))
    
    # 2. Various sinusoids
    for freq in [1, 2, 3, 5, 7, 10, 13, 17, 20, 25]:
        for amp in [0.3, 0.4, 0.45]:
            h = 0.5 + amp * np.sin(2 * np.pi * freq * np.arange(N) / N)
            initializers.append(project(h))
    
    # 3. Random phase in frequency domain (structured random)
    np.random.seed(42)
    for trial in range(10):
        # Random magnitudes close to flat, random phases
        freq_idx = np.arange(N)
        target_mag = np.ones(N) * 0.3
        # Keep DC component for mean 0.5
        target_mag[0] = N / 2
        # Add some variation
        target_mag[1:] *= (1 + 0.3 * (np.random.random(N-1) - 0.5))
        phases = 2 * np.pi * np.random.random(N)
        H = target_mag * np.exp(1j * phases)
        # Ensure real: H[-k] = conj(H[k])
        H_real = np.zeros(N, dtype=complex)
        H_real[0] = H[0].real
        for k in range(1, N//2 + 1):
            H_real[k] = H[k]
            H_real[N-k] = np.conj(H[k])
        h = np.fft.ifft(H_real).real
        initializers.append(project(h))
    
    # 4. Block patterns
    for block_size in [2, 3, 4, 5, 10]:
        h = np.zeros(N)
        for i in range(0, N, 2*block_size):
            h[i:i+block_size] = 1.0
        initializers.append(project(h))
    
    # 5. Random
    for trial in range(5):
        h = np.random.random(N)
        initializers.append(project(h))

    # Run annealing on each initializer with time budget
    n_init = len(initializers)
    per_init_time = 130.0 / n_init
    
    for i, h0 in enumerate(initializers):
        remaining = deadline - time.time()
        if remaining < 5:
            break
        alloc = min(per_init_time, remaining - 5)
        if alloc < 1:
            break
        h_res, score = anneal(h0, time.time() + alloc)
        if score < best_score:
            best_score = score
            best_h = h_res

    # Final local refinement on the best
    if best_h is not None and time.time() < deadline - 10:
        h = best_h.copy()
        lr = 0.003
        for step in range(5000):
            if time.time() > deadline - 5:
                break
            fft_h = np.fft.fft(h, n=2*N)
            corr = np.fft.ifft(fft_h * np.conj(fft_h)).real[:N]
            k_min = np.argmin(corr)
            idx = np.arange(N)
            grad = -(h[(idx + k_min) % N] + h[(idx - k_min) % N])
            h = h - lr * grad
            h = project(h)
        
        score = max_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h

    if best_h is None:
        return np.full(N, 0.5)

    # Ensure exact sum
    best_h = project(best_h)
    # Fix sum exactly
    s = best_h.sum()
    if abs(s - N/2) > 1e-10:
        diff = N/2 - s
        # Distribute difference
        if diff > 0:
            room = 1.0 - best_h
            mask = room > 1e-9
            if mask.sum() > 0:
                add = diff / mask.sum()
                best_h[mask] = np.minimum(best_h[mask] + add, 1.0)
        else:
            mask = best_h > 1e-9
            if mask.sum() > 0:
                sub = -diff / mask.sum()
                best_h[mask] = np.maximum(best_h[mask] - sub, 0.0)
        # Re-project if needed
        best_h = project(best_h)

    return best_h

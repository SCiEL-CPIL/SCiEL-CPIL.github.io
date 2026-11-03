"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 145.0  # safety margin before 150s kill

    dx = 1.0 / (2 * N)  # interval width: [-1/4, 1/4] has length 1/2, N intervals

    def autoconv_max_ratio(h):
        """Compute max(autoconv) / ||h||_2^2 for step function h on N equal intervals."""
        h = np.asarray(h, dtype=np.float64)
        if np.any(h < 0):
            h = np.maximum(h, 0)
        if np.sum(h) < 1e-15:
            return float('inf')
        
        # Use FFT for efficient autoconvolution
        n_fft = 2 * N
        h_padded = np.zeros(n_fft)
        h_padded[:N] = h
        
        H = np.fft.fft(h_padded)
        autoconv_circ = np.fft.ifft(H * H).real
        
        # The linear autoconvolution: indices 0..N-1 are lags 0..(N-1),
        # indices N..2N-1 are lags -(N-1)..-1
        autoconv_vals = autoconv_circ * dx**2
        max_autoconv = np.max(autoconv_vals)
        
        l2_sq = np.sum(h**2) * dx
        if l2_sq < 1e-15:
            return float('inf')
        
        return max_autoconv / l2_sq

    def gradient_descent_step(h, score, lr=0.01, max_iter=5000):
        """Gradient descent with subgradient approximation for the max autoconv term."""
        h = np.asarray(h, dtype=np.float64).copy()
        current_score = autoconv_max_ratio(h)
        
        # Precompute FFT setup
        n_fft = 2 * N
        h_padded = np.zeros(n_fft)
        h_padded[:N] = h
        H = np.fft.fft(h_padded)
        autoconv_circ = np.fft.ifft(H * H).real
        autoconv_vals = autoconv_circ * dx**2
        max_autoconv = np.max(autoconv_vals)
        l2_sq = np.sum(h**2) * dx
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Find the lag where max autoconv occurs
            max_idx = np.argmax(autoconv_circ)
            
            # Compute gradient of max_autoconv w.r.t. h
            # autoconv[k] = sum_i h[i] * h[i-k] (approximately, for circular)
            # d(autoconv[k])/d(h[j]) = h[j-k] + h[j+k] (for linear conv, with boundary handling)
            # For circular conv of size n_fft, this is cleaner
            
            # For circular convolution: (h*h)[k] = sum_j h[j] h[(k-j) mod n_fft]
            # d/dh[j] = h[(k-j) mod n_fft]
            
            # But we only care about the first N elements of h (the rest are zero)
            # So gradient w.r.t. h[j] for j in 0..N-1:
            # d(max_autoconv)/d(h[j]) = h[(max_idx - j) mod n_fft] + (if max_idx >= N, also from the wrap)
            
            # Actually for circular conv: d(A[k])/d(h[j]) where A = ifft(fft(h)^2)
            # A[k] = sum_m h[m] h[(k-m) mod n_fft]
            # dA[k]/dh[j] = h[(k-j) mod n_fft] + h[j] * delta_{k,0}... no
            # Actually: A = h ⊛ h (circular)
            # dA[k]/dh[j] = h[(k-j) mod n_fft]
            
            grad_autoconv = np.zeros(n_fft)
            k = max_idx
            for j in range(N):
                grad_autoconv[j] = h_padded[(k - j) % n_fft]
            
            # Gradient of l2_sq = sum(h[i]^2) * dx
            grad_l2 = 2 * h * dx
            
            # Score = max_autoconv / l2_sq
            # d(score)/d(h[j]) = (grad_autoconv[j] * l2_sq - max_autoconv * grad_l2[j]) / l2_sq^2
            # We want to minimize, so gradient descent: h -= lr * grad_score
            
            grad_score = (grad_autoconv[:N] * l2_sq - max_autoconv * grad_l2) / (l2_sq ** 2)
            
            # Project gradient descent with nonnegativity constraint
            h_new = h - lr * grad_score
            h_new = np.maximum(h_new, 0)
            
            # Check if it improved
            new_score = autoconv_max_ratio(h_new)
            if new_score < current_score:
                h = h_new
                current_score = new_score
                # Update precomputed values
                h_padded = np.zeros(n_fft)
                h_padded[:N] = h
                H = np.fft.fft(h_padded)
                autoconv_circ = np.fft.ifft(H * H).real
                autoconv_vals = autoconv_circ * dx**2
                max_autoconv = np.max(autoconv_vals)
                l2_sq = np.sum(h**2) * dx
            else:
                # Reduce learning rate
                lr *= 0.95
                if lr < 1e-8:
                    break
        
        return h, current_score

    def annealing_refine(h, max_iters=2000):
        """Simulated annealing style refinement."""
        h = np.asarray(h, dtype=np.float64).copy()
        current_score = autoconv_max_ratio(h)
        best_h = h.copy()
        best_score = current_score
        temperature = 0.01
        cooling = 0.995
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random coordinate or small set of coordinates to perturb
            n_perturb = np.random.randint(1, 6)
            indices = np.random.choice(N, size=n_perturb, replace=False)
            
            h_trial = h.copy()
            for idx in indices:
                delta = np.random.normal(0, 0.05)
                h_trial[idx] = max(0, h[idx] + delta)
            
            trial_score = autoconv_max_ratio(h_trial)
            
            # Accept if better, or with probability exp(-(delta_score)/temperature)
            if trial_score < current_score:
                h = h_trial
                current_score = trial_score
                if current_score < best_score:
                    best_h = h.copy()
                    best_score = current_score
            else:
                delta_s = trial_score - current_score
                if np.random.rand() < np.exp(-delta_s / temperature):
                    h = h_trial
                    current_score = trial_score
            
            temperature *= cooling
        
        return best_h, best_score

    best_score = float('inf')
    best_h = None

    centers = np.arange(N) - N // 2  # centered indices: -300, -299, ..., 299

    # Generate diverse initial candidates
    candidates = []
    
    # 1. Gaussian peaks with various widths
    for sigma in [1, 2, 5, 10, 20, 50, 100, 200]:
        h = np.exp(-centers**2 / (2 * sigma**2))
        candidates.append(h)
    
    # 2. Triangular peaks
    for width in [5, 10, 20, 50, 100, 200, 300]:
        h = np.maximum(0, 1 - np.abs(centers) / width)
        candidates.append(h)
    
    # 3. Box functions (constant on central region)
    for half_width in [5, 10, 20, 50, 100, 200, 300]:
        h = np.zeros(N)
        mask = np.abs(centers) <= half_width
        h[mask] = 1.0
        candidates.append(h)
    
    # 4. Gaussian + base
    for sigma in [2, 5, 10, 20, 50, 100]:
        for base in [0.0, 0.05, 0.1, 0.2, 0.3]:
            h = base + (1 - base) * np.exp(-centers**2 / (2 * sigma**2))
            candidates.append(h)
    
    # 5. Inverted Gaussian (valley in center, high at edges)
    for sigma in [50, 100, 200]:
        h = 1.0 - np.exp(-centers**2 / (2 * sigma**2))
        h = np.maximum(h, 0)
        candidates.append(h)
    
    # 6. Two-sided peaks (peaks at both ends)
    for sigma in [5, 10, 20, 50]:
        h = np.exp(-centers**2 / (2 * sigma**2)) + np.exp(-(centers - N/2)**2 / (2 * sigma**2)) + np.exp(-(centers + N/2)**2 / (2 * sigma**2))
        candidates.append(h)
    
    # 7. Random smooth functions
    np.random.seed(42)
    for trial in range(10):
        freqs = np.random.uniform(0.5, 10.0, size=5)
        phases = np.random.uniform(0, 2*np.pi, size=5)
        amps = np.random.uniform(0.5, 2.0, size=5)
        h = np.ones(N)
        for f, p, a in zip(freqs, phases, amps):
            h += a * np.cos(f * centers / N * 2 * np.pi + p)
        h = np.maximum(h, 0)
        if np.sum(h) > 1e-10:
            candidates.append(h)
    
    # 8. Very narrow central peak (delta-like)
    for width in [1, 2, 3, 5]:
        h = np.zeros(N)
        center_idx = N // 2
        h[max(0, center_idx - width):min(N, center_idx + width + 1)] = 1.0
        candidates.append(h)

    # Evaluate all candidates
    for h in candidates:
        if time.time() > deadline:
            break
        h = np.asarray(h, dtype=np.float64)
        if np.any(h < 0):
            h = np.maximum(h, 0)
        if np.sum(h) < 1e-10:
            continue
        score = autoconv_max_ratio(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()

    # Refine the best candidate with gradient descent
    if best_h is not None and time.time() < deadline - 5:
        refined_h, refined_score = gradient_descent_step(best_h, best_score, lr=0.05, max_iter=10000)
        if refined_score < best_score:
            best_score = refined_score
            best_h = refined_h

    # Additional refinement with annealing
    if best_h is not None and time.time() < deadline - 5:
        refined_h, refined_score = annealing_refine(best_h, max_iters=5000)
        if refined_score < best_score:
            best_score = refined_score
            best_h = refined_h

    # Try gradient descent from a few other good candidates too
    if time.time() < deadline - 10:
        # Re-evaluate top candidates
        scored_candidates = []
        for h in candidates:
            h = np.asarray(h, dtype=np.float64)
            if np.any(h < 0):
                h = np.maximum(h, 0)
            if np.sum(h) < 1e-10:
                continue
            score = autoconv_max_ratio(h)
            scored_candidates.append((score, h))
        
        scored_candidates.sort(key=lambda x: x[0])
        
        # Try gradient descent on the top 5 candidates
        for score, h in scored_candidates[:5]:
            if time.time() > deadline - 3:
                break
            refined_h, refined_score = gradient_descent_step(h, score, lr=0.05, max_iter=5000)
            if refined_score < best_score:
                best_score = refined_score
                best_h = refined_h

    if best_h is None:
        return [1.0] * N

    return list(best_h)

# EVOLVE-BLOCK-END

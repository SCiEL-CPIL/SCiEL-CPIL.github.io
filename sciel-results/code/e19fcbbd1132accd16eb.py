import numpy as np
import time

N_FFT = 16384

def _fast_supnorm_fft(c):
    padded = np.zeros(N_FFT, dtype=complex)
    padded[:len(c)] = c
    vals = np.fft.fft(padded)
    return np.max(np.abs(vals))

def _accurate_supnorm(c):
    n_samples = 20000
    theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    k = np.arange(len(c))
    # Use matrix multiplication for speed
    # Z is n_samples x len(c)
    # We can use FFT for the dense sampling part too?
    # FFT gives 16384 points. 20000 points is similar.
    # Let's just use direct evaluation for the 20000 points, it's fast enough.
    # Or use FFT with N=32768 for the dense part.
    # Let's stick to direct for simplicity and accuracy in the refinement step.
    z = np.exp(1j * theta)
    powers = z[:, None] ** k[None, :]
    vals = powers @ c
    abs_vals = np.abs(vals)
    max_val = np.max(abs_vals)
    
    idx = np.argsort(abs_vals)[::-1][:10]
    best_refined = max_val
    gr = (np.sqrt(5) - 1) / 2
    
    def eval_at_angle(ang):
        zz = np.exp(1j * ang)
        powers = zz ** k
        val = np.sum(c * powers)
        return np.abs(val)
        
    for i in idx:
        delta = 2 * np.pi / n_samples
        a = theta[i] - delta
        b = theta[i] + delta
        c_pt = b - gr * (b - a)
        d_pt = a + gr * (b - a)
        fc = eval_at_angle(c_pt)
        fd = eval_at_angle(d_pt)
        for _ in range(40):
            if fc < fd:
                b = d_pt
                d_pt = c_pt
                fd = fc
                c_pt = b - gr * (b - a)
                fc = eval_at_angle(c_pt)
            else:
                a = c_pt
                c_pt = d_pt
                fc = fd
                d_pt = a + gr * (b - a)
                fd = eval_at_angle(d_pt)
        refined_val = max(fc, fd)
        if refined_val > best_refined:
            best_refined = refined_val
    return best_refined

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    rng = np.random.default_rng(42)
    
    # Initialize
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:24], dtype=int)
    best_fft_score = _fast_supnorm_fft(best_coeffs)
    
    # List of top candidates (coeffs, fft_score)
    top_candidates = [(best_coeffs.copy(), best_fft_score)]
    
    current_coeffs = best_coeffs.copy()
    current_fft_score = best_fft_score
    
    while time.time() < deadline:
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            indices = np.arange(24)
            rng.shuffle(indices)
            for i in indices:
                if time.time() > deadline:
                    break
                old_val = current_coeffs[i]
                current_coeffs[i] = -old_val
                new_fft_score = _fast_supnorm_fft(current_coeffs)
                if new_fft_score < current_fft_score:
                    current_fft_score = new_fft_score
                    improved = True
                else:
                    current_coeffs[i] = old_val
        
        # Add to top candidates if good
        # Simple check: if it's better than the worst in the top 10, add it
        if len(top_candidates) < 10:
            top_candidates.append((current_coeffs.copy(), current_fft_score))
        else:
            # Find worst
            worst_idx = np.argmax([s for _, s in top_candidates])
            if current_fft_score < top_candidates[worst_idx][1]:
                top_candidates[worst_idx] = (current_coeffs.copy(), current_fft_score)
        
        # Sort top candidates by FFT score
        top_candidates.sort(key=lambda x: x[1])
        # Keep only top 5 to save memory/computation?
        # Just keeping 10 is fine.
        
        # Perturb
        n_perturb = rng.integers(2, 8)
        flip_indices = rng.choice(24, size=n_perturb, replace=False)
        current_coeffs[flip_indices] *= -1
        current_fft_score = _fast_supnorm_fft(current_coeffs)
        
        # If too bad, reset to best
        if current_fft_score > top_candidates[0][1] * 1.2:
            current_coeffs = top_candidates[0][0].copy()
            current_fft_score = top_candidates[0][1]
            # Perturb again
            n_perturb = rng.integers(2, 6)
            flip_indices = rng.choice(24, size=n_perturb, replace=False)
            current_coeffs[flip_indices] *= -1
            current_fft_score = _fast_supnorm_fft(current_coeffs)

    # Final Evaluation
    # Evaluate the top candidates with the accurate method
    best_accurate_score = float('inf')
    best_final_coeffs = top_candidates[0][0]
    
    for coeffs, _ in top_candidates[:5]:
        if time.time() > deadline - 5:
            break
        acc_score = _accurate_supnorm(coeffs)
        if acc_score < best_accurate_score:
            best_accurate_score = acc_score
            best_final_coeffs = coeffs

    return best_final_coeffs.tolist()

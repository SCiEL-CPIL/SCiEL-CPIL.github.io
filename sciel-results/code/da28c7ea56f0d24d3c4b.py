import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    candidates = [
        [1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1],
        [1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1],
        [1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1],
        [1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1],
        [1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1],
    ]
    
    M = 2**15
    k = np.arange(M)
    n_idx = np.arange(24)
    phases = np.outer(k, n_idx)
    D = np.exp(-2j * np.pi * phases / M)
    
    def local_search_1flip(current_c, current_f, current_max):
        """Thorough 1-flip local search until no improvement."""
        improved = True
        while improved:
            improved = False
            best_flip = -1
            best_max_val = current_max
            for n in range(24):
                f_new = current_f - 2.0 * current_c[n] * D[:, n]
                max_val = np.max(np.abs(f_new))
                if max_val < best_max_val - 1e-12:
                    best_max_val = max_val
                    best_flip = n
            if best_flip != -1:
                current_c[best_flip] *= -1.0
                current_f = D @ current_c
                current_max = best_max_val
                improved = True
        return current_c, current_f, current_max
    
    def local_search_2flip(current_c, current_f, current_max):
        """Try 2-flips to escape local minima."""
        best_flip_2 = (-1, -1)
        best_max_2 = current_max
        for n1 in range(24):
            for n2 in range(n1 + 1, 24):
                f_new = current_f - 2.0 * (current_c[n1] * D[:, n1] + current_c[n2] * D[:, n2])
                max_val = np.max(np.abs(f_new))
                if max_val < best_max_2 - 1e-12:
                    best_max_2 = max_val
                    best_flip_2 = (n1, n2)
        if best_flip_2 != (-1, -1):
            n1, n2 = best_flip_2
            current_c[n1] *= -1.0
            current_c[n2] *= -1.0
            current_f = D @ current_c
            current_max = best_max_2
        return current_c, current_f, current_max
    
    def spectral_synthesis(target_type='flattened', taper_power=2.0):
        """Create a globally coherent starting point by inverting a target waveform.
        
        The idea: design a target frequency response with low peaks, then find
        the closest ±1 sequence via projection. This bypasses local minima traps.
        """
        # Design target waveform in frequency domain
        if target_type == 'flattened':
            # Try to make |P(e^{iθ})| as flat as possible
            # Start with a constant magnitude, then refine
            target_mag = np.ones(M)
        elif target_type == 'cosine_tapered':
            # Cosine taper to reduce sidelobes
            theta = 2 * np.pi * k / M
            target_mag = 1.0 + 0.1 * np.cos(theta) ** taper_power
        elif target_type == 'gaussian_tapered':
            # Gaussian window in frequency
            theta = 2 * np.pi * k / M
            sigma = 0.3
            target_mag = np.exp(-theta ** 2 / (2 * sigma ** 2))
        else:
            target_mag = np.ones(M)
        
        # Assign phases to minimize peak
        # Use iterative phase retrieval (Gerchberg-Saxton style)
        # Start with random phases
        rng = np.random.RandomState(42)
        phase = rng.uniform(0, 2 * np.pi, M)
        
        for iter in range(50):
            if time.time() > deadline - 30:
                break
            
            # Compute current spectrum
            spectrum = target_mag * np.exp(1j * phase)
            
            # Inverse FFT to get time domain
            time_domain = np.fft.ifft(spectrum).real
            time_domain = time_domain / np.max(np.abs(time_domain))
            
            # Project to ±1
            projected = np.sign(time_domain)
            projected[projected == 0] = 1.0
            
            # Forward FFT to get new spectrum
            new_spectrum = np.fft.fft(projected)
            
            # Keep magnitude from target, update phase
            phase = np.angle(new_spectrum)
        
        # Final projection
        final_spectrum = target_mag * np.exp(1j * phase)
        final_time = np.fft.ifft(final_spectrum).real
        final_time = final_time / np.max(np.abs(final_time))
        coeffs = np.sign(final_time)
        coeffs[coeffs == 0] = 1.0
        
        return coeffs
    
    def continuous_relaxation(start_c, max_iters=200, lr_init=0.1):
        """Optimize over real coefficients in [-1,1] using gradient descent
        on a smooth surrogate (log-sum-exp) of the sup-norm, then project."""
        x = start_c.astype(float).copy()
        lr = lr_init
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            F = D @ x
            abs_F = np.abs(F)
            
            # Smooth max via log-sum-exp with temperature
            beta = 50.0 + 50.0 * (it / max_iters)  # increase sharpness
            scaled = beta * abs_F
            max_scaled = np.max(scaled)
            logsumexp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
            f_val = logsumexp / beta
            
            # Gradient of log-sum-exp wrt x
            weights = np.exp(scaled - max_scaled)
            weights /= np.sum(weights)
            
            # Gradient: beta * D^H * (weights * sign(F))
            grad_F = weights * np.sign(F)  # shape (M,)
            grad_x = D.conj().T @ (grad_F * beta)
            grad_x = grad_x.real / M  # normalize
            
            # Gradient of penalty for x in [-1,1]: push towards ±1
            penalty_grad = -2.0 * x
            
            total_grad = grad_x + 0.1 * penalty_grad
            
            # Update with adaptive learning rate
            x -= lr * total_grad
            x = np.clip(x, -1.0, 1.0)
            
            # Decay learning rate
            lr *= 0.995
            
            # Check convergence
            if it > 0 and abs(f_val - prev_f) < 1e-10:
                break
            prev_f = f_val
        
        # Project to ±1
        x_proj = np.sign(x)
        x_proj[x_proj == 0] = 1.0
        return x_proj
    
    best_score = float('inf')
    best_c = None
    
    # Phase 1: Evaluate candidates with local search
    for c in candidates:
        if time.time() > deadline:
            break
        c_arr = np.array(c, dtype=float)
        f_arr = D @ c_arr
        max_val = np.max(np.abs(f_arr))
        
        c_arr, f_arr, max_val = local_search_1flip(c_arr, f_arr, max_val)
        
        for _ in range(5):
            if time.time() > deadline:
                break
            c_arr, f_arr, max_val = local_search_2flip(c_arr, f_arr, max_val)
            c_arr, f_arr, max_val = local_search_1flip(c_arr, f_arr, max_val)
        
        score = max_val / np.sqrt(24)
        if score < best_score:
            best_score = score
            best_c = c_arr.copy()
    
    # Phase 2: Spectral synthesis - create globally coherent starting points
    if time.time() < deadline - 60:
        # Try different target waveforms
        target_types = ['flattened', 'cosine_tapered', 'gaussian_tapered']
        for target_type in target_types:
            if time.time() > deadline - 30:
                break
            
            # Generate spectral synthesis candidate
            spec_c = spectral_synthesis(target_type)
            spec_c = np.array(spec_c, dtype=float)
            
            f_spec = D @ spec_c
            max_spec = np.max(np.abs(f_spec))
            
            # Polish with local search
            spec_c, f_spec, max_spec = local_search_1flip(spec_c, f_spec, max_spec)
            
            for _ in range(3):
                if time.time() > deadline - 15:
                    break
                spec_c, f_spec, max_spec = local_search_2flip(spec_c, f_spec, max_spec)
                spec_c, f_spec, max_spec = local_search_1flip(spec_c, f_spec, max_spec)
            
            score_spec = max_spec / np.sqrt(24)
            if score_spec < best_score:
                best_score = score_spec
                best_c = spec_c.copy()
    
    # Phase 3: Continuous relaxation from best candidate
    if time.time() < deadline - 30:
        x_relaxed = continuous_relaxation(best_c, max_iters=300, lr_init=0.1)
        f_relaxed = D @ x_relaxed
        max_relaxed = np.max(np.abs(f_relaxed))
        
        # Project and polish
        x_proj = np.sign(x_relaxed)
        x_proj[x_proj == 0] = 1.0
        f_proj = D @ x_proj
        max_proj = np.max(np.abs(f_proj))
        
        x_proj, f_proj, max_proj = local_search_1flip(x_proj, f_proj, max_proj)
        for _ in range(3):
            if time.time() > deadline - 10:
                break
            x_proj, f_proj, max_proj = local_search_2flip(x_proj, f_proj, max_proj)
            x_proj, f_proj, max_proj = local_search_1flip(x_proj, f_proj, max_proj)
        
        score_proj = max_proj / np.sqrt(24)
        if score_proj < best_score:
            best_score = score_proj
            best_c = x_proj.copy()
    
    # Phase 4: Iterated local search with random perturbations
    if time.time() < deadline - 10:
        current_c = best_c.copy()
        current_f = D @ current_c
        current_max = np.max(np.abs(current_f))
        
        rng = np.random.RandomState(42)
        
        while time.time() < deadline:
            # Try 1-flip
            best_flip_1 = -1
            best_max_1 = current_max
            for n in range(24):
                f_new = current_f - 2.0 * current_c[n] * D[:, n]
                max_val = np.max(np.abs(f_new))
                if max_val < best_max_1 - 1e-12:
                    best_max_1 = max_val
                    best_flip_1 = n
            
            if best_flip_1 != -1:
                current_c[best_flip_1] *= -1.0
                current_f = D @ current_c
                current_max = best_max_1
                if current_max / np.sqrt(24) < best_score:
                    best_score = current_max / np.sqrt(24)
                    best_c = current_c.copy()
                continue
            
            # Try 2-flip
            best_flip_2 = (-1, -1)
            best_max_2 = current_max
            for n1 in range(24):
                for n2 in range(n1 + 1, 24):
                    f_new = current_f - 2.0 * (current_c[n1] * D[:, n1] + current_c[n2] * D[:, n2])
                    max_val = np.max(np.abs(f_new))
                    if max_val < best_max_2 - 1e-12:
                        best_max_2 = max_val
                        best_flip_2 = (n1, n2)
            
            if best_flip_2 != (-1, -1):
                n1, n2 = best_flip_2
                current_c[n1] *= -1.0
                current_c[n2] *= -1.0
                current_f = D @ current_c
                current_max = best_max_2
                if current_max / np.sqrt(24) < best_score:
                    best_score = current_max / np.sqrt(24)
                    best_c = current_c.copy()
                continue
            
            # Stuck: try continuous relaxation from current point
            if time.time() < deadline - 15:
                x_rel = continuous_relaxation(current_c, max_iters=100, lr_init=0.05)
                x_p = np.sign(x_rel)
                x_p[x_p == 0] = 1.0
                f_p = D @ x_p
                m_p = np.max(np.abs(f_p))
                x_p, f_p, m_p = local_search_1flip(x_p, f_p, m_p)
                if m_p / np.sqrt(24) < best_score:
                    best_score = m_p / np.sqrt(24)
                    best_c = x_p.copy()
                    current_c = best_c.copy()
                    current_f = D @ current_c
                    current_max = np.max(np.abs(current_f))
                    continue
            
            # Random restart
            n_flips = rng.choice([2, 3], p=[0.6, 0.4])
            indices = rng.choice(24, n_flips, replace=False)
            current_c[indices] *= -1.0
            current_f = D @ current_c
            current_max = np.max(np.abs(current_f))
            
            current_c, current_f, current_max = local_search_1flip(current_c, current_f, current_max)
            if current_max / np.sqrt(24) < best_score:
                best_score = current_max / np.sqrt(24)
                best_c = current_c.copy()
    
    return [int(x) for x in best_c]

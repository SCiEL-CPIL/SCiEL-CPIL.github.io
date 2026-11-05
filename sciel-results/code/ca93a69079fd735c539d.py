import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    # Known good candidates from literature (Odlyzko tables, Golay sequences, etc.)
    candidates = [
        [1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1],
        [1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1],
        [1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1],
        [1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1],
        [1, 1, -1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1],
        # Additional candidates: Golay-Rudin-Shapiro and variants
        [1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -5, -1, -1, 1, 1, -1, 1, 1, -1, -1],
        [1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1],
        [1, 1, -1, 1, -1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1],
    ]

    M = 2**15  # 32768 points: good balance of accuracy and speed
    k = np.arange(M)
    n_idx = np.arange(24)
    phases = np.outer(k, n_idx)
    D = np.exp(-2j * np.pi * phases / M)
    sqrt24 = np.sqrt(24.0)

    def evaluate_full(coeffs):
        """Evaluate sup-norm using full DFT array."""
        F = D @ coeffs
        return np.max(np.abs(F)) / sqrt24

    def golden_section_refine(coeffs, peak_indices, n_refine=50):
        """Refine peak locations using golden-section search."""
        # Evaluate at dense points near each peak
        best_val = 0.0
        for idx in peak_indices:
            theta_center = 2.0 * np.pi * idx / M
            # Search in a small window around the peak
            window = 6.0 / M  # ~3 sample points on each side for better capture
            a = theta_center - window
            b = theta_center + window
            gr = (np.sqrt(5.0) - 1.0) / 2.0
            c_point = b - gr * (b - a)
            d_point = a + gr * (b - a)

            def eval_at(theta):
                z = np.exp(1j * theta)
                # Horner's method for polynomial evaluation
                val = 0.0
                for ck in reversed(coeffs):
                    val = val * z + ck
                return abs(val)

            fc = eval_at(c_point)
            fd = eval_at(d_point)
            for _ in range(n_refine):
                if fc < fd:
                    b = d_point
                    d_point = c_point
                    fd = fc
                    c_point = b - gr * (b - a)
                    fc = eval_at(c_point)
                else:
                    a = c_point
                    c_point = d_point
                    fc = fd
                    d_point = a + gr * (b - a)
                    fd = eval_at(d_point)
            local_max = max(fc, fd)
            if local_max > best_val:
                best_val = local_max
        return best_val / sqrt24

    def evaluate_accurate(coeffs):
        """Evaluate with golden-section refinement of top peaks."""
        F = D @ coeffs
        abs_F = np.abs(F)
        # Get top 15 peak locations for more thorough refinement
        top_k = 15
        peak_indices = np.argsort(abs_F)[-top_k:]
        peak_indices = np.unique(peak_indices)  # Remove duplicates
        refined_max = golden_section_refine(coeffs, peak_indices, n_refine=50)
        return refined_max

    # Spectral initialization: create candidates from inverse FFT of target waveforms
    def spectral_init(target_type='flat', n_variants=5, seed=123):
        """Generate candidates by inverting a target frequency-domain waveform."""
        rng = np.random.RandomState(seed)
        candidates_list = []
        
        for variant in range(n_variants):
            # Create target spectrum with varying characteristics
            if target_type == 'flat':
                # Flat magnitude with random phase
                mag = np.ones(M)
                phase = 2 * np.pi * rng.rand(M)
                target = mag * np.exp(1j * phase)
            elif target_type == 'cosine_taper':
                # Cosine-tapered magnitude to reduce sidelobes
                t = k / M
                mag = 0.5 * (1 + np.cos(2 * np.pi * t))  # Hanning-like
                mag[M//2 - 2:M//2 + 3] = 0  # Zero out DC and Nyquist
                phase = 2 * np.pi * rng.rand(M)
                target = mag * np.exp(1j * phase)
            else:
                # Gaussian-tapered
                t = k / M
                sigma = 0.1 + 0.2 * variant / n_variants
                mag = np.exp(-0.5 * ((t - 0.5) / sigma) ** 2)
                mag[0] = 0
                mag[M//2] = 0
                phase = 2 * np.pi * rng.rand(M)
                target = mag * np.exp(1j * phase)
            
            # Inverse FFT to get time-domain signal
            time_signal = np.fft.ifft(target)
            
            # Threshold to get ±1 coefficients
            real_part = np.real(time_signal[:24])
            coeffs = np.sign(real_part)
            coeffs[coeffs == 0] = 1  # Handle zeros
            
            # Ensure first coefficient is +1 for normalization
            if coeffs[0] < 0:
                coeffs = -coeffs
            
            candidates_list.append(coeffs.astype(float))
        
        return candidates_list

    # Generate spectral candidates
    spectral_candidates = spectral_init('flat', n_variants=8, seed=42)
    spectral_candidates += spectral_init('cosine_taper', n_variants=5, seed=123)
    spectral_candidates += spectral_init('gaussian', n_variants=5, seed=456)

    # Combine all candidates
    all_candidates = [np.array(c, dtype=float) for c in candidates] + spectral_candidates

    best_score = float('inf')
    best_c = None

    for c in all_candidates:
        score = evaluate_accurate(c)
        if score < best_score:
            best_score = score
            best_c = c.copy()

    current_c = best_c.copy()
    current_f = D @ current_c
    current_max = np.max(np.abs(current_f)) / sqrt24

    np.random.seed(42)

    # Iterated Local Search with simulated annealing
    temperature = 0.08
    cooling = 0.998
    min_temp = 0.005

    while time.time() < deadline:
        if temperature < min_temp:
            temperature = 0.08
            # Random restart: flip 4-8 random bits for larger jumps
            n_flips = np.random.randint(4, 9)
            indices = np.random.choice(24, n_flips, replace=False)
            current_c[indices] *= -1.0
            current_f = D @ current_c
            current_max = np.max(np.abs(current_f)) / sqrt24

        # 1. Try 1-bit flips
        best_flip_1 = -1
        best_max_1 = current_max

        for n in range(24):
            f_new = current_f - 2.0 * current_c[n] * D[:, n]
            max_val = np.max(np.abs(f_new)) / sqrt24
            if max_val < best_max_1:
                best_max_1 = max_val
                best_flip_1 = n

        if best_flip_1 != -1:
            current_c[best_flip_1] *= -1.0
            current_f = D @ current_c
            current_max = best_max_1
            if current_max < best_score:
                best_score = current_max
                best_c = current_c.copy()
            temperature *= cooling
            continue

        # 2. Try 2-bit flips if 1-bit failed
        best_flip_2 = (-1, -1)
        best_max_2 = current_max

        for n1 in range(24):
            for n2 in range(n1 + 1, 24):
                f_new = current_f - 2.0 * (current_c[n1] * D[:, n1] + current_c[n2] * D[:, n2])
                max_val = np.max(np.abs(f_new)) / sqrt24
                if max_val < best_max_2:
                    best_max_2 = max_val
                    best_flip_2 = (n1, n2)

        if best_flip_2 != (-1, -1):
            n1, n2 = best_flip_2
            current_c[n1] *= -1.0
            current_c[n2] *= -1.0
            current_f = D @ current_c
            current_max = best_max_2
            if current_max < best_score:
                best_score = current_max
                best_c = current_c.copy()
            temperature *= cooling
            continue

        # 3. Try 3-bit flips if 2-bit failed (deeper escape)
        best_flip_3 = None
        best_max_3 = current_max

        for n1 in range(24):
            for n2 in range(n1 + 1, 24):
                for n3 in range(n2 + 1, 24):
                    f_new = current_f - 2.0 * (current_c[n1] * D[:, n1] + current_c[n2] * D[:, n2] + current_c[n3] * D[:, n3])
                    max_val = np.max(np.abs(f_new)) / sqrt24
                    if max_val < best_max_3:
                        best_max_3 = max_val
                        best_flip_3 = (n1, n2, n3)

        if best_flip_3 is not None:
            n1, n2, n3 = best_flip_3
            current_c[n1] *= -1.0
            current_c[n2] *= -1.0
            current_c[n3] *= -1.0
            current_f = D @ current_c
            current_max = best_max_3
            if current_max < best_score:
                best_score = current_max
                best_c = current_c.copy()
            temperature *= cooling
            continue

        # 4. Stuck: accept worse move with probability (simulated annealing)
        # Try random 3-4 bit flip and accept with probability exp(-delta/T)
        n_flips = np.random.choice([3, 4], p=[0.7, 0.3])
        indices = np.random.choice(24, n_flips, replace=False)
        old_c = current_c.copy()
        old_max = current_max

        current_c[indices] *= -1.0
        current_f = D @ current_c
        new_max = np.max(np.abs(current_f)) / sqrt24

        delta = new_max - old_max
        if delta < 0 or np.random.rand() < np.exp(-delta / max(temperature, 1e-10)):
            # Accept
            current_max = new_max
            if current_max < best_score:
                best_score = current_max
                best_c = current_c.copy()
        else:
            # Reject: revert
            current_c = old_c
            current_f = D @ current_c
            current_max = old_max

        temperature *= cooling

    # Final accurate evaluation of best candidate
    final_score = evaluate_accurate(best_c)
    if final_score < best_score:
        best_score = final_score

    return [int(x) for x in best_c]

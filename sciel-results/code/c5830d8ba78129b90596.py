import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _legendre_symbol(a, p):
    """Compute the Legendre symbol (a/p)."""
    if a % p == 0:
        return 0
    res = pow(a, (p - 1) // 2, p)
    if res == 1:
        return 1
    elif res == p - 1:
        return -1
    else:
        return 0

def _generate_legendre_sequence(p, N):
    """Generate a sequence of length N based on the Legendre symbol of a prime p."""
    h = np.zeros(N)
    for i in range(N):
        sym = _legendre_symbol(i + 1, p)
        h[i] = 1.0 if sym >= 0 else -1.0
    return h

def _windowed_sinc_sequence(N, window_type='blackman-harris', center_freq=1.0, phase=0.0):
    """
    Generate a signed sequence from the sign of a windowed sinc function.
    The window is designed to suppress sidelobes in the power spectrum,
    which structurally reduces peak autoconvolution sidelobes.
    """
    x = np.arange(N)
    # Windowed sinc: sinc(t - t0) * window(t)
    t = x - N / 2.0
    t0 = center_freq * N / 2.0
    
    # Sinc function
    sinc_val = np.sinc(t / N)  # np.sinc(x) = sin(pi*x)/(pi*x)
    
    # Apply window
    if window_type == 'blackman-harris':
        # 4-term Blackman-Harris
        a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
        n = np.arange(N)
        window = (a0 - a1 * np.cos(2 * np.pi * n / (N - 1)) +
                  a2 * np.cos(4 * np.pi * n / (N - 1)) -
                  a3 * np.cos(6 * np.pi * n / (N - 1)))
    elif window_type == 'kaiser':
        beta = 8.6  # ~31 dB sidelobe attenuation
        n = np.arange(N)
        alpha = (N - 1) / 2.0
        window = np.i0(np.sqrt(1 - ((n - alpha) / alpha) ** 2)) / np.i0(1.0)
    elif window_type == 'hamming':
        n = np.arange(N)
        window = 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    elif window_type == 'hann':
        n = np.arange(N)
        window = 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))
    else:
        window = np.ones(N)
    
    signal = sinc_val * window
    # Add phase shift
    signal = np.roll(signal, int(phase * N))
    
    h = np.sign(signal)
    h[h == 0] = 1.0
    return h

def _local_sign_flip(h, max_passes=5, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves.
    Uses incremental autoconvolution updates (O(N) per flip).
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    size = 2 * n - 1

    ac = np.convolve(h, h, mode='full')
    max_abs_ac = float(np.max(np.abs(ac)))
    s = float(np.sum(h))
    if abs(s) < 1e-6:
        return h, 1e10
    best_val = max_abs_ac / (s * s)

    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            old = h[i]
            delta = -2.0 * old

            s_new = s + delta
            if abs(s_new) < 1e-6:
                continue

            ac_new = ac.copy()
            ac_new[i:i+n] += delta * h
            ac_new[n-1-i:2*n-1-i] += delta * h
            max_abs_new = float(np.max(np.abs(ac_new)))
            val_new = max_abs_new / (s_new * s_new)

            if val_new < best_val:
                h[i] = -old
                ac = ac_new
                s = s_new
                best_val = val_new
                max_abs_ac = max_abs_new
                improved = True
        if not improved:
            break
    return h, best_val

def _spectral_shaping(h, max_iter=50, deadline=None):
    """
    Spectral-domain optimization: adjust signs to minimize peak autoconvolution
    by working in the frequency domain. For each frequency bin, we can think of
    the autoconvolution as |H(f)|^2. We want to minimize the peak of the
    inverse FFT of |H(f)|^2.
    
    This is done by iteratively:
    1. Compute H(f) = FFT(h)
    2. Compute ac = IFFT(|H|^2)
    3. Find the peak in ac
    4. Adjust h to reduce the peak (via gradient-like update in sign space)
    
    Since we're constrained to ±1, we use a soft approach: compute a "pressure"
    field from the autoconvolution peaks and use it to bias sign flips.
    """
    h = h.copy().astype(np.float64)
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    s = float(np.sum(h))
    if abs(s) < 1e-6:
        return h, 1e10
    ac = np.fft.irfft(np.fft.rfft(h, fsize) ** 2, fsize)[:size]
    max_abs_ac = float(np.max(np.abs(ac)))
    best_val = max_abs_ac / (s * s)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Find the location of the maximum absolute autoconvolution
        peak_idx = int(np.argmax(np.abs(ac)))
        peak_val = ac[peak_idx]
        
        # The autoconvolution at lag k is sum_i h[i]*h[i+k] (with appropriate bounds)
        # To reduce the peak at lag k, we want to flip signs where h[i]*h[i+k] has the same sign as peak_val
        # and large magnitude contribution
        
        k = peak_idx
        if k < n:
            # Positive lag: ac[k] = sum_{i=0}^{n-1-k} h[i]*h[i+k]
            # Contribution to ac[k] from pair (i, i+k) is h[i]*h[i+k]
            # Flipping h[i] changes ac[k] by -2*h[i]*h[i+k] for that pair
            # But flipping h[i] also affects other lags
            
            # Simple heuristic: find the index i that, when flipped, most reduces |ac[peak_idx]|
            # while considering the overall max
            best_flip = -1
            best_improvement = 0.0
            
            # Try flipping each h[i] and see effect on peak
            # For efficiency, only try a subset
            candidates = np.random.choice(n, min(n, 100), replace=False)
            for i in candidates:
                old = h[i]
                delta = -2.0 * old
                s_new = s + delta
                if abs(s_new) < 1e-6:
                    continue
                
                ac_new = ac.copy()
                ac_new[i:i+n] += delta * h
                ac_new[n-1-i:2*n-1-i] += delta * h
                max_abs_new = float(np.max(np.abs(ac_new)))
                val_new = max_abs_new / (s_new * s_new)
                
                if val_new < best_val - 1e-9:
                    best_improvement = best_val - val_new
                    best_flip = i
            
            if best_flip >= 0:
                old = h[best_flip]
                delta = -2.0 * old
                s = s + delta
                ac[best_flip:best_flip+n] += delta * h
                ac[n-1-best_flip:2*n-1-best_flip] += delta * h
                h[best_flip] = -old
                max_abs_ac = float(np.max(np.abs(ac)))
                best_val = max_abs_ac / (s * s)
            else:
                break
        else:
            # Negative lag (by symmetry, ac[-k] = ac[k] for real sequences in linear conv)
            # Actually for linear autoconvolution, ac is symmetric: ac[k] = ac[size-1-k]
            # So peak at index > n-1 corresponds to a negative lag
            k_eff = size - 1 - peak_idx
            if k_eff < n:
                # Same logic as positive lag
                best_flip = -1
                candidates = np.random.choice(n, min(n, 100), replace=False)
                for i in candidates:
                    old = h[i]
                    delta = -2.0 * old
                    s_new = s + delta
                    if abs(s_new) < 1e-6:
                        continue
                    
                    ac_new = ac.copy()
                    ac_new[i:i+n] += delta * h
                    ac_new[n-1-i:2*n-1-i] += delta * h
                    max_abs_new = float(np.max(np.abs(ac_new)))
                    val_new = max_abs_new / (s_new * s_new)
                    
                    if val_new < best_val - 1e-9:
                        best_improvement = best_val - val_new
                        best_flip = i
                
                if best_flip >= 0:
                    old = h[best_flip]
                    delta = -2.0 * old
                    s = s + delta
                    ac[best_flip:best_flip+n] += delta * h
                    ac[n-1-best_flip:2*n-1-best_flip] += delta * h
                    h[best_flip] = -old
                    max_abs_ac = float(np.max(np.abs(ac)))
                    best_val = max_abs_ac / (s * s)
                else:
                    break
            else:
                break
    
    return h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    
    best_h = np.ones(N)
    best_val = _score(best_h)
    
    np.random.seed(42)
    
    n_restarts = 300
    
    # Pre-compute Legendre sequences
    primes_to_try = [397, 401, 389, 409, 383, 419, 431, 433, 439, 443, 449, 457, 461, 463, 467]
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break

        if r < len(primes_to_try):
            p = primes_to_try[r]
            h = _generate_legendre_sequence(p, N)
        elif r % 7 == 0:
            # Windowed sinc with Blackman-Harris
            cf = 0.5 + 0.1 * (r // 7) % 10
            ph = (r // 7) % 5 * 0.2
            h = _windowed_sinc_sequence(N, 'blackman-harris', cf, ph)
        elif r % 7 == 1:
            # Windowed sinc with Kaiser
            cf = 0.5 + 0.1 * (r // 7) % 10
            ph = (r // 7) % 5 * 0.2
            h = _windowed_sinc_sequence(N, 'kaiser', cf, ph)
        elif r % 7 == 2:
            # Windowed sinc with Hamming
            cf = 0.5 + 0.1 * (r // 7) % 10
            ph = (r // 7) % 5 * 0.2
            h = _windowed_sinc_sequence(N, 'hamming', cf, ph)
        elif r % 7 == 3:
            # Random +/-1
            h = np.random.choice([-1.0, 1.0], size=N)
        elif r % 7 == 4:
            # Low frequency cosine
            k = (r // 7) % 10 + 1
            x = np.arange(N)
            h = np.sign(np.cos(2 * np.pi * k * x / N))
            h[h == 0] = 1.0
        elif r % 7 == 5:
            # Block structure
            h = np.ones(N)
            pos = 0
            while pos < N:
                block_len = np.random.randint(2, 20)
                sign = np.random.choice([-1.0, 1.0])
                h[pos:pos+block_len] = sign
                pos += block_len
        else:
            # Mostly +1 with sparse -1
            h = np.ones(N)
            n_neg = np.random.randint(0, N//2)
            neg_idx = np.random.choice(N, n_neg, replace=False)
            h[neg_idx] = -1.0
        
        s = np.sum(h)
        if abs(s) < N * 0.1:
            if s < 0:
                h = -h
            if abs(np.sum(h)) < 1e-6:
                h[0] = 1.0
        
        # Local optimization
        h_opt, val_opt = _local_sign_flip(h, max_passes=4, deadline=deadline)

        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt

    # Final polish
    if time.time() < deadline - 5.0:
        best_h, best_val = _local_sign_flip(best_h, max_passes=15, deadline=deadline)

    return best_h

# EVOLVE-BLOCK-END

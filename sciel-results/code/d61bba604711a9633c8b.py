import numpy as np
import time

N = 400


def _autocorr(h):
    # linear one-sided autocorrelation ac[k] = sum_i h[i] h[i+k], k = 0..N-1
    n = 2 * N
    F = np.fft.rfft(h, n=n)
    ac = np.fft.irfft(F * np.conj(F), n=n)[:N]
    return ac


def _hard_score(h):
    # the actual metric: peak autoconvolution concentration (lower is better)
    ac = _autocorr(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    return float(np.max(np.abs(ac)) / dc)


def _smooth_score(h, beta):
    # smooth (log-sum-exp) surrogate of the peak sidelobe energy, guides SA
    ac = _autocorr(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    y = (ac[1:] / dc) ** 2
    m = float(np.max(y))
    z = beta * (y - m)
    return m + float(np.log(np.sum(np.exp(z)))) / beta


def _fBm_signs(rng, H=0.5, n=400, seed=42):
    """Generate a sign sequence from fractional Brownian motion structure.
    
    fBm with Hurst exponent H has self-similar increments. For H=0.5 this is
    standard Brownian motion, but we use the spectral representation to create
    a continuous signal, then discretize to signs. The key idea: fBm has
    power spectrum ~ 1/f^(2H+1), which distributes energy across scales in a
    way that avoids both low-frequency bias (periodic) and high-frequency noise
    (random bits).
    """
    # Generate fBm via spectral method
    # Power spectrum: S(f) ~ 1/f^(2H+1) for f > 0
    freqs = np.fft.rfftfreq(n, d=1.0/n)
    # Avoid DC component
    freqs[0] = 1.0
    # Power spectrum for fBm: ~ f^(-(2H+1))
    power = 1.0 / (freqs ** (2.0 * H + 1.0))
    power[0] = 0.0  # Remove DC
    
    # Random phases
    phases = rng.uniform(0, 2 * np.pi, size=len(freqs))
    # Complex spectrum
    spectrum = np.sqrt(power) * np.exp(1j * phases)
    spectrum[0] = 0.0
    
    # Inverse FFT to get fBm signal
    signal = np.fft.irfft(spectrum, n=n)
    
    # Normalize and convert to signs
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)
    return np.sign(signal)


def _fBm_signs_multiscale(rng, n=400):
    """Generate multiple fBm-based sign sequences with different Hurst exponents
    and scales to provide diverse initializations."""
    results = []
    for H in [0.3, 0.4, 0.5, 0.6, 0.7]:
        # Use different seeds for diversity
        local_rng = np.random.default_rng(rng.integers(0, 2**31))
        signs = _fBm_signs(local_rng, H=H, n=n)
        results.append(signs)
    return results


def construct():
    try:
        t0 = time.time()
        deadline = t0 + 110.0          # safety margin inside the 120s limit
        rng = np.random.default_rng(20240517)

        # ---------- diverse initializations ----------
        idx = np.arange(N)
        inits = [
            np.ones(N),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            np.cos(2 * np.pi * idx ** 2 / (2 * N)),
            np.cos(2 * np.pi * idx ** 2 / (N + 1)),
            np.cos(2 * np.pi * idx ** 2 / (3 * N)),
            rng.choice(np.array([-1.0, 1.0]), size=N),
        ]
        
        # Add fBm-based initializations with multiple Hurst exponents
        fbm_inits = _fBm_signs_multiscale(rng, N)
        inits.extend(fbm_inits)
        
        # Also try fBm with different block structures
        for H in [0.4, 0.6]:
            local_rng = np.random.default_rng(rng.integers(0, 2**31))
            fbm_signal = _fBm_signs(local_rng, H=H, n=N)
            # Apply block averaging to create smoother transitions
            block_size = 8
            n_blocks = N // block_size
            block_means = []
            for b in range(n_blocks):
                block = fbm_signal[b*block_size:(b+1)*block_size]
                block_means.append(np.mean(block))
            block_means = np.array(block_means)
            # Create sign sequence from block means
            block_signs = np.sign(block_means)
            # Expand back to full length
            expanded = np.repeat(block_signs, block_size)
            if len(expanded) < N:
                expanded = np.concatenate([expanded, np.ones(N - len(expanded))])
            inits.append(expanded)

        best_h = None
        best_hard = 1e9
        for h in inits:
            s = _hard_score(h)
            if s < best_hard:
                best_hard = s
                best_h = h.copy()

        h = best_h.copy()
        cur_hard = best_hard
        cur_smooth = _smooth_score(h, 30.0)

        # ---------- simulated annealing with coordinated block sign-flips ----------
        T = 0.02
        T_min = 1e-4
        decay = 0.9995
        beta = 30.0
        sizes = np.array([1, 2, 3, 4, 8, 12, 16, 20, 24, 32])

        while time.time() < deadline - 3.0:
            if rng.random() < 0.75:
                size = int(rng.choice(sizes))
                start = int(rng.integers(0, N - size + 1))
                sl = slice(start, start + size)
                old = h[sl].copy()
                h[sl] *= -1.0
                new_hard = _hard_score(h)
                new_smooth = _smooth_score(h, beta)
                dS = new_smooth - cur_smooth
                if new_hard < cur_hard or (dS <= 0 or rng.random() < np.exp(-dS / T)):
                    cur_hard = new_hard
                    cur_smooth = new_smooth
                    if new_hard < best_hard:
                        best_hard = new_hard
                        best_h = h.copy()
                else:
                    h[sl] = old
            else:
                i = int(rng.integers(0, N))
                old = h[i]
                h[i] *= -1.0
                new_hard = _hard_score(h)
                new_smooth = _smooth_score(h, beta)
                dS = new_smooth - cur_smooth
                if new_hard < cur_hard or (dS <= 0 or rng.random() < np.exp(-dS / T)):
                    cur_hard = new_hard
                    cur_smooth = new_smooth
                    if new_hard < best_hard:
                        best_hard = new_hard
                        best_h = h.copy()
                else:
                    h[i] = old
            T = max(T_min, T * decay)

        # ---------- greedy multi-scale local polish ----------
        h = best_h.copy()
        sizes_polish = [1, 2, 4, 8, 16, 32]
        for _pass in range(40):
            if time.time() > deadline - 1.0:
                break
            any_improve = False
            for size in sizes_polish:
                for start in range(0, N - size + 1):
                    if time.time() > deadline - 1.0:
                        return best_h
                    sl = slice(start, start + size)
                    old = h[sl].copy()
                    h[sl] *= -1.0
                    new_hard = _hard_score(h)
                    if new_hard < best_hard - 1e-12:
                        best_hard = new_hard
                        best_h = h.copy()
                        any_improve = True
                    else:
                        h[sl] = old
            if not any_improve:
                break

        return best_h
    except Exception:
        return np.ones(N)

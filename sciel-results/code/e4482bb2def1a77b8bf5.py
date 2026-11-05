import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    def compute_value(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        c = np.convolve(h, h)
        return abs(2.0 * N * np.max(c) / (s * s))

    def compute_value_from_freq(H):
        """Compute objective from frequency-domain representation H (DFT of h)."""
        h = np.fft.ifft(H).real
        return compute_value(h)

    def freq_grad(H):
        """
        Compute gradient of the objective w.r.t. H (frequency coefficients).
        
        Key insight: c = ifft(|H|^2) (up to normalization).
        The max of c is what we want to minimize.
        
        We work in the frequency domain: h = ifft(H), c = ifft(|H|^2) * N (for proper scaling)
        Actually: ifft(fft(h)*conj(fft(h))) gives circular autocorrelation.
        For full convolution, we need zero-padding.
        
        Let's use: pad h to length 2N-1, compute FFT, then c_full = ifft(|H_padded|^2)
        But this is expensive. Instead, use the relation:
        full_conv(h,h)[k] = sum_i h[i]*h[k-i] for valid i.
        
        Alternative: work directly with the time-domain h but use frequency insights
        for generating good initial candidates and for a different optimization path.
        
        Actually, let me reconsider. The key structural change is:
        1. Use the fact that the objective is scale-invariant, so fix sum(h) = N (i.e., mean = 1)
        2. In frequency domain, minimizing max(c) relates to minimizing the peak of the 
           inverse Fourier transform of |H|^2.
        3. Use a water-filling / spectral flattening approach: make |H|^2 as flat as possible
           to minimize the peak of its inverse transform.
        
        For a flat spectrum |H|^2 = const, c would be a delta function (peak at center).
        But we want to MINIMIZE the max of c, so we want c to be as flat as possible.
        That means |H|^2 should be as "peaked" as possible in frequency... 
        Wait, no. c = ifft(|H|^2). If |H|^2 is flat, c is a spike. If |H|^2 is peaked,
        c is more spread out (flatter). So we want |H|^2 to be concentrated.
        
        Actually, let me think again. We want max_k c[k] to be small.
        c = ifft(|H|^2) (circular, with appropriate padding for full conv).
        The max of c is bounded by the L1 norm of |H|^2 / (2*pi) or similar.
        
        Let me just do a different optimization: use scipy minimize with a 
        frequency-domain parametrization.
        """
        pass

    # ---- New Strategy: Frequency-domain optimization with scipy ----
    from scipy.optimize import minimize
    
    def objective_and_grad(h):
        """Objective and its gradient for time-domain h, but we'll use this in a
        different way: we parametrize h via its DFT magnitudes and phases."""
        s = np.sum(h)
        if abs(s) < 1e-12:
            return 1e10, np.zeros_like(h)
        c = np.convolve(h, h)
        val = abs(2.0 * N * np.max(c) / (s * s))
        # Gradient (same as before but cleaned up)
        idx_max = np.argmax(c)
        k = idx_max
        grad_c = np.zeros(N)
        valid = (k - np.arange(N) >= 0) & (k - np.arange(N) < N)
        grad_c[valid] = h[k - np.arange(N)[valid]] + h[np.arange(N)[valid]]
        
        val_inner = 2.0 * N * c[idx_max] / (s * s)
        sign = 1.0 if val_inner >= 0 else -1.0
        grad = 2.0 * N * (grad_c * s - 2.0 * c[idx_max]) / (s ** 3) * sign
        return val, grad

    # Phase 1: Generate diverse spectral initializations
    rng = np.random.default_rng(123)
    candidates = []
    
    # Baseline
    h0 = np.ones(N)
    candidates.append(compute_value(h0))
    
    # Spectral shaping: create h with controlled frequency content
    # Strategy: h = 1 + alpha * signal, where signal has specific spectral shape
    # We want the autoconvolution to be as flat as possible.
    
    # Type A: Low-frequency dominant (smooth signals)
    for alpha in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        for freq_scale in [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
            if time.time() > deadline - 30:
                break
            n = np.arange(N)
            # Smooth bump: h = 1 + alpha * exp(-(n/N - 0.5)^2 / (2*freq_scale^2))
            h = 1.0 + alpha * np.exp(-((n / N - 0.5) ** 2) / (2 * freq_scale ** 2))
            # Center it to keep mean ~1
            h = h - np.mean(h) + 1.0
            val = compute_value(h)
            candidates.append(val)
    
    # Type B: Cosine windows of various widths
    for width in [0.2, 0.4, 0.6, 0.8, 1.0]:
        for amp in [0.5, 1.0, 2.0, 3.0, 5.0]:
            if time.time() > deadline - 30:
                break
            n = np.arange(N)
            x = n / N
            # Cosine window centered at 0.5
            w = np.cos(np.pi * (x - 0.5) / width) ** 2
            w[x > 0.5 + width] = 0
            w[x < 0.5 - width] = 0
            h = 1.0 + amp * (w - np.mean(w))
            val = compute_value(h)
            candidates.append(val)
    
    # Type C: Random spectral shapes (random Fourier coefficients)
    for _ in range(80):
        if time.time() > deadline - 30:
            break
        # Random frequency content
        H_mag = np.abs(rng.normal(0, 1, N)) + 0.5
        H_phase = rng.uniform(-np.pi, np.pi, N)
        H = H_mag * np.exp(1j * H_phase)
        h = np.fft.ifft(H).real
        h = h - np.mean(h) + 1.0  # Keep mean at 1
        if np.all(np.isfinite(h)):
            val = compute_value(h)
            candidates.append(val)
    
    # Type D: Structured sign patterns
    for period in [2, 3, 4, 5, 7, 10, 13, 17, 20, 25, 33, 50]:
        for amp in [0.5, 1.0, 2.0, 3.0]:
            if time.time() > deadline - 30:
                break
            n = np.arange(N)
            # Square wave
            sw = np.sign(np.sin(2 * np.pi * n / period))
            h = 1.0 + amp * sw
            val = compute_value(h)
            candidates.append(val)
    
    # Type E: Piecewise constant blocks
    for nblocks in [2, 3, 4, 5, 8, 10, 16]:
        for amp in [1.0, 2.0, 3.0, 5.0]:
            if time.time() > deadline - 30:
                break
            block_size = N // nblocks
            h = np.zeros(N)
            for i in range(nblocks):
                sign = 1.0 if i % 2 == 0 else -1.0
                start = i * block_size
                end = min(start + block_size, N)
                h[start:end] = sign
            h = 1.0 + amp * h
            val = compute_value(h)
            candidates.append(val)
    
    # Type F: Gaussian mixtures in time domain
    for _ in range(40):
        if time.time() > deadline - 30:
            break
        h = np.ones(N)
        n_gauss = rng.integers(2, 8)
        for _ in range(n_gauss):
            center = rng.uniform(0, N)
            width = rng.uniform(2, 20)
            amp = rng.uniform(-3, 3)
            n = np.arange(N)
            h += amp * np.exp(-((n - center) ** 2) / (2 * width ** 2))
        val = compute_value(h)
        candidates.append(val)

    # Find the best candidate value and reconstruct h
    # We need to actually store h, not just values. Let me redo this properly.
    
    # Actually, let me restructure: store (val, h) pairs
    candidates_h = []
    
    # Re-evaluate: I'll just pick the best from a fresh set
    # Let me be more efficient and just generate and track
    
    h_best = np.ones(N)
    val_best = compute_value(h_best)
    
    def try_h(h):
        nonlocal h_best, val_best
        if not np.all(np.isfinite(h)):
            return
        if abs(np.sum(h)) < 1e-12:
            return
        val = compute_value(h)
        if val < val_best:
            val_best = val
            h_best = h.copy()
    
    # Generate and evaluate candidates
    for alpha in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]:
        for freq_scale in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]:
            if time.time() > deadline - 40:
                break
            n = np.arange(N)
            h = 1.0 + alpha * np.exp(-((n / N - 0.5) ** 2) / (2 * freq_scale ** 2))
            h = h - np.mean(h) + 1.0
            try_h(h)
    
    for width in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        for amp in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
            if time.time() > deadline - 40:
                break
            n = np.arange(N)
            x = n / N
            w = np.cos(np.pi * (x - 0.5) / width) ** 2
            w[x > 0.5 + width] = 0
            w[x < 0.5 - width] = 0
            h = 1.0 + amp * (w - np.mean(w))
            try_h(h)
    
    for _ in range(100):
        if time.time() > deadline - 40:
            break
        H_mag = np.abs(rng.normal(0, 1, N)) + 0.1
        H_phase = rng.uniform(-np.pi, np.pi, N)
        H = H_mag * np.exp(1j * H_phase)
        h = np.fft.ifft(H).real
        h = h - np.mean(h) + 1.0
        try_h(h)
    
    for period in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 33, 40, 50, 67, 80, 100]:
        for amp in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
            if time.time() > deadline - 40:
                break
            n = np.arange(N)
            sw = np.sign(np.sin(2 * np.pi * n / period))
            h = 1.0 + amp * sw
            try_h(h)
    
    for nblocks in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32]:
        for amp in [1.0, 2.0, 3.0, 5.0, 10.0]:
            if time.time() > deadline - 40:
                break
            block_size = N // nblocks
            h = np.zeros(N)
            for i in range(nblocks):
                sign = 1.0 if i % 2 == 0 else -1.0
                start = i * block_size
                end = min(start + block_size, N)
                h[start:end] = sign
            h = 1.0 + amp * h
            try_h(h)
    
    for _ in range(50):
        if time.time() > deadline - 40:
            break
        h = np.ones(N)
        n_gauss = rng.integers(2, 10)
        for _ in range(n_gauss):
            center = rng.uniform(0, N)
            width = rng.uniform(1, 30)
            amp = rng.uniform(-5, 5)
            n = np.arange(N)
            h += amp * np.exp(-((n - center) ** 2) / (2 * width ** 2))
        try_h(h)
    
    # Phase 2: Spectral coordinate descent
    # Work in the frequency domain: H = fft(h)
    # Objective: minimize max(c) where c = ifft(|H|^2) (circular, padded)
    # For full convolution, pad to 2N-1
    # Coordinate descent: adjust one frequency bin at a time
    
    h_cur = h_best.copy()
    val_cur = val_best
    
    # Pad for full convolution
    Np = 2 * N - 1
    
    def compute_c_padded(h):
        """Full autoconvolution via FFT with padding."""
        H = np.fft.rfft(h, Np)
        c = np.fft.irfft(H * np.conj(H), Np)
        return c
    
    def freq_cd_step(h, max_iters=200):
        """Coordinate descent in frequency domain."""
        h = h.copy()
        Np = 2 * N - 1
        s = np.sum(h)
        
        for it in range(max_iters):
            if time.time() > deadline - 20:
                break
            
            c = compute_c_padded(h)
            idx_max = np.argmax(c)
            c_max = c[idx_max]
            val = abs(2.0 * N * c_max / (s * s))
            
            if val < val_cur:
                val_cur = val
                h_best = h.copy()
            
            # Get current spectrum
            H = np.fft.rfft(h, Np)
            
            # Try adjusting each frequency bin's magnitude (keeping phase)
            improved = False
            for k in range(len(H)):
                if time.time() > deadline - 20:
                    break
                
                H_orig = H[k].copy()
                mag = np.abs(H_orig)
                phase = np.angle(H_orig)
                
                # Try a few magnitude adjustments
                for scale in [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]:
                    H_new = H.copy()
                    H_new[k] = mag * scale * np.exp(1j * phase)
                    
                    h_new = np.fft.irfft(H_new, Np)[:N]
                    s_new = np.sum(h_new)
                    if abs(s_new) < 1e-12:
                        continue
                    
                    c_new = compute_c_padded(h_new)
                    val_new = abs(2.0 * N * np.max(c_new) / (s_new * s_new))
                    
                    if val_new < val:
                        H = H_new
                        h = h_new
                        improved = True
                        break
            
            # Also try phase adjustments for a few random bins
            for _ in range(5):
                if time.time() > deadline - 20:
                    break
                k = rng.integers(0, len(H))
                H_orig = H[k].copy()
                mag = np.abs(H_orig)
                phase = np.angle(H_orig)
                
                for dphase in [0.1, -0.1, 0.3, -0.3, 0.5, -0.5]:
                    H_new = H.copy()
                    H_new[k] = mag * np.exp(1j * (phase + dphase))
                    
                    h_new = np.fft.irfft(H_new, Np)[:N]
                    s_new = np.sum(h_new)
                    if abs(s_new) < 1e-12:
                        continue
                    
                    c_new = compute_c_padded(h_new)
                    val_new = abs(2.0 * N * np.max(c_new) / (s_new * s_new))
                    
                    if val_new < val:
                        H = H_new
                        h = h_new
                        improved = True
                        break
            
            if not improved:
                break
        
        return h, val_cur
    
    h_cur, val_cur = freq_cd_step(h_cur)
    
    # Phase 3: Multi-start spectral optimization
    # Generate several diverse spectral initializations and run CD on each
    for restart in range(15):
        if time.time() > deadline - 25:
            break
        
        # Create diverse initial spectrum
        H_mag = np.abs(rng.normal(0, 1, N // 2 + 1)) + 0.1
        H_phase = rng.uniform(-np.pi, np.pi, N // 2 + 1)
        H_init = H_mag * np.exp(1j * H_phase)
        h_init = np.fft.irfft(H_init, Np)[:N]
        h_init = h_init - np.mean(h_init) + 1.0
        
        if not np.all(np.isfinite(h_init)):
            continue
        
        h_cd, val_cd = freq_cd_step(h_init, max_iters=100)
        if val_cd < val_best:
            val_best = val_cd
            h_best = h_cd.copy()
    
    # Phase 4: Fine-tuning in time domain with small perturbations
    if time.time() < deadline - 15:
        h_cur = h_best.copy()
        val_cur = val_best
        
        for step in range(300):
            if time.time() > deadline - 10:
                break
            
            improved = False
            # Try single-element perturbations
            for _ in range(30):
                idx = rng.integers(0, N)
                for delta in [0.01, -0.01, 0.05, -0.05, 0.1, -0.1]:
                    h_new = h_cur.copy()
                    h_new[idx] += delta
                    s_new = np.sum(h_new)
                    if abs(s_new) < 1e-6:
                        continue
                    val_new = compute_value(h_new)
                    if val_new < val_cur:
                        h_cur = h_new
                        val_cur = val_new
                        improved = True
                        break
            
            # Try small block perturbations
            for _ in range(10):
                start = rng.integers(0, max(1, N - 10))
                length = rng.integers(2, 10)
                end = min(start + length, N)
                delta = rng.normal(0, 0.05)
                h_new = h_cur.copy()
                h_new[start:end] += delta
                s_new = np.sum(h_new)
                if abs(s_new) < 1e-6:
                    continue
                val_new = compute_value(h_new)
                if val_new < val_cur:
                    h_cur = h_new
                    val_cur = val_new
                    improved = True
            
            if not improved:
                # Try larger perturbations
                for _ in range(10):
                    idx = rng.integers(0, N)
                    delta = rng.normal(0, 0.5)
                    h_new = h_cur.copy()
                    h_new[idx] += delta
                    s_new = np.sum(h_new)
                    if abs(s_new) < 1e-6:
                        continue
                    val_new = compute_value(h_new)
                    if val_new < val_cur:
                        h_cur = h_new
                        val_cur = val_new
                        improved = True
                        break
            
            if not improved:
                break
        
        if val_cur < val_best:
            val_best = val_cur
            h_best = h_cur.copy()
    
    # Final safety check
    if not np.all(np.isfinite(h_best)) or abs(np.sum(h_best)) < 1e-12:
        h_best = np.ones(N)
    
    return h_best


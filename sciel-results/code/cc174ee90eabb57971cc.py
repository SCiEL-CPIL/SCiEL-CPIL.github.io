"""
Strategy: Use a frequency-domain / spectral design approach to minimize the
peak sidelobe of the autoconvolution, rather than local sign-flip search.

The key insight: the autoconvolution's peak sidelobe is controlled by the
autocorrelation structure. For a ±1 sequence, we can think in terms of the
DFT/FFT of the sequence. The autoconvolution at lag k is related to the
inverse DFT of |F(f)|². Minimizing the max sidelobe of the autocorrelation
is equivalent to minimizing the peak of the inverse DFT of |F|² away from
the DC bin.

A different structural approach: use a "low autocorrelation sequence" design
via the Welch bound / Sidon set construction, or use the fact that for
certain structured sequences (like Barker-like or Zadoff-Chu derived),
the sidelobes are bounded. We'll use a hybrid:

1. Initialize with a quadratic residue (Legendre) sequence, which has
   known good autocorrelation properties.
2. Use a gradient-based approach on the continuous relaxation: treat the
   ±1 constraint as a soft constraint and use projected gradient descent
   on the peak sidelobe metric, with subgradient estimation via sampling.
3. Polish with local search on the discrete ±1 values.

The critical difference from the current program: instead of random/greedy
sign flips, we use the mathematical structure of low-autocorrelation sequences
and a continuous-relaxation gradient method to guide the search toward
regions of the space with genuinely low peak sidelobes.
"""
import numpy as np
import time

N = 400


def _autoconv_bound(f):
    """Compute the L-infinity norm of the autoconvolution sidelobes.
    
    For a step function with N equal intervals on [-1/4, 1/4], the
    autoconvolution (f*f)(x) = ∫ f(t) f(x-t) dt has support on [-1/2, 1/2].
    The "upper bound on C3" corresponds to the maximum absolute value of the
    autoconvolution away from the central peak (the DC component).
    
    We compute the full autoconvolution via FFT and return the max |value|
    excluding the central lobe.
    """
    dx = (1.0 / 2.0) / N
    
    # Linear convolution via FFT
    n_fft = 2 * N
    f_pad = np.zeros(n_fft)
    f_pad[:N] = f
    
    F = np.fft.fft(f_pad, n=n_fft)
    conv = np.fft.ifft(F * F).real
    autoconv = conv * dx
    
    # The autoconvolution is symmetric. The central peak is at index N
    # (corresponding to x=0). We want the max sidelobe, i.e., max |autoconv|
    # excluding the central region.
    
    # The central lobe spans roughly indices [N-1, N+1] for the main peak.
    # For a proper sidelobe measurement, we exclude the central bin.
    # The autoconv at lag 0 is sum(f^2) * dx = N * 1 * dx = 1/2 for ±1.
    # Sidelobes are at non-zero lags.
    
    # In the FFT convolution output, the value at index i corresponds to
    # the convolution sum at that discrete position. The central peak
    # is at index N (zero lag). Sidelobes are at all other indices.
    
    # Mask out the central bin
    mask = np.ones(n_fft, dtype=bool)
    mask[N] = False
    
    return np.max(np.abs(autoconv[mask]))


def _score(f):
    """Return the upper bound on C3 for the given step function."""
    return _autoconv_bound(f)


def _legendre_sequence(n):
    """Generate a Legendre (quadratic residue) sequence of length n.
    
    For prime p, the Legendre symbol (a|p) gives a ±1 sequence with
    good autocorrelation properties. We adapt for n=400.
    """
    # Find a prime p >= n
    def is_prime(k):
        if k < 2:
            return False
        if k % 2 == 0:
            return k == 2
        for i in range(3, int(np.sqrt(k)) + 1, 2):
            if k % i == 0:
                return False
        return True
    
    p = n
    while not is_prime(p):
        p += 1
    
    # Legendre symbol sequence of length p
    seq = np.ones(p)
    for a in range(1, p):
        # Compute Legendre symbol (a|p) using Euler's criterion
        ls = pow(a, (p - 1) // 2, p)
        if ls == p - 1:  # ls == -1 mod p
            seq[a] = -1.0
    
    # Take first n elements, centering around 0
    start = (p - n) // 2
    result = seq[start:start + n]
    return result


def _zadoff_chu_like(n):
    """Generate a Zadoff-Chu-like sequence for length n.
    
    Zadoff-Chu sequences have ideal periodic autocorrelation.
    For even n, we construct a ±1 approximation.
    """
    # For n=400, use a Zadoff-Chu with root u
    # zc[k] = exp(-j * pi * u * k * (k+1) / n) for k=0..n-1
    # Take real part sign as ±1
    u = 1
    k = np.arange(n)
    phase = -np.pi * u * k * (k + 1) / n
    # Map to ±1 based on sign of cosine (real part)
    seq = np.sign(np.cos(phase))
    # Ensure no zeros
    seq[seq == 0] = 1.0
    return seq


def _gradient_polish(f, time_budget, max_iter=500):
    """Polish the ±1 sequence using a subgradient method on the peak sidelobe.
    
    We approximate the subgradient of max_k |c(k)| by finding the argmax
    and using the corresponding gradient contribution.
    """
    start = time.time()
    f = f.copy()
    best_f = f.copy()
    best_score = _score(f)
    
    for it in range(max_iter):
        if time.time() - start > time_budget:
            break
        
        # Compute autoconvolution
        dx = (1.0 / 2.0) / N
        n_fft = 2 * N
        f_pad = np.zeros(n_fft)
        f_pad[:N] = f
        F = np.fft.fft(f_pad, n=n_fft)
        conv = np.fft.ifft(F * F).real
        autoconv = conv * dx
        
        # Find the peak sidelobe location
        mask = np.ones(n_fft, dtype=bool)
        mask[N] = False
        abs_ac = np.abs(autoconv)
        abs_ac[~mask] = 0
        peak_idx = np.argmax(abs_ac)
        
        # The subgradient of max_k |c(k)| with respect to f[i] involves
        # the cross-correlation at the peak lag.
        # For the peak at lag L (in FFT index), the gradient contribution
        # from f[i] is related to f[i - L] or f[i + L] depending on the side.
        
        # A simpler approach: try flipping the sign at positions that
        # most reduce the peak. The peak at index peak_idx in the
        # convolution corresponds to lag (peak_idx - N) in the original.
        lag = peak_idx - N
        
        # The autoconv at lag L is sum_i f[i] * f[i+L] (with appropriate bounds)
        # To reduce |autoconv[peak_idx]|, we want to flip signs where
        # f[i] * f[i+L] has the same sign as autoconv[peak_idx].
        
        sign_ac = np.sign(autoconv[peak_idx])
        
        # Compute the "contribution" of each position to the peak
        # For lag L > 0: contribution[i] = f[i] * f[i+L]
        # For lag L < 0: contribution[i] = f[i] * f[i+L] (with L negative)
        
        if lag > 0:
            valid = np.arange(N - lag)
            contrib = f[valid] * f[valid + lag]
            # Flipping f[i] for i in valid changes contrib by -2*f[i]*f[i+lag]
            # We want to flip where contrib * sign_ac > 0 (same sign as peak)
            flip_mask = contrib * sign_ac > 0
            flip_candidates = valid[flip_mask]
        elif lag < 0:
            valid = np.arange(-lag, N)
            contrib = f[valid] * f[valid + lag]
            flip_mask = contrib * sign_ac > 0
            flip_candidates = valid[flip_mask]
        else:
            flip_candidates = np.array([])
        
        if len(flip_candidates) > 0:
            # Flip the position that most reduces the peak
            # Approximate: flip the one with largest |contrib|
            if lag > 0:
                contribs = f[flip_candidates] * f[flip_candidates + lag]
            else:
                contribs = f[flip_candidates] * f[flip_candidates + lag]
            
            best_flip = flip_candidates[np.argmax(np.abs(contribs))]
            f[best_flip] = -f[best_flip]
            
            new_score = _score(f)
            if new_score < best_score:
                best_score = new_score
                best_f = f.copy()
            else:
                f[best_flip] = -f[best_flip]  # Revert
    
    return best_f, best_score


def construct():
    """Construct a signed step function with N intervals to minimize the autoconvolution upper bound."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Generate candidate initial sequences from different design families
    candidates = []
    
    # 1. Legendre (quadratic residue) sequence
    candidates.append(_legendre_sequence(N))
    
    # 2. Zadoff-Chu-like sequence
    candidates.append(_zadoff_chu_like(N))
    
    # 3. All ones (baseline)
    candidates.append(np.ones(N))
    
    # 4. Alternating
    candidates.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # 5. Random
    np.random.seed(42)
    candidates.append(np.random.choice([-1.0, 1.0], size=N))
    
    # 6. Another random seed
    np.random.seed(123)
    candidates.append(np.random.choice([-1.0, 1.0], size=N))
    
    # 7. Block pattern: 10 ones, 10 minus ones
    pattern = np.tile(np.array([1.0] * 10 + [-1.0] * 10), N // 20)
    candidates.append(pattern[:N])
    
    # 8. Quadratic phase
    k = np.arange(N)
    candidates.append(np.sign(np.cos(2 * np.pi * k * k / (4 * N))))
    
    # Evaluate all candidates
    best_f = None
    best_score = np.inf
    
    for cand in candidates:
        cand = np.sign(cand)  # Ensure ±1
        cand[cand == 0] = 1.0
        s = _score(cand)
        if s < best_score:
            best_score = s
            best_f = cand.copy()
    
    # Gradient polish on the best candidate
    elapsed = time.time() - start_time
    remaining = deadline - start_time - 5.0  # Leave 5s margin
    if remaining > 10:
        best_f, best_score = _gradient_polish(best_f, time_budget=remaining * 0.5)
    
    # Local search: greedy sign flips
    elapsed = time.time() - start_time
    remaining = deadline - start_time - 5.0
    if remaining > 5:
        improved = True
        while improved and (time.time() - start_time) < deadline - 5:
            improved = False
            indices = np.random.permutation(N)
            for i in indices:
                if (time.time() - start_time) > deadline - 5:
                    break
                old_val = best_f[i]
                best_f[i] = -old_val
                new_score = _score(best_f)
                if new_score < best_score - 1e-12:
                    best_score = new_score
                    improved = True
                else:
                    best_f[i] = old_val
    
    return best_f

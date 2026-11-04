"""Rewritten program: phase-optimized signed step functions for C3-prime."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def _autoconvolve(a):
    """Compute full autoconvolution of a (length N) -> length 2N-1."""
    fa = np.fft.rfft(a, n=2*N)
    ac = np.fft.irfft(fa * np.conj(fa), n=2*N)
    return ac[:2*N-1]


def _score(a):
    """Score = max|AC| / (sum(a))^2. Return inf if sum is 0."""
    s = a.sum()
    if abs(s) < 1e-9:
        return float('inf')
    ac = _autoconvolve(a)
    return np.max(np.abs(ac)) / (s * s)


def _soft_max(x, beta):
    """Smooth max via log-sum-exp (for |x|)."""
    m = np.max(np.abs(x))
    if m < 1e-300:
        return 0.0
    ex = np.exp(beta * (np.abs(x) - m))
    return m + np.log(np.sum(ex)) / beta


def _phase_gradient_step(phi, a, beta, lr):
    """
    Gradient step on phase to minimize soft-max autoconvolution.
    phi: complex phases (unit magnitude), a: real sequence.
    We optimize the real sequence a directly but with spectral flatness penalty.
    """
    # Current autoconvolution
    fa = np.fft.rfft(a, n=2*N)
    ac = np.fft.irfft(fa * np.conj(fa), n=2*N)[:2*N-1]
    
    # Gradient of soft-max(|ac|) w.r.t. ac
    m = np.max(np.abs(ac))
    ex = np.exp(beta * (np.abs(ac) - m))
    w = ex / np.sum(ex)
    # d(soft_max)/d(ac_j) = w_j * sign(ac_j)
    grad_ac = w * np.sign(ac)
    
    # Gradient w.r.t. a: ac = a * a (autoconv), dac/da = 2 * a convolved with delta-like
    # Actually ac[k] = sum_i a[i] a[i+k], so dac[k]/da[j] = a[j-k] + a[j+k] (with bounds)
    # This is the convolution of a with a reflected... let's use the FFT approach
    # ac = ifft(|fft(a)|^2), so d(ac)/da involves the "phase" of the spectrum
    # Simpler: use the fact that ac = a ⊛ a (autoconv), gradient = 2 * (a convolved with a reversed appropriately)
    # For full autoconvolution, gradient of sum_k g[k]*ac[k] w.r.t. a[j]:
    # = 2 * sum_k g[k] * (a[j-k] + a[j+k] with bounds)
    # This is 2 * (g * a)_j where * is cross-correlation-like
    # Let's compute it via FFT for efficiency
    # ac[k] = sum_i a[i] a[i+k]
    # d/d a[j] sum_k g[k] ac[k] = sum_k g[k] (a[j-k] + a[j+k]) for valid indices
    # = (g ⊛ a)_j + (g ⊛ a_rev)_j ... let me just do it directly with the structure
    # Actually: sum_k g[k] a[j-k] = (g conv a)[j] and sum_k g[k] a[j+k] = (g conv a_flipped)[j]
    # where a_flipped[i] = a[N-1-i]
    # Let me use a cleaner formulation. For zero-padded length 2N:
    g_pad = np.zeros(2*N)
    g_pad[:len(grad_ac)] = grad_ac
    a_pad = np.zeros(2*N)
    a_pad[:N] = a
    # ac[k] for k in 0..2N-2: sum_{i=0}^{N-1-k} a[i] a[i+k]
    # d ac[k] / d a[j] = 1 if j in [0, N-1-k] and (j+k in range) ... 
    # = 1 if 0<=j<N and 0<=j+k<2N-1 and j+k<N and j>=0 and j+k>=0
    # = 1 if 0<=j<N and 0<=j+k<N (since k>=0, j+k>=0 automatic if j>=0)
    # So d ac[k]/d a[j] = 1 if 0<=j<N and 0<=j+k<N
    # Sum over k: sum_k g[k] * [0<=j+k<N] = sum_k g[k] * [k < N-j] = sum_{k=0}^{min(len(g), N-j-1)} g[k]
    # This is a prefix sum of g!
    # Similarly the "other" term: when we differentiate a[i+k] w.r.t. a[j], i+k=j, so i=j-k
    # d ac[k]/d a[j] from the second factor = 1 if 0<=j-k<N and 0<=k, i.e., j-k>=0 and j-k<N
    # = 1 if k<=j and j-k<N, i.e., k in [0, j] and k > j-N
    # Sum over k: sum_{k=max(0,j-N+1)}^{min(j, len(g)-1)} g[k]
    
    # Let's compute prefix sums of grad_ac
    g = grad_ac  # length 2N-1
    prefix = np.cumsum(g)
    # d/d a[j] sum_k g[k] ac[k] = sum_{k=0}^{min(2N-2, N-1-j)} g[k] + sum_{k=max(0, j-N+1)}^{min(j, 2N-2)} g[k]
    # First term: k from 0 to N-1-j (if N-1-j >= 0)
    # Second term: k from max(0, j-N+1) to min(j, 2N-2)
    
    grad_a = np.zeros(N)
    for j in range(N):
        # First sum: k = 0 to N-1-j
        k1_max = N - 1 - j
        if k1_max >= 0 and k1_max < len(g):
            s1 = prefix[min(k1_max, len(g)-1)]
        else:
            s1 = 0.0
        # Second sum: k = max(0, j-N+1) to min(j, 2N-2)
        k2_min = max(0, j - N + 1)
        k2_max = min(j, 2*N - 2)
        if k2_min <= k2_max and k2_max < len(g):
            if k2_min == 0:
                s2 = prefix[k2_max]
            else:
                s2 = prefix[k2_max] - prefix[k2_min - 1]
        else:
            s2 = 0.0
        grad_a[j] = s1 + s2
    
    # Spectral flatness penalty: encourage |FFT(a)| to be constant
    fa = np.fft.rfft(a, n=2*N)
    mag = np.abs(fa)
    target_mag = np.mean(mag)
    # Gradient of sum (|fa| - target)^2 w.r.t. a
    # d |fa| / d a: fa = F a, d|fa|/da = F^H (fa/|fa|) where defined
    phase = fa / np.where(mag > 1e-10, mag, 1.0)
    grad_flat = np.fft.irfft(np.conj(phase) * (mag - target_mag), n=2*N)[:N]
    
    # Combine gradients
    total_grad = grad_a + 0.1 * grad_flat
    
    # Update a
    a_new = a - lr * total_grad
    return a_new, grad_a, grad_flat


def construct():
    deadline = time.time() + 130  # 130s safety margin before 150s kill
    
    # Start with a good initial guess: use a known low-autocorrelation pattern
    # Try a few different initializations and pick the best
    rng = np.random.default_rng(42)
    
    best_a = np.ones(N)
    best_score = _score(best_a)
    
    # Try several initializations
    initializations = []
    
    # 1. All ones (seed)
    initializations.append(np.ones(N))
    
    # 2. Random ±1
    initializations.append(rng.choice([-1, 1], size=N))
    
    # 3. Alternating
    initializations.append(np.ones(N) * (1 if N % 2 == 1 else 1))
    alt = np.ones(N)
    alt[1::2] = -1
    initializations.append(alt)
    
    # 4. Block-based: long blocks of +1 then -1
    for block_len in [10, 20, 50, 100]:
        b = np.ones(N)
        for i in range(0, N, block_len):
            b[i:i+block_len] = 1 if (i // block_len) % 2 == 0 else -1
        initializations.append(b)
    
    # Evaluate all initializations
    for a0 in initializations:
        s = _score(a0)
        if s < best_score:
            best_score = s
            best_a = a0.copy()
    
    # Phase/continuous optimization to find a better starting point
    # Use the best initialization and run gradient descent on continuous relaxation
    a = best_a.astype(float)
    beta = 50.0  # sharpness of soft-max
    lr = 0.01
    lr_decay = 0.999
    
    # Run gradient steps with deadline
    n_steps = 0
    while time.time() < deadline - 30:  # leave 30s for projection and polish
        a_new, _, _ = _phase_gradient_step(a, a, beta, lr)
        # Project to bounded range
        a = np.clip(a_new, -2.0, 2.0)
        lr *= lr_decay
        n_steps += 1
        if n_steps % 100 == 0:
            # Check if we're still improving
            pass
    
    # Project to ±1
    a_binary = np.sign(a)
    a_binary[a_binary == 0] = 1
    s_bin = _score(a_binary)
    
    if s_bin < best_score:
        best_score = s_bin
        best_a = a_binary.copy()
    
    # Local search / polish on the binary sequence
    a = best_a.copy()
    current_score = best_score
    
    # Single-flip local search
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        # Try flipping each position
        for j in range(0, N, 1):
            old = a[j]
            a[j] = -old
            s = _score(a)
            if s < current_score - 1e-12:
                current_score = s
                improved = True
            else:
                a[j] = old
    
    # Multi-flip local search: try flipping pairs
    if time.time() < deadline - 5:
        # Random pair flips
        for _ in range(1000):
            if time.time() > deadline - 5:
                break
            i, j = rng.integers(0, N, size=2)
            if i == j:
                continue
            a[i] *= -1
            a[j] *= -1
            s = _score(a)
            if s < current_score - 1e-12:
                current_score = s
            else:
                a[i] *= -1
                a[j] *= -1
    
    if current_score < best_score:
        best_score = current_score
        best_a = a.copy()
    
    return best_a
# EVOLVE-BLOCK-END

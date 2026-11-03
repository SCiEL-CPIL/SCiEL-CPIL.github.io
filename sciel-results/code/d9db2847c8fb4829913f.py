"""Spectral-flatness gradient descent + local search for signed autocorrelation minimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve

def _compute_score(x):
    """Compute the C3' metric: max |autoconvolution| / (sum(x))^2"""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = fftconvolve(x, x, mode='full')
    max_abs = np.max(np.abs(c))
    return max_abs / (s * s)

def _compute_conv(x):
    """Compute full autoconvolution using FFT."""
    return fftconvolve(x, x, mode='full')

def _spectral_flatness_gradient(x):
    """
    Compute gradient of spectral flatness loss.
    We want to minimize the peak of |autoconvolution|.
    
    Key insight: The autoconvolution's DFT is |X(f)|^2 where X is the DFT of x.
    By Parseval, sum_k c[k]^2 = (1/M) * sum_f |X(f)|^4 where M is DFT length.
    
    To minimize max|c|, we want to spread the autoconvolution energy evenly.
    This is related to making |X(f)|^2 as flat as possible.
    
    We use a continuous relaxation: let x be real, optimize the phase of X.
    The gradient descent on the phase drives |X(f)| toward uniform.
    
    For a ±1 sequence, we do:
    1. Compute X = FFT(x)
    2. Compute target: |X_target| = sqrt(mean(|X|^2)) (flat magnitude)
    3. Update: X_new = X * (|X_target| / |X|) approximately
    4. x_new = IFFT(X_new), then round to ±1
    
    But we can do gradient descent directly on the continuous relaxation.
    """
    M = 2 * N  # DFT length for full autoconvolution
    X = np.fft.fft(x, n=M)
    mag = np.abs(X)
    phase = np.angle(X)
    
    # Target: flat magnitude
    mean_mag2 = np.mean(mag ** 2)
    target_mag = np.sqrt(mean_mag2)
    
    # Gradient of loss = sum((|X| - target_mag)^2) w.r.t. phase
    # d/dphase |X|^2 = 2*|X|*d|X|/dphase ... this gets complicated.
    # Simpler: use the "spectral shaping" approach.
    # Update X to have target magnitude, keep phase.
    X_new = target_mag * np.exp(1j * phase)
    x_new = np.fft.ifft(X_new).real
    
    return x_new

def _spectral_descent(x, deadline, n_steps=2000, lr=0.5):
    """
    Gradient descent on continuous relaxation to flatten spectrum.
    Then round to ±1 and do local search.
    """
    M = 2 * N
    x_cont = x.astype(float).copy()
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Compute FFT
        X = np.fft.fft(x_cont, n=M)
        mag = np.abs(X)
        phase = np.angle(X)
        
        # Target magnitude: flat
        mean_mag2 = np.mean(mag ** 2)
        target_mag = np.sqrt(mean_mag2)
        
        # Interpolate between current and target
        # This is like a "spectral equalizer" step
        X_new = (1 - lr) * X + lr * target_mag * np.exp(1j * phase)
        
        x_cont = np.fft.ifft(X_new).real
        
        # Project: push toward ±1 by applying tanh-like scaling
        # Keep magnitude but push signs
        # Actually, let's just track the best ±1 rounding
        x_rounded = np.sign(x_cont)
        x_rounded[x_rounded == 0] = 1
    
    # Return the best rounded version
    return x_rounded

def _local_search_fast(x, deadline, max_rounds=200):
    """
    Fast local search using vectorized flip evaluation.
    Key optimization: precompute the "flip profile" for each position.
    """
    best_seq = x.copy()
    best_score = _compute_score(best_seq)
    current = x.copy()
    current_score = best_score
    c = _compute_conv(current)
    s = np.sum(current)
    s2 = s * s
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        best_improvement = 0
        best_flip_idx = -1
        best_new_c = None
        
        # For each position, compute the delta in max|c| if we flip it
        # delta_c[k] = -2*x[i]*(x[i+k] + x[i-k]) for k != 0
        
        # Vectorized: for each i, the delta profile is -2*x[i] * profile_i
        # where profile_i[k] = x[i+k] + x[i-k]
        
        # We can compute all profiles at once using convolution tricks,
        # but for N=400, let's do it per-position but efficiently.
        
        for i in range(N):
            if time.time() > deadline:
                break
            
            v = current[i]
            if v == 0:
                continue
            
            # Compute delta profile
            # For lag k (1 to N-1):
            #   delta[k] = -2*v*(x[i+k] + x[i-k])
            
            new_c = c.copy()
            
            # Efficient computation of the delta
            # x[i+k] for k=1..N-1-i: x[i+1], x[i+2], ..., x[N-1]
            # x[i-k] for k=1..i: x[i-1], x[i-2], ..., x[0]
            
            # Build the profile array
            profile = np.zeros(2 * N - 1)
            center = N - 1
            
            # Positive lags
            if i + 1 < N:
                fwd = current[i+1:]  # x[i+1], ..., x[N-1]
                profile[center+1:center+1+len(fwd)] = fwd
            if i > 0:
                bwd = current[i-1::-1]  # x[i-1], x[i-2], ..., x[0]
                profile[center+1:center+1+len(bwd)] += bwd
            
            # Negative lags (mirror)
            profile[:center][::-1] = profile[center+1:]
            profile[center] = 0
            
            delta = -2.0 * v * profile
            new_c += delta
            
            new_max_abs = np.max(np.abs(new_c))
            new_score = new_max_abs / s2
            
            if new_score < current_score - 1e-12:
                if new_score < best_score - best_improvement:
                    best_improvement = current_score - new_score
                    best_flip_idx = i
                    best_new_c = new_c
        
        if best_flip_idx >= 0:
            current[best_flip_idx] *= -1
            current_score = best_score - best_improvement
            c = best_new_c
            best_score = current_score
            best_seq = current.copy()
        else:
            break
    
    return best_seq, best_score

def _pair_flip_search_fast(x, deadline, max_rounds=50):
    """
    Pair flip search to escape local minima.
    """
    best_seq = x.copy()
    best_score = _compute_score(best_seq)
    current = x.copy()
    current_score = best_score
    c = _compute_conv(current)
    s = np.sum(current)
    s2 = s * s
    
    rng = np.random.RandomState(42)
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        best_improvement = 0
        best_pair = (-1, -1)
        
        # Sample pairs
        n_samples = min(500, N * (N-1) // 2)
        
        for _ in range(n_samples):
            if time.time() > deadline:
                break
            
            i, j = rng.choice(N, 2, replace=False)
            vi = current[i]
            vj = current[j]
            
            if vi == 0 or vj == 0:
                continue
            
            new_c = c.copy()
            center = N - 1
            
            # Flip i
            profile_i = np.zeros(2 * N - 1)
            if i + 1 < N:
                profile_i[center+1:center+1+N-1-i] = current[i+1:]
            if i > 0:
                profile_i[center+1:center+1+i] += current[i-1::-1]
            profile_i[:center][::-1] = profile_i[center+1:]
            profile_i[center] = 0
            new_c += -2.0 * vi * profile_i
            
            # Flip j
            profile_j = np.zeros(2 * N - 1)
            if j + 1 < N:
                profile_j[center+1:center+1+N-1-j] = current[j+1:]
            if j > 0:
                profile_j[center+1:center+1+j] += current[j-1::-1]
            profile_j[:center][::-1] = profile_j[center+1:]
            profile_j[center] = 0
            new_c += -2.0 * vj * profile_j
            
            # Correction for interaction at lag |i-j|
            # When we flip both i and j, the cross term at lag k=|i-j|
            # involves x[i]*x[j] which changes sign twice (so no net change)
            # Actually, the autoconvolution term c[k] includes x[i]*x[i+k].
            # If i+k = j, then this term is x[i]*x[j] which becomes (-vi)*(-vj) = vi*vj (unchanged).
            # So no correction needed for the product term.
            # But the individual terms x[i]*x[i+k] and x[j]*x[j+k] are handled by the profiles.
            # The issue is that when computing profile_i, we used the OLD value of x[j].
            # After flipping j, x[j] changes, so the profile_i at lag (j-i) should use new x[j].
            # This is a second-order correction; we ignore it for speed.
            
            new_max_abs = np.max(np.abs(new_c))
            new_score = new_max_abs / s2
            
            if new_score < current_score - 1e-12:
                if new_score < best_score - best_improvement:
                    best_improvement = current_score - new_score
                    best_pair = (i, j)
        
        if best_pair != (-1, -1):
            i, j = best_pair
            current[i] *= -1
            current[j] *= -1
            current_score = best_score - best_improvement
            c = _compute_conv(current)
            s = np.sum(current)
            s2 = s * s
            best_score = current_score
            best_seq = current.copy()
        else:
            break
    
    return best_seq, best_score

def _m_sequence_pattern(n):
    """Generate an M-sequence (maximal length LFSR sequence) of length n."""
    # Use a simple LFSR with a known primitive polynomial
    # For n up to 400, we can use a 9-bit LFSR (length 511) and truncate
    # Or use a simpler approach: generate a pseudo-random ±1 sequence with good autocorrelation
    
    # Actually, let's use a known good construction:
    # Barker-like or M-sequence
    # For simplicity, use a periodic sequence based on quadratic residues
    
    # Quadratic residue sequence: x[i] = 1 if i^2 mod p is a quadratic residue, else -1
    # For p = 397 (prime), this gives a good sequence
    p = 397
    x = np.zeros(n, dtype=float)
    for i in range(n):
        # Use Legendre symbol: (i/p) = i^((p-1)/2) mod p
        if i == 0:
            x[i] = 1
        else:
            x[i] = 1 if pow(i, (p-1)//2, p) == 1 else -1
    return x[:n]

def _random_sequence(rng, n, frac_neg=0.5):
    """Generate a random ±1 sequence with given fraction of negatives."""
    x = np.ones(n)
    n_neg = int(frac_neg * n)
    if n_neg > 0:
        idx = rng.choice(n, n_neg, replace=False)
        x[idx] = -1
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    best_seq = None
    best_score = float('inf')
    
    def update_best(x):
        nonlocal best_seq, best_score
        s = _compute_score(x)
        if s < best_score:
            best_score = s
            best_seq = x.copy()
    
    # 1. Try M-sequence based constructions
    if time.time() < deadline - 60:
        x_m = _m_sequence_pattern(N)
        update_best(x_m)
        
        # Try negated and shifted versions
        for shift in range(0, N, 10):
            x_shifted = np.roll(x_m, shift)
            update_best(x_shifted)
            x_neg = -x_shifted
            update_best(x_neg)
    
    # 2. Random initializations with spectral descent
    rng = np.random.RandomState(123)
    
    for trial in range(5):
        if time.time() > deadline - 40:
            break
        
        frac = 0.3 + 0.1 * trial
        x_rand = _random_sequence(rng, N, frac)
        update_best(x_rand)
        
        # Spectral descent to flatten the spectrum
        x_spectral = _spectral_descent(x_rand, deadline - 40, n_steps=500, lr=0.3)
        update_best(x_spectral)
        
        # Local search on the spectral result
        if time.time() < deadline - 20:
            x_ls, score_ls = _local_search_fast(x_spectral, deadline - 20, max_rounds=50)
            update_best(x_ls)
    
    # 3. Start from the best so far and do more local search
    if best_seq is not None and time.time() < deadline - 15:
        x_ls, score_ls = _local_search_fast(best_seq, deadline - 15, max_rounds=100)
        update_best(x_ls)
    
    # 4. Pair flip search
    if best_seq is not None and time.time() < deadline - 10:
        x_pf, score_pf = _pair_flip_search_fast(best_seq, deadline - 10, max_rounds=30)
        update_best(x_pf)
    
    # 5. Final polish
    if best_seq is not None and time.time() < deadline:
        x_polish, score_polish = _local_search_fast(best_seq, deadline, max_rounds=50)
        update_best(x_polish)
    
    # Fallback: all ones
    if best_seq is None:
        best_seq = np.ones(N)
    
    return best_seq
# EVOLVE-BLOCK-END

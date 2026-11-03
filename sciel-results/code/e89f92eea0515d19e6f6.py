"""Spectral-design local search: minimize max|autoconv| via flat-spectrum initialization + fast flip deltas."""
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

def _flip_delta_vec(x, i):
    """
    Vectorized: return delta array of length 2N-1 for flipping x[i].
    delta[k_idx] = -2*x[i]*(x[i+k]+x[i-k]) for k != 0, 0 for k=0.
    k_idx = N-1+k, k in [-(N-1), N-1].
    """
    center = N - 1
    n_c = 2 * N - 1
    delta = np.zeros(n_c)
    v = x[i]
    # Positive lags k=1..N-1
    for k in range(1, N):
        val = 0.0
        if i + k < N:
            val += x[i + k]
        if i - k >= 0:
            val += x[i - k]
        if val != 0.0:
            d = -2.0 * v * val
            delta[center + k] = d
            delta[center - k] = d
    return delta

def _fast_local_search(x, deadline, max_rounds=200):
    """
    Fast local search: each round scans all N positions, computes flip delta,
    applies best improving flip. Uses precomputed delta vectors per position.
    """
    best_seq = x.copy()
    best_score = _compute_score(best_seq)
    current = x.copy()
    current_score = best_score
    c = _compute_conv(current)
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        s = np.sum(current)
        s2 = s * s
        
        best_improvement = 0.0
        best_flip_idx = -1
        best_new_c = None
        
        for i in range(N):
            if time.time() > deadline:
                break
            
            delta = _flip_delta_vec(current, i)
            new_c = c + delta
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

def _spectral_init(rng, n_init=20):
    """
    Generate initial sequences with near-flat DFT magnitude.
    Strategy: start with random phases at constant magnitude, then threshold to ±1.
    Also try M-sequences and other structured sequences.
    """
    candidates = []
    
    # M-sequences (maximal-length LFSR sequences)
    # For length 2^m - 1, but we need N=400, so use periodic extension
    for m in range(5, 10):  # lengths 31, 63, 127, 255, 511
        if time.time() > 60:
            break
        # Generate M-sequence of length 2^m - 1
        length = (1 << m) - 1
        # Simple LFSR-based M-sequence
        # Use a primitive polynomial; for simplicity, use a known one
        # m=5: x^5+x^2+1, m=6: x^6+x+1, m=7: x^7+x^3+1, m=8: x^8+x^4+x^3+x^2+1, m=9: x^9+x^4+x^3+x+1
        polys = {
            5: [5, 2, 0],   # x^5+x^2+1
            6: [6, 1, 0],   # x^6+x+1
            7: [7, 3, 0],   # x^7+x^3+1
            8: [8, 4, 3, 2, 0],  # x^8+x^4+x^3+x^2+1
            9: [9, 4, 3, 1, 0],  # x^9+x^4+x^3+x+1
        }
        if m not in polys:
            continue
        taps = polys[m]
        # Generate M-sequence
        state = 1
        seq = []
        for _ in range(length):
            seq.append(1 if (state & 1) else -1)
            # LFSR step
            feedback = 0
            for t in taps:
                if t < m:
                    feedback ^= (state >> t) & 1
            state = (state >> 1) | (feedback << (m - 1))
        
        # Extend to length N by tiling
        x_m = np.array(seq)
        x_full = np.tile(x_m, N // length + 1)[:N]
        candidates.append(x_full)
    
    # Random flat-spectrum initialization
    for trial in range(n_init):
        if time.time() > 60:
            break
        # Start with random ±1
        x_rand = rng.choice([-1, 1], size=N)
        # Iterate: compute DFT, adjust to flatten magnitude, threshold
        for _ in range(10):
            X = np.fft.fft(x_rand)
            mag = np.abs(X)
            # Target: constant magnitude
            target_mag = np.sqrt(N)
            # Phase from current, magnitude = target
            phase = np.angle(X)
            X_new = target_mag * np.exp(1j * phase)
            x_new = np.real(np.fft.ifft(X_new))
            # Threshold to ±1
            x_rand = np.sign(x_new)
            x_rand[x_rand == 0] = 1
        candidates.append(x_rand)
    
    # Balanced random sequences (equal ±1)
    for trial in range(10):
        if time.time() > 60:
            break
        n_pos = N // 2
        x_bal = np.ones(N)
        neg_idx = rng.choice(N, N - n_pos, replace=False)
        x_bal[neg_idx] = -1
        candidates.append(x_bal)
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    rng = np.random.RandomState(42)
    
    # Generate spectral initializations
    candidates = _spectral_init(rng, n_init=15)
    
    # Add all-ones as baseline
    candidates.append(np.ones(N))
    
    # Evaluate all candidates
    best_seq = None
    best_score = float('inf')
    for x_try in candidates:
        if time.time() > deadline:
            break
        s = _compute_score(x_try)
        if s < best_score:
            best_score = s
            best_seq = x_try.copy()
    
    if best_seq is None:
        best_seq = np.ones(N)
        best_score = _compute_score(best_seq)
    
    # Run fast local search from best
    if time.time() < deadline - 5:
        best_seq, best_score = _fast_local_search(best_seq, deadline - 5)
    
    # Try local search from a few other good candidates
    if time.time() < deadline - 5:
        candidates_sorted = sorted(candidates, key=lambda x: _compute_score(x))
        for x_try in candidates_sorted[:3]:
            if time.time() > deadline - 5:
                break
            if np.array_equal(x_try, best_seq):
                continue
            seq2, score2 = _fast_local_search(x_try, deadline - 5)
            if score2 < best_score:
                best_score = score2
                best_seq = seq2
    
    return best_seq
# EVOLVE-BLOCK-END

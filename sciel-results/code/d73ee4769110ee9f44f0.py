"""Combined program: Spectral shaping + vectorized multi-flip local search."""
import numpy as np
import time
import random
from scipy.signal import fftconvolve

N = 400

# EVOLVE-BLOCK-START
def _score_sequence(s):
    """True score: max|autoconv| / (sum s)^2."""
    sq_int = (np.sum(s)) ** 2
    if sq_int < 1e-10:
        return np.inf
    ac = fftconvolve(s, s, mode='full')
    max_ac = np.max(np.abs(ac))
    return max_ac / sq_int

def _spectral_shaping_init(rng, n_iter=200):
    """
    Initialize a binary sequence by iterative spectral shaping.
    We want to minimize max|ac|, which relates to flattening |H(f)|^2.
    Use iterative thresholding: project onto binary constraint while
    trying to flatten the power spectrum.
    """
    n = N
    # Start with random binary
    h = rng.choice([-1.0, 1.0], size=n)
    
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    for it in range(n_iter):
        # Compute power spectrum
        H = np.fft.rfft(h, fsize)
        P = np.abs(H) ** 2
        
        # Target: flatten the spectrum (minimize variance of P)
        # We want to reduce peaks in P
        # Use water-filling style: reduce high-P bins, increase low-P bins
        P_mean = np.mean(P)
        P_std = np.std(P)
        
        if P_std < 1e-8:
            break
            
        # Create modified spectrum: compress dynamic range
        alpha = 0.5  # compression factor
        P_mod = P_mean + (P - P_mean) * alpha
        
        # Ensure non-negative
        P_mod = np.maximum(P_mod, 0)
        
        # Take inverse FFT to get new time-domain signal
        # H_mod = sqrt(P_mod) * exp(i*angle(H))
        H_mod = np.sqrt(P_mod) * np.exp(1j * np.angle(H))
        h_new = np.fft.irfft(H_mod, fsize)[:n]
        
        # Threshold to binary
        h = np.sign(h_new)
        h[h == 0] = 1.0
        
    return h

def _vectorized_flip_scores(cur, cur_ac, cur_sum, indices):
    """
    Vectorized evaluation of multiple single-flip scores.
    Returns array of scores for flipping each index in indices.
    """
    n = len(cur)
    size = 2 * n - 1
    scores = np.empty(len(indices))
    
    for i, j in enumerate(indices):
        val = cur[j]
        delta = -2.0 * val
        new_sum = cur_sum + delta
        nsq = new_sum ** 2
        if nsq < 1e-10:
            scores[i] = np.inf
            continue
        
        # Compute new ac efficiently
        # ac_new = ac + 2*delta*cur (shifted) + delta^2 at position j
        # We need max|ac_new|
        # Instead of copying full ac, compute the change in max
        # For speed, approximate by checking if the flip affects the max
        
        # Full computation for correctness
        new_ac = cur_ac.copy()
        new_ac[j:j+n] += 2.0 * delta * cur
        new_ac[j] += delta * delta
        scores[i] = float(np.max(np.abs(new_ac)) / nsq)
    
    return scores

def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin
    
    def _now():
        return time.time()
        
    rng = np.random.RandomState(42)
    
    best_score = np.inf
    best_h = None
    
    # Phase 1: Spectral Shaping Initialization (multiple variants)
    n_shaping_restarts = 5
    for r in range(n_shaping_restarts):
        if _now() > deadline - 60:
            break
            
        # Different initialization strategies
        if r == 0:
            h = _spectral_shaping_init(rng, n_iter=300)
        elif r == 1:
            # Start with alternating pattern
            h = np.ones(N)
            h[::2] = -1.0
            h = _spectral_shaping_init(rng, n_iter=300)
        elif r == 2:
            # Start with random
            h = rng.choice([-1.0, 1.0], size=N)
            h = _spectral_shaping_init(rng, n_iter=400)
        elif r == 3:
            # Use a known good structure: Zadoff-Chu like
            n = N
            q = 17  # prime-ish
            h = np.exp(2j * np.pi * (np.arange(n) ** 2) / (2 * q))
            h = np.sign(h.real)
            h[h == 0] = 1.0
            h = _spectral_shaping_init(rng, n_iter=300)
        else:
            # Pure random
            h = rng.choice([-1.0, 1.0], size=N)
        
        # Ensure non-zero sum
        if np.sum(h) == 0:
            h[0] = 1.0
            if np.sum(h) == 0:
                h[1] = 1.0
        
        sc = _score_sequence(h)
        if sc < best_score:
            best_score = sc
            best_h = h.copy()
        
        # Quick local polish on this candidate
        cur = h.copy()
        cur_ac = fftconvolve(cur, cur, mode='full')
        cur_sum = float(np.sum(cur))
        cur_score = sc
        
        # Fast greedy: try each position once
        for j in range(N):
            if _now() > deadline - 60:
                break
            val = cur[j]
            delta = -2.0 * val
            new_sum = cur_sum + delta
            nsq = new_sum ** 2
            if nsq < 1e-10:
                continue
            new_ac = cur_ac.copy()
            new_ac[j:j+N] += 2.0 * delta * cur
            new_ac[j] += delta * delta
            new_score = float(np.max(np.abs(new_ac)) / nsq)
            if new_score < cur_score - 1e-12:
                cur_ac = new_ac
                cur[j] = -val
                cur_sum = new_sum
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_h = cur.copy()
        
        # If we found a good one, do more intensive search
        if best_score < 0.01:
            break
    
    # If no good result, fallback
    if best_h is None:
        best_h = np.ones(N)
        best_score = _score_sequence(best_h)
    
    # Phase 2: Intensive Local Search with Multi-Flip Moves
    cur = best_h.copy()
    cur_ac = fftconvolve(cur, cur, mode='full')
    cur_sum = float(np.sum(cur))
    cur_score = best_score
    
    # Batch flip evaluation: evaluate multiple flips at once
    def _batch_flip_search(cur, cur_ac, cur_sum, cur_score, max_passes=100, batch_size=50):
        """Local search with batched flip evaluation."""
        n = len(cur)
        for _ in range(max_passes):
            if _now() > deadline - 20:
                break
                
            # Sample a random subset of positions to try
            if batch_size >= n:
                indices = list(range(n))
            else:
                indices = list(rng.choice(n, size=batch_size, replace=False))
            
            # Evaluate all flips in the batch
            best_j = -1
            best_js = cur_score
            best_new_sum = cur_sum
            best_new_ac = None
            
            for j in indices:
                val = cur[j]
                delta = -2.0 * val
                new_sum = cur_sum + delta
                nsq = new_sum ** 2
                if nsq < 1e-10:
                    continue
                new_ac = cur_ac.copy()
                new_ac[j:j+n] += 2.0 * delta * cur
                new_ac[j] += delta * delta
                ns = float(np.max(np.abs(new_ac)) / nsq)
                if ns < best_js:
                    best_js = ns
                    best_j = j
                    best_new_sum = new_sum
                    best_new_ac = new_ac
                    
            if best_j == -1 or best_js >= cur_score - 1e-12:
                # Try all positions if no improvement found in batch
                if batch_size < n:
                    for j in range(n):
                        val = cur[j]
                        delta = -2.0 * val
                        new_sum = cur_sum + delta
                        nsq = new_sum ** 2
                        if nsq < 1e-10:
                            continue
                        new_ac = cur_ac.copy()
                        new_ac[j:j+n] += 2.0 * delta * cur
                        new_ac[j] += delta * delta
                        ns = float(np.max(np.abs(new_ac)) / nsq)
                        if ns < best_js:
                            best_js = ns
                            best_j = j
                            best_new_sum = new_sum
                            best_new_ac = new_ac
                    if best_j == -1 or best_js >= cur_score - 1e-12:
                        break
                else:
                    break
                    
            # Apply the best flip
            cur_ac = best_new_ac
            cur[best_j] = -cur[best_j]
            cur_sum = best_new_sum
            cur_score = best_js
            
            if cur_score < best_score:
                best_score = cur_score
                best_h = cur.copy()
                
        return cur, cur_ac, cur_sum, cur_score, best_score, best_h
    
    cur, cur_ac, cur_sum, cur_score, best_score, best_h = _batch_flip_search(
        cur, cur_ac, cur_sum, cur_score, max_passes=200, batch_size=100
    )
    
    # Phase 3: Simulated Annealing for escape from local minima
    if _now() < deadline - 10:
        T_start = 0.01
        T_min = 1e-7
        sa_steps = 50000
        time_left = deadline - 5 - _now()
        
        for it in range(sa_steps):
            if _now() > deadline - 2:
                break
                
            idx = rng.randint(N)
            val = cur[idx]
            delta = -2.0 * val
            new_sum = cur_sum + delta
            nsq = new_sum ** 2
            if nsq < 1e-10:
                continue
            
            new_ac = cur_ac.copy()
            new_ac[idx:idx+N] += 2.0 * delta * cur
            new_ac[idx] += delta * delta
            new_score = float(np.max(np.abs(new_ac)) / nsq)
            
            d = new_score - cur_score
            T = max(T_start * (0.9995 ** it), T_min)
            
            if d < 0 or rng.rand() < np.exp(-d / T):
                cur_ac = new_ac
                cur[idx] = -val
                cur_sum = new_sum
                cur_score = new_score
                
                if cur_score < best_score:
                    best_score = cur_score
                    best_h = cur.copy()
        
        # Final greedy polish after SA
        if _now() < deadline - 3:
            for j in range(N):
                if _now() > deadline - 2:
                    break
                val = cur[j]
                delta = -2.0 * val
                new_sum = cur_sum + delta
                nsq = new_sum ** 2
                if nsq < 1e-10:
                    continue
                new_ac = cur_ac.copy()
                new_ac[j:j+N] += 2.0 * delta * cur
                new_ac[j] += delta * delta
                new_score = float(np.max(np.abs(new_ac)) / nsq)
                if new_score < cur_score - 1e-12:
                    cur_ac = new_ac
                    cur[j] = -val
                    cur_sum = new_sum
                    cur_score = new_score
                    if cur_score < best_score:
                        best_score = cur_score
                        best_h = cur.copy()
    
    return best_h

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import random
import itertools

def _max_overlap(h):
    """Compute the maximum overlap of h with shifted 1-h using FFT.
    
    The overlap at shift k is sum_i h[i] * (1-h[(i+k) % N])
    We want to minimize the maximum over all shifts k.
    """
    N = len(h)
    one_minus_h = 1.0 - h
    # Use FFT for circular cross-correlation
    # overlap[k] = sum_i h[i] * one_minus_h[(i+k) % N]
    # This is the circular cross-correlation of h and one_minus_h
    H = np.fft.fft(h)
    O = np.fft.fft(one_minus_h)
    # Cross-correlation: ifft(fft(h) * conj(fft(one_minus_h))) gives sum h[i]*one_minus_h[i+k]
    # Actually, let's think carefully.
    # We want overlap[k] = sum_i h[i] * one_minus_h[(i+k) % N]
    # This is the circular cross-correlation.
    # np.fft.ifft(H * np.conj(O)) gives sum_i h[(i-k)%N] * one_minus_h[i]
    # Let me verify: ifft(fft(a)*conj(fft(b)))[k] = sum_n a[(n-k)%N] * b[n]
    # So we want sum_i h[i] * one_minus_h[(i+k)%N]
    # Let j = i+k, then i = j-k, sum_j h[(j-k)%N] * one_minus_h[j]
    # This is ifft(fft(h) * conj(fft(one_minus_h)))[k]... wait, let me re-derive.
    # Actually: sum_i h[i] * one_minus_h[(i+k)%N] 
    # Let m = i+k mod N, so i = m-k mod N
    # = sum_m h[(m-k)%N] * one_minus_h[m]
    # The circular cross-correlation (h ⊛ one_minus_h)[k] = sum_m h[(m-k)%N] * one_minus_h[m]
    # And this equals ifft(fft(h) * conj(fft(one_minus_h)))... 
    # Let me just use a direct but vectorized approach for N=100 which is fast enough.
    # Actually for N=100, even O(N^2) is only 10000 ops, which is fast.
    # The issue was the Python loop. Let's vectorize.
    # Create a circulant matrix approach or use rolling
    # overlap[k] = dot(h, roll(one_minus_h, k))
    # We can compute all at once using matrix operations
    # Or just use a simple vectorized loop - for N=100 it's fine
    overlaps = np.array([np.dot(h, np.roll(one_minus_h, k)) for k in range(N)])
    return np.max(overlaps)

def _score(h):
    """Score is the upper bound on C5, which is the max overlap."""
    return _max_overlap(h)

def _construct_from_pattern(pattern, N):
    """Construct h from a repeating pattern, scaled to have sum = N/2."""
    # Pattern is a sequence of values in [0,1]
    # We tile it to length N, then scale
    if len(pattern) == 0:
        return np.full(N, 0.5)
    # Tile pattern to length N
    repeats = (N // len(pattern)) + 1
    tiled = np.tile(pattern, repeats)[:N]
    # Scale to have sum exactly N/2
    current_sum = np.sum(tiled)
    if current_sum > 0:
        scaled = tiled * (N / 2.0) / current_sum
    else:
        scaled = np.full(N, 0.5)
    # Clip to [0,1]
    scaled = np.clip(scaled, 0, 1)
    # Adjust to ensure exact sum
    # If sum is off, adjust elements
    diff = N / 2.0 - np.sum(scaled)
    if abs(diff) > 1e-12:
        # Adjust one element
        if diff > 0:
            # Add to an element that can accept it
            for i in range(N):
                if scaled[i] < 1.0 - 1e-12:
                    add = min(diff, 1.0 - scaled[i])
                    scaled[i] += add
                    diff -= add
                    if abs(diff) < 1e-12:
                        break
        else:
            # Subtract from an element
            for i in range(N):
                if scaled[i] > 1e-12:
                    sub = min(-diff, scaled[i])
                    scaled[i] -= sub
                    diff += sub
                    if abs(diff) < 1e-12:
                        break
    return scaled

def _try_alternating(N):
    """Try alternating high/low values."""
    # h[i] = a if i even, b if i odd
    # Sum constraint: (N/2)*a + (N/2)*b = N/2 => a + b = 1
    # Overlap at shift k:
    # If k even: overlap = sum h[i]*(1-h[i+k])
    #   h[i+k] has same parity as h[i]
    #   overlap = (N/2)*a*(1-a) + (N/2)*b*(1-b) = (N/2)*a*(1-a) + (N/2)*(1-a)*a = N*a*(1-a)
    # If k odd: h[i+k] has opposite parity
    #   overlap = (N/2)*a*(1-b) + (N/2)*b*(1-a) = (N/2)*a*a + (N/2)*(1-a)*(1-a)
    #   = (N/2)*(a^2 + (1-a)^2)
    # Max overlap = max(N*a*(1-a), (N/2)*(a^2+(1-a)^2))
    # Let's find optimal a
    best_a = 0.5
    best_val = float('inf')
    for a in np.linspace(0, 1, 1000):
        val_even = N * a * (1-a)
        val_odd = (N/2) * (a*a + (1-a)*(1-a))
        val = max(val_even, val_odd)
        if val < best_val:
            best_val = val
            best_a = a
    h = np.zeros(N)
    h[0::2] = best_a
    h[1::2] = 1 - best_a
    # Ensure exact sum
    diff = N/2.0 - np.sum(h)
    if abs(diff) > 1e-12:
        h[0] += diff
        h[0] = np.clip(h[0], 0, 1)
    return h

def _try_periodic(N, period):
    """Try periodic pattern with given period."""
    if period <= 0 or period > N:
        return None
    # Define pattern of length 'period'
    # Sum constraint: sum(pattern) * (N//period) + partial = N/2
    # For simplicity, assume N is divisible by period
    if N % period != 0:
        return None
    repeats = N // period
    # sum(pattern) * repeats = N/2 => sum(pattern) = N/(2*repeats) = period/2
    target_sum = period / 2.0
    
    # Try various patterns
    best_pattern = None
    best_score = float('inf')
    
    # Uniform pattern
    uniform = np.full(period, target_sum / period)
    h = np.tile(uniform, repeats)
    s = _score(h)
    if s < best_score:
        best_score = s
        best_pattern = h.copy()
    
    # Alternating within period
    if period >= 2:
        a = target_sum / 2.0
        pattern = np.zeros(period)
        pattern[0::2] = a
        pattern[1::2] = a  # same, so uniform
        # Try different
        pattern = np.zeros(period)
        pattern[0] = target_sum
        # Rest zero
        h = np.tile(pattern, repeats)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_pattern = h.copy()
    
    return best_pattern

def _local_search(h, max_iter=1000, step_size=0.05):
    """Local search to improve h."""
    N = len(h)
    current = h.copy()
    current_score = _score(current)
    
    for it in range(max_iter):
        # Try perturbing one element
        i = random.randint(0, N-1)
        delta = random.uniform(-step_size, step_size)
        new_val = current[i] + delta
        if new_val < 0 or new_val > 1:
            continue
        # Adjust another element to maintain sum
        j = random.randint(0, N-1)
        if j == i:
            continue
        # We need to adjust j so that sum stays N/2
        adj = delta
        new_j = current[j] - adj
        if new_j < 0 or new_j > 1:
            continue
        
        new_h = current.copy()
        new_h[i] = new_val
        new_h[j] = new_j
        
        new_score = _score(new_h)
        if new_score < current_score - 1e-12:
            current = new_h
            current_score = new_score
    
    return current, current_score

def _hill_climbing(h, max_iter=500):
    """Hill climbing - simplified to avoid O(N^3) cost per iteration."""
    N = len(h)
    best = h.copy()
    best_score = _score(best)
    
    for _ in range(max_iter):
        # Pick two random indices to swap-adjust
        i = random.randint(0, N-1)
        j = random.randint(0, N-1)
        if i == j:
            continue
        delta = random.uniform(-0.1, 0.1)
        new_i = best[i] + delta
        new_j = best[j] - delta
        if new_i < 0 or new_i > 1 or new_j < 0 or new_j > 1:
            continue
        
        new_h = best.copy()
        new_h[i] = new_i
        new_h[j] = new_j
        
        s = _score(new_h)
        if s < best_score:
            best = new_h
            best_score = s
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 20s safety margin from 130s limit
    
    random.seed(42)
    np.random.seed(42)
    
    # Start with alternating pattern
    best_h = _try_alternating(N)
    best_score = _score(best_h)
    
    # Try periodic patterns
    for period in [2, 4, 5, 10, 20, 25, 50]:
        if period > N or N % period != 0:
            continue
        pat = _try_periodic(N, period)
        if pat is not None:
            s = _score(pat)
            if s < best_score:
                best_score = s
                best_h = pat
    
    # Local search from alternating
    if time.time() < deadline - 30:
        h1, s1 = _local_search(best_h, max_iter=500, step_size=0.05)
        if s1 < best_score:
            best_score = s1
            best_h = h1
    
    # Hill climbing
    if time.time() < deadline - 15:
        h2, s2 = _hill_climbing(best_h, max_iter=200)
        if s2 < best_score:
            best_score = s2
            best_h = h2
    
    # More local search
    if time.time() < deadline - 5:
        h3, s3 = _local_search(best_h, max_iter=500, step_size=0.02)
        if s3 < best_score:
            best_score = s3
            best_h = h3
    
    # Final hill climbing
    if time.time() < deadline - 2:
        h4, s4 = _hill_climbing(best_h, max_iter=200)
        if s4 < best_score:
            best_score = s4
            best_h = h4
    
    # Ensure exact sum
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-12:
        # Adjust the element with most room
        if diff > 0:
            idx = np.argmin(1.0 - best_h)
            add = min(diff, 1.0 - best_h[idx])
            best_h[idx] += add
            diff -= add
            if abs(diff) > 1e-12:
                idx2 = np.argmin(1.0 - best_h)
                best_h[idx2] += diff
        else:
            idx = np.argmax(best_h)
            sub = min(-diff, best_h[idx])
            best_h[idx] -= sub
            diff += sub
            if abs(diff) > 1e-12:
                idx2 = np.argmax(best_h)
                best_h[idx2] -= diff
    
    # Final clip
    best_h = np.clip(best_h, 0, 1)
    
    # Final sum correction
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-12:
        best_h[0] += diff
        best_h[0] = np.clip(best_h[0], 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END

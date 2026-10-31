import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = _START + 110.0  # safety margin vs 150 s kill

def _compute_sets_fft(u, max_val):
    """
    Compute |U+U| and |U-U| using FFT.
    u: list of nonnegative integers
    max_val: max(u)
    
    Returns: (size_sum, size_diff)
    """
    if not u:
        return 0, 0
    if len(u) == 1:
        return 1, 1
    
    # We need to represent the indicator function of U.
    # Let f(x) = 1 if x in U, 0 otherwise.
    # Sum set S = {a+b | a,b in U}. The number of representations is the convolution f * f.
    # |S| is the number of indices k where (f*f)[k] > 0.
    
    # Difference set D = {a-b | a,b in U}.
    # Note that a-b can be negative.
    # Let g(x) = f(x).
    # The autocorrelation h(k) = sum_x f(x) f(x+k) counts pairs (x, x+k) such that both are in U.
    # This corresponds to differences k.
    # Specifically, h(k) > 0 iff there exists a, b in U such that a - b = k.
    # So |D| is the number of k in [-max_u, max_u] such that h(k) > 0.
    
    # FFT size needs to be at least 2*max_val + 1 for convolution to avoid wrap-around.
    # Let N = next power of 2 >= 2*max_val + 1.
    
    M = max_val
    N = 1
    while N < 2 * M + 1:
        N *= 2
    
    # Create indicator array
    # f is length N, with f[x] = 1 if x in U
    f = np.zeros(N, dtype=np.float64)
    for x in u:
        if 0 <= x < N:
            f[x] = 1.0
    
    # FFT
    F = np.fft.fft(f)
    
    # Sum set: convolution f * f
    # Inverse FFT of F^2
    conv_sum = np.fft.ifft(F * F).real
    # Due to floating point errors, values might be slightly negative or non-integer.
    # We just need > 0.
    # The valid range for sums is [0, 2*M].
    # conv_sum[k] represents the count of pairs (a,b) with a+b=k.
    # We count k in [0, 2*M] where conv_sum[k] > 0.5 (threshold to handle noise).
    size_sum = np.sum(conv_sum[:2 * M + 1] > 0.5)
    
    # Difference set: autocorrelation
    # Autocorrelation is ifft(F * conj(F))
    auto_diff = np.fft.ifft(F * np.conj(F)).real
    # auto_diff[k] for k in [0, N-1] represents sum_x f(x) f(x-k) ??
    # Let's check the indexing.
    # Standard circular autocorrelation r[k] = sum_x f[x] f[(x+k) mod N].
    # For linear autocorrelation (no wrap-around), we need N large enough.
    # r[k] for k >= 0 corresponds to sum_x f[x] f[x+k].
    # If f[x] and f[x+k] are both 1, then x in U and x+k in U => (x+k) - x = k.
    # So r[k] > 0 implies k is in U-U.
    # What about negative differences?
    # r[k] = sum_x f[x] f[x+k].
    # Let a = x+k, b = x. Then a-b = k.
    # If we want -k, we look at r[N-k]?
    # Actually, the set of differences is symmetric: if d in U-U, then -d in U-U.
    # So |U-U| = 2 * (number of positive differences) + (1 if 0 in U-U).
    # 0 is always in U-U since 0 in U (0-0=0).
    # We need to count distinct non-zero differences.
    # The autocorrelation r[k] for k in [1, M] gives the number of pairs with difference k.
    # If r[k] > 0, then k is a difference.
    # Since differences are symmetric, the total size is 1 + 2 * count(k in [1, M] such that r[k] > 0).
    
    # Let's verify the index mapping for numpy ifft of F*conj(F).
    # numpy.fft.ifft returns array y where y[k] = sum_j x[j] exp(2 pi i j k / N).
    # Here x = F * conj(F).
    # This computes the circular autocorrelation.
    # y[k] = sum_j F[j] conj(F[j]) exp(2 pi i j k / N) ?? No.
    # Let's stick to the property: ifft(fft(f) * conj(fft(f))) is the circular autocorrelation.
    # r[k] = sum_{x=0}^{N-1} f[x] f[(x+k) mod N].
    # For k in [0, M], since f is zero outside [0, M], the wrap-around doesn't affect the sum for the "positive" lag interpretation if we are careful.
    # Actually, standard linear autocorrelation c[k] = sum_x f[x] f[x+k] is obtained from the real part of ifft(fft(f) * conj(fft(f))) at index k.
    # And c[-k] = c[k].
    # So we just need to count k in [1, M] where c[k] > 0.
    
    # The output of ifft is centered? No, numpy ifft output index k corresponds to lag k.
    # c[0] = sum f[x]^2 = |U|.
    # c[k] for k>0 is sum_x f[x] f[x+k].
    
    # We need to be careful with the range.
    # The array auto_diff has length N.
    # Indices 0 to M contain the non-negative lags.
    # Indices N-M to N-1 contain the negative lags (symmetric).
    
    # Count positive differences
    # Threshold 0.5 to avoid noise
    pos_diffs = np.sum(auto_diff[1:M + 1] > 0.5)
    size_diff = 1 + 2 * pos_diffs
    
    return int(size_sum), int(size_diff)

def _score(u):
    """
    Compute the score: 1 + log(|U-U|/|U+U|) / log(2*max U + 1)
    """
    if not u:
        return 0.0
    m = max(u)
    if m < 1:
        return 1.0 # If max is 0, log(1)=0, score=1. But U must contain 0. If U={0}, |U-U|=1, |U+U|=1, ratio=1, log=0. Score=1.
    
    size_sum, size_diff = _compute_sets_fft(u, m)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / float(size_sum)
    
    if ratio <= 0:
        return 0.0
        
    denom = math.log(2 * m + 1)
    if denom == 0:
        return 1.0
        
    return 1.0 + math.log(ratio) / denom

def _random_initial_set(n, max_val):
    """
    Generate a random set of size n containing 0, within [0, max_val].
    """
    if n <= 0:
        return [0]
    if n == 1:
        return [0]
    
    # Ensure 0 is included
    others = random.sample(range(1, max_val + 1), n - 1)
    u = [0] + others
    return sorted(u)

def _construct_golomb(n):
    """
    Construct a Golomb ruler-like set of size n (including 0) with minimal span.
    Uses a greedy approach to place marks so all pairwise differences are distinct.
    """
    if n <= 1:
        return [0]
    if n == 2:
        return [0, 1]
    
    marks = [0]
    used_diffs = set()
    
    for i in range(1, n):
        # Try to place the next mark as close as possible to the previous one
        candidate = marks[-1] + 1
        while True:
            # Check if adding candidate creates a duplicate difference
            valid = True
            for m in marks:
                d = candidate - m
                if d in used_diffs:
                    valid = False
                    break
            if valid:
                break
            candidate += 1
        
        # Add all new differences
        for m in marks:
            used_diffs.add(candidate - m)
        marks.append(candidate)
    
    return marks

def _local_search(u, max_val, iterations, temperature_start=1.0, cooling=0.999):
    """
    Simulated Annealing local search with improved move generation.
    """
    current_u = list(u)
    current_score = _score(current_u)
    best_u = list(current_u)
    best_score = current_score
    
    temperature = temperature_start
    n_current = len(current_u)
    
    # Precompute set for fast membership
    current_set = set(current_u)
    
    for i in range(iterations):
        if time.time() >= _DEADLINE:
            break
            
        # Generate neighbor with more diverse moves
        neighbor = list(current_u)
        move_type = random.random()
        
        if move_type < 0.6:
            # Replace a random element (not 0) with a new value
            if len(neighbor) > 1:
                idx = random.randint(1, len(neighbor) - 1)
                # Try several random values to find one not in the set
                new_val = None
                for _ in range(5):
                    candidate = random.randint(0, max_val)
                    if candidate not in current_set:
                        new_val = candidate
                        break
                if new_val is None:
                    new_val = random.randint(0, max_val)
                
                old_val = neighbor[idx]
                neighbor[idx] = new_val
                neighbor = sorted(set(neighbor))
                
                if len(neighbor) == n_current:
                    # Update set
                    current_set.discard(old_val)
                    current_set.add(new_val)
                else:
                    # Size changed, revert set tracking
                    current_set = set(neighbor)
        elif move_type < 0.8:
            # Add an element
            candidate = random.randint(0, max_val)
            if candidate not in current_set:
                neighbor.append(candidate)
                neighbor = sorted(neighbor)
                current_set.add(candidate)
                n_current = len(neighbor)
            else:
                continue
        else:
            # Remove an element (not 0)
            if len(neighbor) > 1:
                non_zero_indices = [j for j, v in enumerate(neighbor) if v != 0]
                if non_zero_indices:
                    rm_idx = random.choice(non_zero_indices)
                    removed_val = neighbor[rm_idx]
                    neighbor.pop(rm_idx)
                    neighbor = sorted(neighbor)
                    current_set.discard(removed_val)
                    n_current = len(neighbor)
                else:
                    continue
        
        if not neighbor:
            continue
            
        neighbor_score = _score(neighbor)
        
        # Acceptance probability
        delta = neighbor_score - current_score
        if delta > 0 or random.random() < math.exp(delta / max(temperature, 1e-9)):
            current_u = neighbor
            current_score = neighbor_score
            
            if current_score > best_score:
                best_u = list(current_u)
                best_score = current_score
        
        temperature *= cooling
    
    return best_u, best_score

def construct():
    global _DEADLINE
    _DEADLINE = max(time.time(), _START + 110.0)
    
    random.seed(42)
    
    best_u = [0, 1, 2]
    best_s = _score(best_u)
    
    # Strategy: Use Golomb rulers as starting points, then optimize with SA.
    # Golomb rulers have all pairwise differences distinct, so |U-U| = n(n-1)+1 (maximal for size n).
    # We want to minimize |U+U| relative to this.
    
    # Phase 1: Try Golomb rulers of various sizes
    for n in [5, 8, 12, 16, 20, 25, 30, 40, 50]:
        if time.time() >= _DEADLINE - 30.0:
            break
        
        golomb = _construct_golomb(n)
        M = max(golomb)
        
        # Score the Golomb ruler itself
        s_golomb = _score(golomb)
        if s_golomb > best_s:
            best_s = s_golomb
            best_u = golomb
        
        # Optimize from Golomb ruler
        u_opt, s_opt = _local_search(golomb, M, 300, temperature_start=0.8, cooling=0.995)
        if s_opt > best_s:
            best_s = s_opt
            best_u = u_opt
    
    # Phase 2: Random starts with SA
    num_phases = 50
    phase_iter = 400
    
    for phase in range(num_phases):
        if time.time() >= _DEADLINE - 20.0:
            break
        
        # Choose n and M
        n = random.randint(5, 50)
        M = random.randint(n, 8 * n)
        
        # Initial candidate
        u_init = _random_initial_set(n, M)
        
        # Run SA
        u_best, s_best = _local_search(u_init, M, phase_iter, temperature_start=1.0, cooling=0.99)
        
        if s_best > best_s:
            best_s = s_best
            best_u = u_best
    
    # Phase 3: Final polish on the best found
    if time.time() < _DEADLINE - 5.0:
        m_best = max(best_u)
        # Allow slight growth
        M_polish = int(m_best * 1.2) + 10
        u_polished, s_polished = _local_search(best_u, M_polish, 800, temperature_start=0.3, cooling=0.998)
        if s_polished > best_s:
            best_s = s_polished
            best_u = u_polished
            
    # Ensure 0 is in best_u and all are non-negative integers
    best_u = sorted(set(int(x) for x in best_u))
    if 0 not in best_u:
        best_u.append(0)
    
    # Final validation and return
    if not best_u or 0 not in best_u:
        return [0]
    
    return best_u
# EVOLVE-BLOCK-END

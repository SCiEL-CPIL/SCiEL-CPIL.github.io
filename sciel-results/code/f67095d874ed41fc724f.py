import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = _START + 115.0  # Safety margin vs 150 s kill

def _score_from_set(u):
    """
    Compute the score for a set of integers u.
    Score = 1 + log(|U-U|/|U+U|) / log(2*max(U) + 1)
    """
    if not u or len(u) < 2:
        return 0.0
    m = max(u)
    if m < 1:
        return 0.0
    
    u_arr = np.array(u, dtype=np.int64)
    n = len(u_arr)
    
    # Compute all pairwise sums and differences
    sums = u_arr[:, None] + u_arr[None, :]
    diffs = u_arr[:, None] - u_arr[None, :]
    
    unique_sums = np.unique(sums)
    unique_diffs = np.unique(diffs)
    
    if len(unique_sums) == 0:
        return 0.0
        
    ratio = len(unique_diffs) / float(len(unique_sums))
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * m + 1)

def _fft_score(u):
    """
    Fast scoring using FFT to count unique sums and differences.
    This is O(M log M) where M is the span, allowing evaluation of larger sets.
    """
    if not u or len(u) < 2:
        return 0.0
    m = max(u)
    if m < 1:
        return 0.0
    
    u_arr = np.array(u, dtype=np.int64)
    
    # Size of FFT needs to be at least 2*max(U) + 1 for sums if we don't shift?
    # Actually, sum of two numbers in [0, M] is in [0, 2M].
    # Diff of two numbers in [0, M] is in [-M, M].
    
    size = 2 * m + 1
    n_fft = 1
    while n_fft < size:
        n_fft *= 2
        
    # Indicator vector
    f = np.zeros(n_fft, dtype=np.float64)
    f[u_arr] = 1.0
    
    # FFT
    F = np.fft.fft(f)
    
    # Sums: Convolution of f with f.
    conv = np.fft.ifft(F * F).real
    unique_sums = np.sum(conv[:size] > 0.5)
    
    # Diffs: Autocorrelation.
    autocorr = np.fft.ifft(F * np.conj(F)).real
    
    mask = np.zeros(n_fft, dtype=bool)
    mask[0] = True
    mask[1:m+1] = True
    mask[n_fft-m:n_fft] = True
    
    unique_diffs = np.sum(autocorr[mask] > 0.5)
    
    ratio = unique_diffs / float(unique_sums)
    if ratio <= 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / math.log(2 * m + 1)

def _greedy_construct(max_size, max_span, timeout=10.0, use_fft=True):
    """
    Greedy construction: start with {0}, then repeatedly add the element
    that maximizes the score improvement.
    """
    if timeout <= 0:
        return [0, 1], 0.0
        
    score_func = _fft_score if use_fft else _score_from_set
    
    u = [0]
    current_score = 0.0
    
    # We need at least one non-zero element to have a span
    # Try adding 1 initially
    u.append(1)
    current_score = score_func(u)
    
    start_time = time.time()
    
    while len(u) < max_size:
        if time.time() > _DEADLINE - 5.0 or time.time() > start_time + timeout:
            break
            
        best_gain = -1e9
        best_val = None
        best_score = current_score
        
        # Candidate range: we explore values up to max_span
        # To speed up, we can sample candidates if max_span is large
        # But for greedy, we want to be somewhat thorough.
        # Let's try a mix of random sampling and local search around existing elements
        
        # 1. Random sampling
        num_random = 200
        candidates = set()
        
        # Add random candidates
        for _ in range(num_random):
            v = random.randint(1, max_span)
            candidates.add(v)
            
        # Add candidates near existing elements (to fill gaps or extend structure)
        for val in u:
            for delta in range(-5, 6):
                v = val + delta
                if 1 <= v <= max_span:
                    candidates.add(v)
                    
        # Add some "stretch" candidates
        for _ in range(50):
            v = random.randint(max_span // 2, max_span)
            candidates.add(v)
            
        # Evaluate candidates
        for v in candidates:
            if v in u:
                continue
                
            new_u = u + [v]
            new_u.sort()
            s = score_func(new_u)
            
            if s > best_score:
                best_score = s
                best_gain = s - current_score
                best_val = v
                
        if best_val is None:
            break
            
        u.append(best_val)
        u.sort()
        current_score = best_score
        
    return u, current_score

def _local_search(u, timeout=5.0, use_fft=True):
    """
    Improve a set using local search.
    Moves: Replace an element with a nearby one that improves score.
    Uses FFT for faster evaluation if enabled.
    """
    u = sorted(set(u))
    if len(u) < 2:
        return [0, 1], 0.0
    
    score_func = _fft_score if use_fft else _score_from_set
    current_score = score_func(u)
    best_u = u[:]
    best_score = current_score
    start_time = time.time()
    
    n = len(u)
    max_val = max(u)
    
    for step in range(20000):
        if time.time() > start_time + timeout or time.time() > _DEADLINE - 5.0:
            break
            
        if n < 2:
            break
        
        # Pick a random element to perturb (excluding 0)
        idx = random.randint(1, n - 1)
        old_val = u[idx]
        
        # Determine search range
        search_range = max(1, int(max_val * 0.05) + 10)
        
        # Generate diverse candidates
        candidates = []
        
        # Small steps
        step_size = max(1, search_range // 10)
        for delta in range(-search_range, search_range + 1, step_size):
            if delta == 0:
                continue
            new_val = old_val + delta
            if new_val < 1 or new_val in u:
                continue
            candidates.append(new_val)
            
        # Medium jumps
        for _ in range(5):
            jump = random.randint(search_range, search_range * 3)
            direction = random.choice([-1, 1])
            new_val = old_val + direction * jump
            if 1 <= new_val <= max_val * 2 and new_val not in u:
                candidates.append(new_val)
                
        # Large jumps
        for _ in range(3):
            new_val = random.randint(1, max_val * 2)
            if new_val not in u:
                candidates.append(new_val)
                
        if not candidates:
            continue
            
        random.shuffle(candidates)
        
        improved = False
        for new_val in candidates[:15]: 
            new_u = u[:]
            new_u[idx] = new_val
            new_u = sorted(set(new_u))
            
            if len(new_u) != n:
                continue
                
            s = score_func(new_u)
            if s > current_score:
                u = new_u
                current_score = s
                if current_score > best_score:
                    best_score = current_score
                    best_u = u[:]
                improved = True
                new_max = max(u)
                if new_max > max_val:
                    max_val = new_max
                break
                
        if not improved:
            # Try adding an element if size is small
            if n < 50 and random.random() < 0.1:
                # Add a random element in a gap
                if n > 1:
                    idx_add = random.randint(0, n-2)
                    low = u[idx_add]
                    high = u[idx_add+1]
                    if high - low > 1:
                        new_val = random.randint(low+1, high-1)
                        if new_val not in u:
                            test_u = sorted(u + [new_val])
                            s = score_func(test_u)
                            if s > current_score:
                                u = test_u
                                current_score = s
                                if current_score > best_score:
                                    best_score = current_score
                                    best_u = u[:]
                                n += 1
            # Try removing an element if size is large
            elif n > 10 and random.random() < 0.05:
                idx_rm = random.randint(1, n - 1)
                test_u = u[:]
                test_u.pop(idx_rm)
                s = score_func(test_u)
                if s > current_score:
                    u = test_u
                    current_score = s
                    if current_score > best_score:
                        best_score = current_score
                        best_u = u[:]
                    n -= 1
                    max_val = max(u) if u else 1
                    
    return best_u, best_score

def construct():
    global _DEADLINE
    
    random.seed(42)
    np.random.seed(42)
    
    best_u = [0, 1]
    best_s = 0.0
    
    # Strategy:
    # 1. Greedy construction with various parameters (max_size, max_span).
    # 2. Local search refinement on the best greedy results.
    # 3. The greedy approach directly optimizes the score, which should find
    #    better structures than random initialization + local search.
    
    # Parameter sweeps for greedy construction
    # max_size: number of elements to aim for
    # max_span: maximum value allowed
    
    greedy_configs = [
        (20, 1000),
        (30, 2000),
        (40, 5000),
        (50, 10000),
        (60, 20000),
        (80, 50000),
        (100, 100000),
        (20, 500),
        (30, 1000),
        (40, 2000),
        (50, 5000),
        (60, 10000),
    ]
    
    for max_size, max_span in greedy_configs:
        if time.time() > _DEADLINE - 20.0:
            break
            
        # Run greedy construction
        u_greedy, s_greedy = _greedy_construct(max_size, max_span, timeout=5.0, use_fft=True)
        
        if s_greedy > best_s:
            best_s = s_greedy
            best_u = u_greedy[:]
            
        # Refine with local search
        if time.time() < _DEADLINE - 15.0:
            u_refined, s_refined = _local_search(u_greedy, timeout=3.0, use_fft=True)
            if s_refined > best_s:
                best_s = s_refined
                best_u = u_refined[:]
                
    # Final polish with the best found
    if time.time() < _DEADLINE - 5.0:
        # Use exact scoring for final verification and potentially fine-tuning
        if len(best_u) < 50:
            u_final, s_final = _local_search(best_u, timeout=10.0, use_fft=False)
            if s_final > best_s:
                best_s = s_final
                best_u = u_final[:]
        else:
            # For larger sets, stick with FFT optimized version
            s_exact = _score_from_set(best_u)
            if s_exact > best_s:
                best_s = s_exact
            if time.time() < _DEADLINE - 2.0:
                 u_final, s_final = _local_search(best_u, timeout=5.0, use_fft=False)
                 if s_final > best_s:
                     best_s = s_final
                     best_u = u_final[:]

    # Sanity check
    best_u = sorted(set(int(x) for x in best_u))
    if 0 not in best_u:
        best_u.append(0)
    if len(best_u) < 2:
        best_u = [0, 1]
        
    return best_u
# EVOLVE-BLOCK-END

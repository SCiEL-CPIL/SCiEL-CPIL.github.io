"""Rewritten program: FFT-accelerated local search for sumset/difference-set exponent."""
import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    """Construct a finite set U of nonnegative integers containing 0 to maximize
    1 + log(|U-U|/|U+U|)/log(2*max(U)+1)."""
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin before 150s kill

    def score_set_fft(U):
        """Compute the score for a set U using FFT."""
        if not U or 0 not in U:
            return 0.0
        U = list(set(U))
        U.sort()
        n = len(U)
        if n < 2:
            return 0.0
        max_u = U[-1]
        if max_u == 0:
            return 0.0
        
        # We need convolution size to be at least 2*max_u + 1 for sumset
        # and max_u + 1 for difference set (autocorrelation)
        # Use a power of 2 >= 2*max_u + 1
        fft_size = 1
        while fft_size < 2 * max_u + 1:
            fft_size *= 2
        
        # Indicator vector
        f = np.zeros(fft_size, dtype=np.float64)
        for u in U:
            f[u] = 1.0
        
        # FFT
        F = np.fft.rfft(f)
        
        # Sumset: convolution of f with itself
        # conv[k] = sum_i f[i]*f[k-i] for k in [0, 2*max_u]
        conv = np.fft.irfft(F * F, n=fft_size)
        threshold = 0.5
        sum_set_size = int(np.sum(conv[:2*max_u+1] > threshold))
        
        # Difference set: autocorrelation
        # auto[k] = sum_i f[i]*f[i+k] for k in [0, max_u]
        # |U-U| = 1 + 2 * (number of k in [1, max_u] with auto[k] > 0)
        auto = np.fft.irfft(F * np.conj(F), n=fft_size)
        pos_diffs = int(np.sum(auto[1:max_u+1] > threshold))
        diff_set_size = 1 + 2 * pos_diffs
        
        if sum_set_size <= 0:
            return 0.0
        
        ratio = diff_set_size / sum_set_size
        if ratio <= 0:
            return 0.0
        
        log_base = np.log(2 * max_u + 1)
        if log_base <= 0:
            return 0.0
        
        return 1.0 + np.log(ratio) / log_base

    def score_set_direct(U):
        """Direct computation for small sets (fallback)."""
        U = list(set(U))
        U.sort()
        n = len(U)
        if n < 2:
            return 0.0
        max_u = U[-1]
        if max_u == 0:
            return 0.0
        
        sum_set = set()
        for a in U:
            for b in U:
                sum_set.add(a + b)
        
        diff_set = set()
        for a in U:
            for b in U:
                diff_set.add(a - b)
        
        ratio = len(diff_set) / len(sum_set)
        if ratio <= 0:
            return 0.0
        log_base = np.log(2 * max_u + 1)
        if log_base <= 0:
            return 0.0
        return 1.0 + np.log(ratio) / log_base

    def score_set_auto(U):
        """Choose between FFT and direct based on set size."""
        U = list(set(U))
        max_u = max(U) if U else 0
        n = len(U)
        # For very small sets, direct is faster due to FFT overhead
        if n <= 15 and max_u <= 200:
            return score_set_direct(U)
        else:
            return score_set_fft(U)

    # Generate multiple diverse initial candidates
    candidates = []
    
    # Geometric progressions
    for base in [2, 3, 4, 5]:
        cand = [0]
        val = 1
        while val <= 5000:
            cand.append(val)
            val *= base
        if len(cand) >= 3:
            candidates.append(cand[:8])  # Cap length
    
    # Golomb ruler-like: use known good Golomb rulers
    # Small Golomb rulers: {0,1,4,10}, {0,1,4,10,18}, {0,1,4,10,18,27}
    golomb_rulers = [
        [0, 1, 4, 10],
        [0, 1, 4, 10, 18],
        [0, 1, 4, 10, 18, 27],
        [0, 1, 4, 10, 18, 27, 41],
        [0, 1, 4, 10, 18, 27, 41, 53],
        [0, 1, 4, 10, 18, 27, 41, 53, 72],
    ]
    candidates.extend(golomb_rulers)
    
    # Random structured sets
    random.seed(123)
    for _ in range(20):
        n = random.randint(3, 12)
        max_val = random.randint(50, 500)
        cand = [0] + sorted(random.sample(range(1, max_val), n-1))
        candidates.append(cand)
    
    # Find best initial candidate
    best_set = [0, 1, 3]
    best_score = score_set_auto(best_set)
    
    for cand in candidates:
        s = score_set_auto(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    # Local search with simulated annealing-like acceptance
    random.seed(42)
    
    # Determine the maximum value to consider
    max_val = 500  # Start with a reasonable range
    
    # Local search loop
    no_improve_count = 0
    max_no_improve = 3000
    temperature = 0.1  # For occasional non-improving moves
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        if random.random() < 0.35:
            # Add a new element
            new_elem = random.randint(1, max_val)
            if new_elem not in best_set:
                candidate = best_set + [new_elem]
                candidate.sort()
                s = score_set_auto(candidate)
                if s > best_score:
                    best_score = s
                    best_set = candidate
                    no_improve_count = 0
                    # Expand max_val if we added a large element
                    if new_elem > max_val * 0.8:
                        max_val = int(max_val * 1.4)
                else:
                    no_improve_count += 1
                    # Occasionally accept non-improving moves to escape local optima
                    if random.random() < temperature / (1 + no_improve_count / 100):
                        best_set = candidate
                        no_improve_count = 0
            else:
                no_improve_count += 1
        elif random.random() < 0.5:
            # Remove an element (keep 0)
            if len(best_set) > 2:
                removable = [x for x in best_set if x != 0]
                if removable:
                    remove_elem = random.choice(removable)
                    candidate = [x for x in best_set if x != remove_elem]
                    s = score_set_auto(candidate)
                    if s > best_score:
                        best_score = s
                        best_set = candidate
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        else:
            # Swap: remove one, add another
            if len(best_set) > 2:
                removable = [x for x in best_set if x != 0]
                if removable:
                    remove_elem = random.choice(removable)
                    add_elem = random.randint(1, max_val)
                    if add_elem not in best_set:
                        candidate = [x for x in best_set if x != remove_elem]
                        candidate.append(add_elem)
                        candidate.sort()
                        s = score_set_auto(candidate)
                        if s > best_score:
                            best_score = s
                            best_set = candidate
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                            if random.random() < temperature / (1 + no_improve_count / 100):
                                best_set = candidate
                                no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        # Periodically try to expand the range
        if no_improve_count % 300 == 0 and max_val < 10000:
            max_val = int(max_val * 1.3)
        
        # Decay temperature
        temperature = max(0.01, temperature * 0.995)
    
    # Final polish: try adding elements in gaps
    if time.time() < deadline:
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            best_set_sorted = sorted(best_set)
            found = False
            for i in range(len(best_set_sorted) - 1):
                gap = best_set_sorted[i+1] - best_set_sorted[i]
                if gap > 2:
                    mid = best_set_sorted[i] + random.randint(1, gap - 1)
                    if mid not in best_set:
                        candidate = best_set + [mid]
                        candidate.sort()
                        s = score_set_auto(candidate)
                        if s > best_score:
                            best_score = s
                            best_set = candidate
                            found = True
                            break
            # Also try extending beyond current max
            if not found and max(best_set) < max_val:
                new_elem = random.randint(max(best_set) + 1, max_val)
                if new_elem not in best_set:
                    candidate = best_set + [new_elem]
                    candidate.sort()
                    s = score_set_auto(candidate)
                    if s > best_score:
                        best_score = s
                        best_set = candidate
    
    # Ensure 0 is in the set
    if 0 not in best_set:
        best_set.append(0)
    
    return sorted(best_set)
# EVOLVE-BLOCK-END

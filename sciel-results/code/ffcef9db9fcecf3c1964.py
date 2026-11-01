import math
import time
import numpy as np
from numpy.fft import rfft, irfft

def construct():
    deadline = time.time() + 145  # safety margin before 150s kill
    
    def compute_score(u):
        """Compute score using FFT for sumset and diffset sizes."""
        max_u = max(u)
        m = 2 * max_u + 1
        
        if m <= 1:
            return 1.0
        
        n = len(u)
        
        # For small sets, use direct computation
        if n <= 60:
            sumset = set()
            for a in u:
                for b in u:
                    sumset.add(a + b)
            diffset = set()
            for a in u:
                for b in u:
                    diffset.add(a - b)
            r = len(diffset) / len(sumset)
        else:
            # FFT-based computation
            size = 2 * max_u + 1
            fft_size = 1
            while fft_size < 2 * size:
                fft_size *= 2
            
            f = np.zeros(fft_size)
            for x in u:
                f[x] = 1.0
            
            conv = irfft(rfft(f) * rfft(f)).round().astype(int)
            sumset_size = np.count_nonzero(conv[:size])
            
            autocorr = irfft(np.abs(rfft(f))**2).round().astype(int)
            pos_diffs = np.count_nonzero(autocorr[1:max_u+1])
            diffset_size = 1 + 2 * pos_diffs
            
            r = diffset_size / sumset_size
        
        score = 1 + math.log(r) / math.log(m)
        return score
    
    # LFSR-based candidate generation
    def lfsr_candidates(bits, taps):
        """Generate a set from LFSR state positions."""
        state = 1
        positions = [0]
        for i in range(bits):
            # Extract LSB
            positions.append(state)
            # Shift right
            state >>= 1
            # Compute feedback bit
            fb = 0
            for t in taps:
                fb ^= (state >> (t-1)) & 1
            state |= fb << (bits - 1)
        return sorted(set(positions))
    
    # Known Golomb rulers
    golomb_rulers = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 12, 17, 25],
        [0, 1, 4, 10, 12, 17, 25, 34],
        [0, 1, 4, 10, 12, 17, 25, 34, 44],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 67],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 67, 80],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 67, 80, 94],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 67, 80, 94, 109],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 67, 80, 94, 109, 126],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 67, 80, 94, 109, 126, 144],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 67, 80, 94, 109, 126, 144, 163],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 67, 80, 94, 109, 126, 144, 163, 183],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 67, 80, 94, 109, 126, 144, 163, 183, 204],
        [0, 1, 4, 10, 12, 17, 25, 34, 44, 55, 67, 80, 94, 109, 126, 144, 163, 183, 204, 226],
    ]
    
    # LFSR candidates with various tap configurations
    lfsr_candidates_list = []
    for bits in [4, 5, 6, 7, 8]:
        # Primitive polynomials for LFSR
        if bits == 4:
            taps_list = [[1, 3]]
        elif bits == 5:
            taps_list = [[1, 2]]
        elif bits == 6:
            taps_list = [[1, 5], [2, 5]]
        elif bits == 7:
            taps_list = [[1, 3], [1, 4]]
        elif bits == 8:
            taps_list = [[1, 4], [1, 5], [1, 6]]
        else:
            taps_list = []
        
        for taps in taps_list:
            cand = lfsr_candidates(bits, taps)
            if len(cand) > 2:
                lfsr_candidates_list.append(cand)
    
    # Also try some other structures
    extra_candidates = []
    for base in [2, 3]:
        s = [0]
        x = 1
        while x <= 500:
            s.append(x)
            x *= base
        extra_candidates.append(s)
    
    s = [0]
    for i in range(1, 10):
        s.append(1 << i)
    extra_candidates.append(s)
    
    fib = [0, 1]
    while fib[-1] <= 500:
        fib.append(fib[-1] + fib[-2])
    extra_candidates.append(fib)
    
    all_candidates = golomb_rulers + lfsr_candidates_list + extra_candidates
    
    best_set = [0, 1, 3]
    best_score = 0.0
    
    # Evaluate all initial candidates
    for u in all_candidates:
        if time.time() > deadline:
            break
        u = sorted(set(u))
        if 0 not in u:
            u = [0] + u
        score = compute_score(u)
        if score > best_score:
            best_score = score
            best_set = list(u)
    
    # Improved local search with simulated annealing
    current = list(best_set)
    current_score = best_score
    max_val = max(current) if current else 0
    
    # Simulated annealing parameters
    T_start = 0.5
    T_end = 0.01
    cooling = 0.995
    
    attempts = 0
    max_attempts = 10000
    T = T_start
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        T = T_start * (T_end / T_start) ** (attempts / max_attempts)
        
        # Generate a random perturbation
        new_set = list(current)
        
        if max_val < 5000 and len(new_set) < 30:
            # Try adding an element
            if len(new_set) > 0:
                # Choose a position to insert
                if len(new_set) > 1:
                    idx = np.random.randint(0, len(new_set))
                    if idx < len(new_set) - 1:
                        gap_start = new_set[idx] + 1
                        gap_end = new_set[idx+1] - 1
                        if gap_end > gap_start:
                            new_elem = np.random.randint(gap_start, gap_end + 1)
                            if new_elem not in new_set:
                                new_set.append(new_elem)
                                new_set.sort()
                else:
                    new_elem = max_val + 1
                    if new_elem not in new_set:
                        new_set.append(new_elem)
                        new_set.sort()
        
        # Try removing an element
        if len(new_set) > 3 and np.random.random() < 0.3:
            idx = np.random.randint(0, len(new_set))
            if idx != 0:  # Keep 0
                new_set.pop(idx)
                new_set.sort()
        
        # Try replacing an element
        if len(new_set) > 3 and np.random.random() < 0.2:
            idx = np.random.randint(1, len(new_set))  # Don't replace 0
            old_elem = new_set[idx]
            # Try nearby values
            for delta in [-3, -2, -1, 1, 2, 3]:
                new_elem = old_elem + delta
                if new_elem > 0 and new_elem not in new_set:
                    new_set[idx] = new_elem
                    new_set.sort()
                    break
        
        if new_set == current:
            continue
        
        new_score = compute_score(new_set)
        delta_score = new_score - current_score
        
        # Accept if better or with probability exp(delta/T)
        if delta_score > 0 or np.random.random() < math.exp(delta_score / max(T, 1e-10)):
            current = new_set
            current_score = new_score
            max_val = max(current)
            
            if current_score > best_score:
                best_score = current_score
                best_set = list(current)
    
    # Final polish: try removing elements to reduce max_val
    while len(best_set) > 2:
        if time.time() > deadline - 10:
            break
        test_set = best_set[:-1]
        test_score = compute_score(test_set)
        if test_score >= best_score - 1e-9:
            best_set = test_set
            best_score = test_score
        else:
            break
    
    return best_set

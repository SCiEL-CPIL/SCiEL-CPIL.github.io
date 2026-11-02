"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from numpy.fft import fft, ifft

def _score_set_fft(U):
    """Compute the score for a finite set U using FFT for speed."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # Compute sumset using FFT
    # U + U: values range from 0 to 2*max_u
    sum_len = 2 * max_u + 1
    # Pad to next power of 2 for efficient FFT
    fft_len = 1
    while fft_len < 2 * sum_len:
        fft_len *= 2
    
    # Create indicator vector for U
    indicator = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        indicator[u] = 1.0
    
    # FFT to compute convolution (sumset)
    freq = fft(indicator)
    conv = ifft(freq * freq).real
    
    # Count nonzero entries in sumset (values 0 to 2*max_u)
    sumset_size = np.sum(conv[:sum_len] > 1e-9)
    
    # Compute difference set using FFT
    # U - U: values range from -max_u to max_u
    # We can compute this by reversing and convolving
    rev_indicator = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        rev_indicator[fft_len - 1 - u] = 1.0
    
    freq_rev = fft(rev_indicator)
    diff_conv = ifft(freq * freq_rev).real
    
    # Difference set values are at positions max_u to 2*max_u in the convolution
    # Actually, let me think about this more carefully.
    # If we compute indicator * reverse(indicator), the result at position k 
    # corresponds to sum of indicator[i] * indicator[k-i] for all i
    # This gives us the autocorrelation, which relates to differences.
    
    # Let me use a different approach for differences:
    # U - U has values from -max_u to max_u
    # We can compute this by noting that |a-b| for a,b in U
    # Actually, let's just compute it directly for smaller sets, or use FFT properly.
    
    # For differences: create a vector where we place U at positions shifted by max_u
    # Then convolve with reversed version
    diff_len = 2 * max_u + 1
    fft_len2 = 1
    while fft_len2 < 2 * diff_len:
        fft_len2 *= 2
    
    # Place U in a vector of length fft_len2, shifted by max_u
    diff_vec = np.zeros(fft_len2, dtype=np.float64)
    for u in U:
        diff_vec[u + max_u] = 1.0
    
    # Reverse it
    diff_vec_rev = diff_vec[::-1]
    
    freq_d = fft(diff_vec)
    freq_dr = fft(diff_vec_rev)
    diff_result = ifft(freq_d * freq_dr).real
    
    # The difference set size is the count of nonzero entries
    # In the convolution of v and reverse(v), the entry at position max_u 
    # corresponds to the number of pairs with difference 0
    # The full difference set spans from -max_u to max_u
    # In the convolution result, differences d appear at position max_u + d
    diffset_size = np.sum(np.abs(diff_result) > 1e-9)
    
    # Actually, let me be more careful. The convolution of v and v_rev where 
    # v has U placed at offset max_u:
    # (v * v_rev)[k] = sum_i v[i] * v_rev[k-i] = sum_i v[i] * v[fft_len2-1-(k-i)]
    # = sum_i v[i] * v[fft_len2-1-k+i]
    
    # This is getting complex. Let me just compute difference set directly for now
    # and use FFT only for sumset, which is the bottleneck.
    
    # Direct difference set computation
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    diffset_size = len(diffset)
    
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom

def _score_set(U):
    """Compute the score for a finite set U containing 0."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0
    
    # For small sets, use direct computation
    if n <= 30:
        sumset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        ratio = len(diffset) / len(sumset)
        if ratio <= 0:
            return 0.0
        log_ratio = math.log(ratio)
        log_denom = math.log(2 * max_u + 1)
        if log_denom == 0:
            return 1.0
        return 1.0 + log_ratio / log_denom
    else:
        return _score_set_fft(U)

def _golomb_ruler(n):
    """Generate a Golomb ruler of order n using greedy construction."""
    if n <= 1:
        return [0]
    known = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 10, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
    }
    if n in known:
        return known[n]
    ruler = [0]
    for i in range(1, n):
        pos = 1
        while True:
            candidate = ruler + [pos]
            diffs = set()
            valid = True
            for j in range(len(candidate)):
                for k in range(j + 1, len(candidate)):
                    d = candidate[k] - candidate[j]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            if valid:
                ruler.append(pos)
                break
            pos += 1
    return ruler

def _local_search(U, max_iters=2000, deadline=None):
    """Improve set U via local search (swap elements)."""
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 100  # search space for swaps
    
    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False
        # Try swapping one element
        for i in range(len(U)):
            if U[i] == 0:
                continue  # must keep 0
            old_val = U[i]
            # Try several new values around old_val and random positions
            candidates = set()
            for offset in range(-20, 21):
                nv = old_val + offset
                if nv > 0:
                    candidates.add(nv)
            # Add some random candidates
            for _ in range(10):
                candidates.add(random.randint(1, upper_bound))
            
            for new_val in sorted(candidates):
                if new_val == old_val:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                new_U_set = set(new_U)
                if len(new_U_set) != len(new_U):
                    continue
                s = _score_set(new_U)
                if s > best_score + 1e-9:
                    best_score = s
                    best_U = new_U[:]
                    U = new_U[:]
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try adding an element
            if len(U) < 200:
                for _ in range(20):
                    new_val = random.randint(1, upper_bound)
                    if new_val in U:
                        continue
                    new_U = U + [new_val]
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
                        break
            
            if not improved:
                # Try removing an element (if size > 2)
                if len(U) > 2:
                    for i in range(len(U)):
                        if U[i] == 0:
                            continue
                        new_U = U[:i] + U[i+1:]
                        s = _score_set(new_U)
                        if s > best_score + 1e-9:
                            best_score = s
                            best_U = new_U[:]
                            U = new_U[:]
                            improved = True
                            break
        
        if not improved:
            # Try a larger random perturbation
            if random.random() < 0.1:
                new_U = U[:]
                # Randomly change a few elements
                for _ in range(min(5, len(new_U))):
                    idx = random.randint(1, len(new_U) - 1)  # skip index 0 (value 0)
                    new_U[idx] = random.randint(1, upper_bound)
                new_U_set = set(new_U)
                if len(new_U_set) == len(new_U):
                    s = _score_set(new_U)
                    if s > best_score + 1e-9:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        improved = True
            
            if not improved:
                break
    
    return sorted(set(best_U)), best_score

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    best_U = [0, 1, 3]
    best_score = _score_set(best_U)
    
    # Try different initial constructions
    candidates = []
    
    # Golomb rulers of various sizes
    for n in range(3, 30):
        if time.time() > deadline - 30:
            break
        gr = _golomb_ruler(n)
        candidates.append(gr)
    
    # Random sets with various sizes
    random.seed(42)
    for trial in range(30):
        if time.time() > deadline - 30:
            break
        n = random.randint(3, 50)
        max_val = random.randint(20, 500)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        U = sorted(U)
        candidates.append(U)
    
    # Structured sets: arithmetic progressions with gaps
    for base in [2, 3, 5, 7, 11]:
        if time.time() > deadline - 30:
            break
        U = [0]
        for i in range(1, 20):
            U.append(i * base)
        candidates.append(U[:min(len(U), 20)])
    
    # Powers-based sets
    for p in [2, 3]:
        if time.time() > deadline - 30:
            break
            U = [0]
            for i in range(1, 15):
                U.append(p ** i)
                if U[-1] > 10000:
                    break
            candidates.append(U)
    
    # Evaluate candidates
    for U in candidates:
        if time.time() > deadline - 20:
            break
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
    
    # Local search on best candidate
    if time.time() < deadline - 5:
        best_U, best_score = _local_search(best_U, max_iters=3000, deadline=deadline - 5)
    
    # Try local search on a few other candidates
    for U in candidates[:8]:
        if time.time() > deadline - 5:
            break
        if U == best_U:
            continue
        s = _score_set(U)
        if s > best_score - 0.02:  # only if close
            improved_U, improved_score = _local_search(U, max_iters=1500, deadline=deadline - 5)
            if improved_score > best_score:
                best_score = improved_score
                best_U = improved_U
    
    return sorted(set(best_U))
# EVOLVE-BLOCK-END

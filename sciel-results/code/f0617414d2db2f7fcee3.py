import numpy as np
import time
import math

def _score_set(U):
    """Compute the score for a given set U (list of nonnegative ints containing 0)."""
    if not U or 0 not in U:
        return 0.0
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Compute |U+U| and |U-U| using FFT
    # U+U: convolve indicator with itself
    # U-U: convolve indicator with reversed indicator
    
    # Use numpy FFT for efficiency
    max_val = max_u
    size = 2 * max_val + 1  # enough for sums up to 2*max_u
    
    # For sumset: indicator array of length max_val+1, convolve with itself
    ind = np.zeros(max_val + 1, dtype=np.float64)
    for x in U:
        ind[x] = 1.0
    
    # Sumset via FFT
    fft_len = 1
    while fft_len < 2 * (max_val + 1):
        fft_len *= 2
    
    ind_fft = np.fft.fft(ind, n=fft_len)
    conv_sum = np.fft.ifft(ind_fft * ind_fft).real
    sumset_size = int(np.sum(conv_sum > 0.5))
    
    # Difference set: U - U = {a - b : a,b in U}, range is [-max_u, max_u]
    # |U-U| = number of distinct values
    # Use convolution: ind * reversed_ind
    ind_rev = ind[::-1]
    ind_rev_fft = np.fft.fft(ind_rev, n=fft_len)
    conv_diff = np.fft.ifft(ind_fft * ind_rev_fft).real
    # conv_diff[i] corresponds to coefficient of x^(i - max_val) in the difference polynomial
    # The difference set size is the number of nonzero coefficients
    diffset_size = int(np.sum(conv_diff > 0.5))
    
    ratio = diffset_size / sumset_size
    if ratio <= 1:
        return 1.0  # base score
    
    denom = math.log(2 * max_u + 1, 2)
    if denom <= 0:
        return 1.0
    
    score = 1.0 + math.log(ratio) / denom
    return score


def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    best_score = 1.0
    best_set = [0, 1, 3]
    
    # Try various structured sets
    # Strategy: mix dense and sparse elements
    
    # Test some known good structures
    candidate_sets = []
    
    # Base-3 like but with modifications
    # For n elements in base b, if b is odd, sumset and diffset sizes relate to carries
    
    # Let's try: U = {0} union {a + i*b : i=0..k} for various a, b, k
    # This gives a block structure
    
    # First, try small exhaustive searches
    # For small max_u, enumerate subsets
    
    # Try sets of form: multiples of some step, plus some offsets
    # Key insight: if U is contained in an arithmetic progression with large common difference,
    # the sumset has structure that may differ from difference set
    
    # Actually, let's think about it more carefully.
    # For U = {0, a, 2a, ..., (n-1)a}: sumset = {0, a, ..., 2(n-1)a}, size = 2n-1
    # diffset = {-(n-1)a, ..., (n-1)a}, size = 2n-1. Ratio = 1.
    
    # For U = {0, a, 2a, ..., (n-1)a} union {b, b+a, ...}: need mixed structure
    
    # Let me try: U where elements are sums of digits in a mixed-radix system
    # where the "sum" representation has more collisions than "difference"
    
    # Alternative: use the fact that for sets with elements having a common divisor d,
    # we can scale down. So assume gcd(U) = 1.
    
    # Let's do a local search starting from various initial configurations
    
    def _try_set(U):
        nonlocal best_score, best_set
        U = sorted(set(U))
        if 0 not in U or len(U) < 2:
            return
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_set = list(U)
    
    # Try structured candidates
    # 1. Sets with a "block" structure: {0} union {M, M+1, ..., M+k}
    for M in range(1, 50):
        for k in range(1, 20):
            U = [0] + list(range(M, M + k + 1))
            _try_set(U)
    
    # 2. Two blocks
    for M1 in range(1, 30):
        for k1 in range(1, 10):
            for M2 in range(M1 + k1 + 1, M1 + k1 + 20):
                for k2 in range(1, 10):
                    U = [0] + list(range(M1, M1 + k1 + 1)) + list(range(M2, M2 + k2 + 1))
                    _try_set(U)
    
    # 3. Geometric-like with perturbations
    for base in range(2, 8):
        for n in range(2, 8):
            U = []
            x = 0
            for i in range(n):
                U.append(x)
                x *= base
            if 0 in U:
                _try_set(U)
    
    # 4. Local search from best found so far
    def _local_search(U, max_iters=5000):
        nonlocal best_score, best_set
        U = sorted(set(U))
        if 0 not in U:
            U = [0] + U
        U = sorted(U)
        
        current_score = _score_set(U)
        if current_score > best_score:
            best_score = current_score
            best_set = list(U)
        
        max_u = max(U)
        n = len(U)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Random move: add, remove, or shift an element
            move_type = np.random.choice(['add', 'remove', 'shift'], p=[0.3, 0.2, 0.5])
            
            new_U = list(U)
            
            if move_type == 'add' and n < 30:
                # Add a new element
                if max_u < 200:
                    new_elem = int(np.random.randint(1, max_u + 10))
                    if new_elem not in new_U:
                        new_U.append(new_elem)
                        new_U = sorted(new_U)
                else:
                    continue
            elif move_type == 'remove' and n > 2:
                # Remove a random element (not 0)
                idx = int(np.random.randint(1, n))
                new_U.pop(idx)
            elif move_type == 'shift':
                # Shift a random element (not 0)
                idx = int(np.random.randint(1, n))
                delta = int(np.random.randint(-5, 6))
                new_val = new_U[idx] + delta
                if new_val >= 0 and new_val != new_U[idx]:
                    # Check no collision
                    if new_val not in new_U:
                        new_U[idx] = new_val
                        new_U = sorted(new_U)
                    else:
                        continue
            
            if len(new_U) < 2 or 0 not in new_U:
                continue
            
            new_score = _score_set(new_U)
            if new_score >= current_score - 0.01:  # allow slight degradation for exploration
                U = new_U
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(U)
        
        return U
    
    # Run local search from multiple starting points
    if time.time() < deadline - 5:
        # Start from best structured candidate
        local_U = list(best_set)
        local_U = _local_search(local_U, max_iters=2000)
    
    if time.time() < deadline - 5:
        # Start from a different structure
        start_U = [0, 2, 5, 11, 23]
        local_U = _local_search(start_U, max_iters=2000)
    
    if time.time() < deadline - 5:
        # Start from dense set
        start_U = list(range(0, 15))
        local_U = _local_search(start_U, max_iters=2000)
    
    if time.time() < deadline - 5:
        # Start from sparse set
        start_U = [0, 3, 9, 27, 81]
        local_U = _local_search(start_U, max_iters=2000)
    
    if time.time() < deadline - 5:
        # More local search from current best
        local_U = list(best_set)
        local_U = _local_search(local_U, max_iters=3000)
    
    return best_set

import time
import math
import random
import numpy as np

def _score(U):
    """Compute the score for set U."""
    U = sorted(set(U))
    if not U or U[0] != 0:
        return -float('inf')
    max_u = max(U)
    if max_u == 0:
        return 1.0
    U_set = set(U)
    sumset = set()
    diffset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
            diffset.add(a - b)
    ratio = len(diffset) / len(sumset)
    if ratio <= 0:
        return -float('inf')
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def _fft_score(U):
    """Compute score using FFT for fast sumset/diffset size computation."""
    U = sorted(set(U))
    if not U or U[0] != 0:
        return -float('inf')
    max_u = max(U)
    if max_u == 0:
        return 1.0
    n = len(U)
    
    # For sumset: use FFT convolution
    # Create indicator array
    size = 2 * max_u + 1
    # Round up to next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < 2 * size:
        fft_size *= 2
    
    indicator = np.zeros(fft_size)
    for x in U:
        indicator[x] = 1.0
    
    # FFT for sumset
    fft_ind = np.fft.fft(indicator)
    conv = np.fft.ifft(fft_ind * fft_ind).real
    # conv[k] >= 0.5 means k is in sumset
    sumset_size = int(np.sum(conv > 0.5))
    
    # For diffset: U-U = {a-b : a,b in U}
    # This is the same as the set of values where cross-correlation is nonzero
    # Actually, diffset size = number of distinct values of a-b
    # We can compute this via convolution of U with reversed U shifted
    # Or more simply: diffset = {d : d = a-b for some a,b in U}
    # The range is from -max_u to max_u
    # Let's compute indicator of U, then convolve with reversed indicator
    
    # Create reversed indicator for difference
    # diffset element d exists if there exist a,b in U with a-b=d
    # This is equivalent to: for each d, check if (indicator * indicator_shifted)(d) > 0
    # Where indicator_shifted is indicator shifted by max_u
    
    # Alternative: use the fact that |U-U| can be computed by checking each difference
    # But for efficiency, use FFT:
    # Let f = indicator of U (length max_u+1)
    # Let g = f reversed (g[i] = f[max_u - i])
    # Then the cross-correlation h = f * g (convolution) gives counts
    # Actually, let me think more carefully.
    
    # The difference set U-U has elements in range [-max_u, max_u]
    # d is in U-U iff there exist a, b in U such that a - b = d
    # This is equivalent to: the number of pairs (a,b) with a-b=d is > 0
    
    # Using FFT: Let f[k] = 1 if k in U, else 0
    # The number of representations of d as a-b is sum over a of f[a] * f[a-d]
    # = (f * f_rev)(d + max_u) where f_rev[i] = f[max_u - i]
    
    f_arr = np.zeros(fft_size)
    for x in U:
        f_arr[x] = 1.0
    
    f_rev = np.zeros(fft_size)
    for x in U:
        f_rev[max_u - x] = 1.0
    
    fft_f = np.fft.fft(f_arr)
    fft_frev = np.fft.fft(f_rev)
    conv_diff = np.fft.ifft(fft_f * fft_frev).real
    
    # d is in [-max_u, max_u], and conv_diff[max_u + d] counts representations
    diffset_size = int(np.sum(conv_diff[max_u - max_u:max_u + max_u + 1] > 0.5))
    
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return -float('inf')
    return 1.0 + math.log(ratio) / math.log(2 * max_u + 1)

def _local_search(U, max_iterations=20000, time_limit=120.0, use_fft=True):
    """Local search to improve the set."""
    start_time = time.time()
    U = sorted(set(U))
    score_fn = _fft_score if use_fft else _score
    current_score = score_fn(U)
    best_U = list(U)
    best_score = current_score
    
    max_u = max(U)
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Try various moves
        moved = False
        max_u = max(U)
        
        # Move 1: Remove an element (not 0)
        if len(U) > 1:
            idx = random.randint(1, len(U) - 1)
            new_U = U[:idx] + U[idx+1:]
            new_score = score_fn(new_U)
            if new_score > current_score + 1e-12:
                U = new_U
                current_score = new_score
                moved = True
        
        # Move 2: Add a new element
        if not moved:
            candidates = []
            for x in U:
                candidates.extend([x-1, x+1, x*2, x+max(1, max_u//10)])
            candidates.extend([max_u + 1, max_u + max(1, max_u//10), max_u * 2, max_u * 3])
            # Add some random candidates in the range
            for _ in range(5):
                candidates.append(random.randint(1, max_u * 3))
            candidates = [x for x in candidates if x >= 1 and x not in U and x <= max_u * 5]
            if candidates:
                new_val = random.choice(candidates)
                new_U = sorted(U + [new_val])
                new_score = score_fn(new_U)
                if new_score > current_score + 1e-12:
                    U = new_U
                    current_score = new_score
                    moved = True
        
        # Move 3: Replace an element
        if not moved and len(U) > 1:
            idx = random.randint(1, len(U) - 1)
            old_val = U[idx]
            candidates = [old_val - 1, old_val + 1, old_val * 2, 
                          old_val // 2 if old_val > 1 else 1,
                          max_u + 1, random.randint(1, max_u * 3),
                          old_val + random.randint(-max_u//5, max_u//5)]
            candidates = [x for x in candidates if x >= 1 and x not in U and x != old_val and x <= max_u * 5]
            if candidates:
                new_val = random.choice(candidates)
                new_U = list(U)
                new_U[idx] = new_val
                new_U = sorted(set(new_U))
                new_score = score_fn(new_U)
                if new_score > current_score + 1e-12:
                    U = new_U
                    current_score = new_score
                    moved = True
        
        if current_score > best_score:
            best_score = current_score
            best_U = list(U)
        
        if not moved:
            # Perturb
            if len(U) > 2:
                idx = random.randint(1, len(U) - 1)
                U = U[:idx] + U[idx+1:]
                current_score = score_fn(U)
            elif len(U) == 2:
                U = [0, random.randint(1, max_u * 3)]
                current_score = score_fn(U)
    
    return best_U, best_score

def construct():
    deadline = time.time() + 135  # Safety margin
    
    # Start with various initial structures
    initial_sets = [
        [0, 1, 3, 9, 27, 81, 243, 729, 2187, 6561, 19683, 59049, 177147, 531441],
        [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
        [0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191],
        [0, 2, 6, 18, 54, 162, 486, 1458, 4374, 13122, 39366, 118098],
        [0, 1, 3, 10, 30, 91, 274, 823, 2470, 7411, 22234],
        [0, 1, 4, 13, 40, 121, 364, 1093, 3280, 9841],
    ]
    
    best_U = initial_sets[0]
    best_score = _fft_score(best_U)
    
    # Find best initial set
    for init in initial_sets:
        if time.time() > deadline:
            break
        s = _fft_score(init)
        if s > best_score:
            best_score = s
            best_U = list(init)
    
    # Local search from best
    remaining = deadline - time.time()
    if remaining > 5:
        U, score = _local_search(best_U, max_iterations=20000, time_limit=min(60, remaining - 5), use_fft=True)
        if score > best_score:
            best_score = score
            best_U = U
    
    # Try more local searches from different starts
    remaining = deadline - time.time()
    if remaining > 10:
        for init in initial_sets:
            if time.time() > deadline - 5:
                break
            time_left = deadline - time.time() - 5
            if time_left < 3:
                break
            U, score = _local_search(init, max_iterations=5000, time_limit=min(20, time_left), use_fft=True)
            if score > best_score:
                best_score = score
                best_U = U
    
    return sorted(set(best_U))


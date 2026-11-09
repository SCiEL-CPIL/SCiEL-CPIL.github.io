"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations

def compute_score(U):
    """Compute the score for a given set U."""
    U = sorted(set(U))
    if not U or U[0] != 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    
    # Compute sumset and difference set sizes
    # Using direct computation for small sets, FFT for larger
    n = len(U)
    if n <= 50:
        sumset = set()
        diffset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
                diffset.add(a - b)
        sum_size = len(sumset)
        diff_size = len(diffset)
    else:
        # Use FFT for larger sets
        M = 2 * max_u + 1
        # Next power of 2 for FFT
        M_fft = 1
        while M_fft < M:
            M_fft *= 2
        
        # Indicator function of U
        f = np.zeros(M_fft)
        for u in U:
            f[u] = 1.0
        
        # Sumset: convolution of f with itself
        f_fft = np.fft.fft(f)
        sum_fft = f_fft * f_fft
        sum_conv = np.fft.ifft(sum_fft).real
        sum_size = int(np.sum(np.round(sum_conv) > 0.5))
        
        # Difference set: convolution of f with reversed f
        f_rev = np.zeros(M_fft)
        for u in U:
            f_rev[M_fft - u] = 1.0
        f_rev_fft = np.fft.fft(f_rev)
        diff_fft = f_fft * f_rev_fft
        diff_conv = np.fft.ifft(diff_fft).real
        diff_size = int(np.sum(np.round(diff_conv) > 0.5))
    
    if sum_size == 0:
        return 0.0
    
    ratio = diff_size / sum_size
    denom = np.log2(2 * max_u + 1)
    if denom == 0:
        return 0.0
    
    score = 1.0 + np.log2(ratio) / denom
    return score

def local_search(initial_set, max_iters=5000, deadline=None):
    """Perform local search to improve the set."""
    current = sorted(set(initial_set))
    if current[0] != 0:
        current[0] = 0
    current_score = compute_score(current)
    best = list(current)
    best_score = current_score
    
    max_u = max(current)
    
    for i in range(max_iters):
        if deadline and time.time() > deadline:
            break
        
        # Try adding a new element
        # Candidate new elements: up to max_u + some margin
        candidates = []
        for x in range(1, max_u + 10):
            if x not in current:
                candidates.append(x)
        
        # Also try removing an element
        if len(current) > 2:
            for x in current[1:]:  # Don't remove 0
                candidates.append(-x)  # Negative indicates removal
        
        best_candidate = None
        best_candidate_score = best_score
        
        for cand in candidates:
            if cand < 0:
                # Removal
                new_set = [x for x in current if x != -cand]
                if len(new_set) < 2:
                    continue
            else:
                # Addition
                new_set = current + [cand]
            
            s = compute_score(new_set)
            if s > best_candidate_score + 1e-10:
                best_candidate_score = s
                best_candidate = new_set
        
        if best_candidate is not None:
            current = best_candidate
            current_score = best_candidate_score
            if current_score > best_score:
                best_score = current_score
                best = list(current)
            max_u = max(current)
        else:
            # Try random perturbations
            if len(current) > 2:
                # Random swap: remove one, add one
                idx = np.random.randint(1, len(current))
                removed = current[idx]
                new_elem = np.random.randint(1, max_u + 20)
                if new_elem not in current:
                    new_set = [x for x in current if x != removed] + [new_elem]
                    s = compute_score(new_set)
                    if s > best_score:
                        best_score = s
                        best = list(new_set)
                        current = new_set
                        current_score = s
                        max_u = max(current)
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Start with a good initial set
    # Known good sets from the paper: sizes 2003 and 54265
    # For small sets, let's try various structured sets
    
    # Try geometric-like progressions
    initial_candidates = [
        [0, 1, 3],
        [0, 1, 4, 9],
        [0, 1, 3, 9, 27],
        [0, 1, 2, 4, 8, 16],
        [0, 1, 3, 6, 12, 24],
        [0, 1, 2, 4, 7, 11],
        [0, 1, 3, 5, 9, 17],
    ]
    
    best_set = [0, 1, 3]
    best_score = compute_score(best_set)
    
    for cand in initial_candidates:
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    # Local search from the best initial
    if time.time() < deadline:
        best_set, best_score = local_search(best_set, max_iters=3000, deadline=deadline)
    
    # Try a few more random restarts if time permits
    if time.time() < deadline:
        for restart in range(5):
            if time.time() > deadline - 20:
                break
            # Generate a random set with 0
            size = np.random.randint(4, 15)
            max_val = np.random.randint(10, 50)
            random_set = [0] + sorted(np.random.choice(range(1, max_val + 1), size=size-1, replace=False).tolist())
            s = compute_score(random_set)
            if s > best_score:
                best_score = s
                best_set = list(random_set)
            
            if time.time() < deadline - 10:
                improved_set, improved_score = local_search(best_set, max_iters=1000, deadline=deadline - 10)
                if improved_score > best_score:
                    best_score = improved_score
                    best_set = improved_set
    
    return best_set
# EVOLVE-BLOCK-END

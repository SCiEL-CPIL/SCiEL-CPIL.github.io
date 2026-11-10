"""Rewritten program: randomized local search with NumPy vectorized sum-counting for B2[2] sets."""
import numpy as np
import random
import time

def _sum_counts(chosen_mask):
    """Compute representation counts for all sums using convolution.
    chosen_mask: boolean array of length 1001 (index 0 unused)
    Returns: array of length 2001 where index s has count of pairs (a,b) with a<=b, a+b=s
    """
    # Use FFT-based convolution for speed
    n = len(chosen_mask)
    # Pad to next power of 2 for efficient FFT
    size = 1
    while size < 2 * n - 1:
        size *= 2
    f = np.zeros(size, dtype=np.float64)
    f[:n] = chosen_mask.astype(np.float64)
    # Convolve f with itself
    F = np.fft.rfft(f)
    G = np.fft.irfft(F * F, n=size)
    # G[s] = sum_{a+b=s} f[a]*f[b] = number of ordered pairs (a,b) with a+b=s
    # We need unordered pairs with a<=b: count = (G[s] + (1 if s even and chosen_mask[s//2] else 0)) / 2
    # Actually: G[s] counts ordered pairs. For a!=b, each unordered pair counted twice.
    # For a==b (s even, a=s/2), counted once.
    # So unordered count with a<=b = (G[s] + I[s even and chosen_mask[s/2]]) / 2
    sums = np.arange(2 * n - 1)
    even_mask = (sums % 2 == 0)
    # For even s, s//2 must be in [0, n-1]
    half = sums // 2
    valid_half = (half < n) & (half >= 0)
    diag = np.zeros(2 * n - 1, dtype=np.float64)
    diag[even_mask & valid_half] = chosen_mask[half[even_mask & valid_half]].astype(np.float64)
    counts = (G[:2 * n - 1] + diag[:2 * n - 1]) / 2.0
    return counts

def construct():
    """Find a large subset of {1,...,1000} where every sum has at most 2 representations."""
    random.seed(42)
    np.random.seed(42)
    
    N = 1000
    max_sum = 2 * N  # 2000
    time_limit = 110.0  # seconds, leave buffer
    start_time = time.time()
    
    # Initialize with a dense random subset
    # B2[2] asymptotic density is at least 3/2 * sqrt(N) / something... 
    # Actually for B2 sets, size ~ c*sqrt(N). For B2[2], larger.
    # Let's start with ~500-600 elements and try to optimize.
    
    # Try different initial densities
    best_size = 0
    best_set = None
    
    for init_size in [400, 450, 500, 550, 600]:
        if time.time() - start_time > time_limit * 0.7:
            break
            
        # Random initialization
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        chosen = set(candidates[:init_size])
        
        # Iterative local search
        for iteration in range(2000):
            if time.time() - start_time > time_limit * 0.8:
                break
            
            # Compute current violation count
            mask = np.zeros(N + 1, dtype=bool)
            for x in chosen:
                mask[x] = True
            counts = _sum_counts(mask)
            
            # Find sums with > 2 representations
            violations = np.where(counts > 2)[0]
            
            if len(violations) == 0:
                # Valid! Record and try to add more
                if len(chosen) > best_size:
                    best_size = len(chosen)
                    best_set = set(chosen)
                
                # Try to add elements greedily
                added_any = True
                while added_any and time.time() - start_time < time_limit * 0.8:
                    added_any = False
                    # Try adding each non-chosen element
                    for x in range(1, N + 1):
                        if x in chosen:
                            continue
                        # Check if adding x keeps validity
                        new_mask = mask.copy()
                        new_mask[x] = True
                        new_counts = _sum_counts(new_mask)
                        if np.max(new_counts) <= 2:
                            chosen.add(x)
                            mask = new_mask
                            added_any = True
                            if len(chosen) > best_size:
                                best_size = len(chosen)
                                best_set = set(chosen)
                            break
            
            if len(chosen) > best_size:
                best_size = len(chosen)
                best_set = set(chosen)
            
            if len(violations) == 0:
                break
            
            # Pick a random violating sum and try to fix it
            viol_sum = random.choice(violations.tolist())
            
            # Find all pairs (a,b) with a<=b, a+b=viol_sum, a,b in chosen
            pairs = []
            for a in range(1, viol_sum // 2 + 1):
                b = viol_sum - a
                if b >= a and b <= N and a in chosen and b in chosen:
                    pairs.append((a, b))
            
            if not pairs:
                continue
            
            # Try removing one element from a random pair
            a, b = random.choice(pairs)
            
            # Try swap: remove one, add another
            removed = random.choice([a, b])
            chosen.discard(removed)
            
            # Try to add a new element
            added = False
            # Prefer adding from a random candidate
            for _ in range(20):
                new_x = random.randint(1, N)
                if new_x in chosen:
                    continue
                new_mask = np.zeros(N + 1, dtype=bool)
                for x in chosen:
                    new_mask[x] = True
                new_mask[new_x] = True
                new_counts = _sum_counts(new_mask)
                if np.max(new_counts) <= 2:
                    chosen.add(new_x)
                    added = True
                    break
            
            if not added:
                # Just keep the removal
                pass
            
            if len(chosen) > best_size:
                best_size = len(chosen)
                best_set = set(chosen)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = set(chosen)
    
    # Final greedy improvement
    if best_set is None:
        best_set = set()
    
    mask = np.zeros(N + 1, dtype=bool)
    for x in best_set:
        mask[x] = True
    
    # Try adding more elements
    for x in range(1, N + 1):
        if x in best_set:
            continue
        if time.time() - start_time > time_limit * 0.9:
            break
        new_mask = mask.copy()
        new_mask[x] = True
        new_counts = _sum_counts(new_mask)
        if np.max(new_counts) <= 2:
            best_set.add(x)
            mask = new_mask
    
    return list(best_set)


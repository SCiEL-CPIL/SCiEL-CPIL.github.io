import time
import multiprocessing as mp

def _can_match_fast(m, L, n=12):
    # Compute bitmasks for each index i (1..n)
    # Bit k (0-indexed) represents value m + k + 1
    masks = [0] * (n + 1)
    for i in range(1, n + 1):
        mask = 0
        # k goes from 1 to L
        # Value is m + k
        # Condition: (m + k) % i == 0
        # k % i == (-m) % i
        rem = (-m) % i
        for k in range(1, L + 1):
            if k % i == rem:
                mask |= (1 << (k - 1))
        masks[i] = mask
        
    # Check if any mask is 0
    for i in range(1, n + 1):
        if masks[i] == 0:
            return False
            
    # Sort indices by number of choices (popcount)
    indices = list(range(1, n + 1))
    indices.sort(key=lambda x: bin(masks[x]).count('1'))
    
    # Recursive backtracking
    # used_mask: bitmask of used offsets (0..L-1)
    def dfs(idx, used_mask):
        if idx == n:
            return True
        i = indices[idx]
        candidates = masks[i] & ~used_mask
        while candidates:
            # Get lowest set bit
            bit = candidates & -candidates
            candidates -= bit
            if dfs(idx + 1, used_mask | bit):
                return True
        return False
        
    return dfs(0, 0)

def _f(m, n=12):
    """Compute f(n, m) using binary search."""
    lo, hi = 1, 144
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match_fast(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    return lo

def _worker(args):
    m_start, m_end, n = args
    best_m = m_start
    best_f = 0
    for m in range(m_start, m_end):
        f_val = _f(m, n)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    return best_m, best_f

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 12
    period = 27720  # lcm(1, 2, ..., 12)
    
    # Use multiprocessing to speed up the scan
    # Split the range into chunks
    num_procs = mp.cpu_count()
    if num_procs > 8:
        num_procs = 8
        
    chunk_size = period // num_procs
    chunks = []
    for i in range(num_procs):
        start = i * chunk_size
        end = period if i == num_procs - 1 else (i + 1) * chunk_size
        chunks.append((start, end, n))
        
    best_m = 0
    best_f = 0
    
    # Check if we have enough time to use multiprocessing
    # Setup overhead is small, but let's be safe.
    if time.time() - start_time < 10:
        with mp.Pool(processes=num_procs) as pool:
            results = pool.map(_worker, chunks)
            
        for m_val, f_val in results:
            if f_val > best_f:
                best_f = f_val
                best_m = m_val
    else:
        # Fallback to sequential if something went wrong with time
        for m in range(period):
            if time.time() > deadline:
                break
            f_val = _f(m, n)
            if f_val > best_f:
                best_f = f_val
                best_m = m
                
    return best_m

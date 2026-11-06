import numpy as np
import time

def construct():
    start_time = time.time()
    TIME_LIMIT = 80  # Safety margin before 90s kill
    
    N_MIN = 10000
    N_MAX = 1000000
    MAX_VAL = N_MAX + 1  # 1000001
    
    # Phase 1: Try smooth number enumeration for quick early answer
    # Generate y-smooth numbers and check for consecutive pairs
    # This can find a good candidate quickly
    
    def find_smooth_pair(y_limit, max_val, time_deadline):
        """Find consecutive y-smooth numbers in [N_MIN, N_MAX+1]"""
        if time.time() > time_deadline:
            return None
        
        # Get primes up to y_limit
        primes = []
        for p in range(2, y_limit + 1):
            is_prime = True
            for q in primes:
                if q * q > p:
                    break
                if p % q == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(p)
        
        # Generate all y-smooth numbers <= max_val using BFS/priority queue
        import heapq
        smooth = set()
        heap = [1]
        while heap:
            if time.time() > time_deadline:
                return None
            x = heapq.heappop(heap)
            if x > max_val:
                continue
            if x in smooth:
                continue
            smooth.add(x)
            for p in primes:
                nx = x * p
                if nx <= max_val:
                    heapq.heappush(heap, nx)
        
        # Check for consecutive pairs in [N_MIN, N_MAX]
        for n in range(N_MIN, N_MAX + 1):
            if n in smooth and (n + 1) in smooth:
                return n
        
        return None
    
    # Try increasing y limits
    y_limits = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    smooth_candidate = None
    smooth_y = None
    
    for y in y_limits:
        if time.time() > start_time + 5:  # Only spend 5s on smooth search
            break
        result = find_smooth_pair(y, MAX_VAL, start_time + 5)
        if result is not None:
            smooth_candidate = result
            smooth_y = y
            break
    
    # Phase 2: Full sieve for exact answer (with deadline guard)
    if time.time() > start_time + (TIME_LIMIT - 5):
        # Out of time, return best we have
        if smooth_candidate is not None:
            return smooth_candidate
        return N_MIN
    
    # Use numpy sieve for LPF
    size = MAX_VAL + 1  # 1000002
    lpf = np.zeros(size, dtype=np.int32)
    
    # Handle factor 2
    lpf[2::2] = 2
    
    # Sieve for odd primes
    for i in range(3, size, 2):
        if lpf[i] == 0:
            if time.time() > start_time + (TIME_LIMIT - 2):
                break
            lpf[i::i] = i
    
    # Calculate scores
    lpf_n = lpf[N_MIN:N_MAX + 1]
    lpf_np1 = lpf[N_MIN + 1:N_MAX + 2]
    
    scores = np.maximum(lpf_n, lpf_np1)
    min_idx = np.argmin(scores)
    best_n = N_MIN + min_idx
    best_score = scores[min_idx]
    
    # If smooth candidate has a better score, use it
    if smooth_candidate is not None:
        smooth_score = max(lpf[smooth_candidate], lpf[smooth_candidate + 1])
        if smooth_score < best_score:
            best_n = smooth_candidate
    
    return int(best_n)

import numpy as np
import time
import random
import math

def _sieve_primes(limit):
    """Return list of primes <= limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    primes = _sieve_primes(1000)
    n_primes = len(primes)
    
    best_residues = None
    best_score = 0
    
    # Dynamic window: start small, expand as we improve
    # Score is capped at 10^5, but we'll search incrementally
    
    def compute_score(residues, N):
        """Compute the doubly-covered prefix length up to N."""
        counts = np.zeros(N + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = int(residues[i])
            if r == 0:
                if p <= N:
                    counts[p::p] += 1
            else:
                if r <= N:
                    counts[r::p] += 1
        # Find first index >= 1 where counts < 2
        if N == 0:
            return 0
        failures = np.nonzero(counts[1:N+1] < 2)[0]
        if len(failures) == 0:
            return N
        return int(failures[0]) + 1
    
    def repair_loop(residues, N, max_iter=5000):
        """Gap-priority repair: find first gap, fix it with cheapest change."""
        counts = np.zeros(N + 1, dtype=np.int32)
        for i, p in enumerate(primes):
            r = int(residues[i])
            if r == 0:
                if p <= N:
                    counts[p::p] += 1
            else:
                if r <= N:
                    counts[r::p] += 1
        
        # Find initial L
        L = 0
        for x in range(1, N + 1):
            if counts[x] < 2:
                break
            L = x
        
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Find first gap in [1, L+50]
            target_x = -1
            search_end = min(L + 50, N)
            for x in range(1, search_end + 1):
                if counts[x] < 2:
                    target_x = x
                    break
            
            if target_x == -1:
                # No gaps found in search range, try to extend
                new_L = L
                for x in range(L + 1, N + 1):
                    if counts[x] < 2:
                        break
                    new_L = x
                L = new_L
                if L >= N:
                    break
                continue
            
            # Find cheapest prime to change to cover target_x
            best_i = -1
            best_cost = float('inf')
            best_new_r = -1
            
            for i, p in enumerate(primes):
                cur_r = int(residues[i])
                new_r = target_x % p
                if new_r == cur_r:
                    continue
                
                # Calculate cost: how many x in [1, L] with counts[x]==2 would be damaged
                cost = 0
                if cur_r == 0:
                    start_y = p
                else:
                    start_y = cur_r
                
                # Only check up to L for cost
                y = start_y
                while y <= L:
                    if counts[y] == 2:
                        cost += 1
                        if cost >= best_cost:
                            break
                    y += p
                
                if cost < best_cost:
                    best_cost = cost
                    best_i = i
                    best_new_r = new_r
                    if cost == 0:
                        break
            
            if best_i == -1:
                break
            
            # Apply change
            p = primes[best_i]
            old_r = int(residues[best_i])
            new_r = best_new_r
            
            # Update counts
            if old_r == 0:
                if p <= N:
                    counts[p::p] -= 1
            else:
                if old_r <= N:
                    counts[old_r::p] -= 1
            
            if new_r == 0:
                if p <= N:
                    counts[p::p] += 1
            else:
                if new_r <= N:
                    counts[new_r::p] += 1
            
            residues[best_i] = new_r
            
            # Update L
            new_L = 0
            for x in range(1, N + 1):
                if counts[x] < 2:
                    break
                new_L = x
            L = new_L
        
        return residues, L
    
    # Strategy: Start with multiple initializations, use repair, then expand window
    initial_strategies = [
        lambda: np.zeros(n_primes, dtype=np.int64),
        lambda: np.ones(n_primes, dtype=np.int64),
        lambda: np.array([(i * 13 + 5) % p for i, p in enumerate(primes)], dtype=np.int64),
        lambda: np.array([(i * 7 + 3) % p for i, p in enumerate(primes)], dtype=np.int64),
        lambda: np.array([random.randint(0, p - 1) for p in primes], dtype=np.int64),
        lambda: np.array([(p // 2) % p for p in primes], dtype=np.int64),
    ]
    
    # Dynamic window sizes to try
    window_sizes = [2000, 5000, 10000, 20000, 50000, 100000]
    
    rng = np.random.default_rng(42)
    random.seed(42)
    
    for strat_idx, strat in enumerate(initial_strategies):
        if time.time() > deadline - 10.0:
            break
        
        residues = strat()
        
        # Try increasing window sizes
        for N in window_sizes:
            if time.time() > deadline - 10.0:
                break
            
            # Run repair loop
            residues, L = repair_loop(residues.copy(), N, max_iter=3000)
            
            if L > best_score:
                best_score = L
                best_residues = residues.copy()
            
            # If we've reached a good score, try to extend
            if L >= N - 10:
                # Try larger window
                continue
            
            # Occasional random perturbation to escape local minima
            if L > 0 and time.time() < deadline - 15.0:
                # Perturb a few random residues
                for _ in range(5):
                    i = int(rng.integers(n_primes))
                    p = primes[i]
                    residues[i] = int(rng.integers(0, p))
                
                # Quick repair after perturbation
                residues, L2 = repair_loop(residues.copy(), N, max_iter=1000)
                if L2 > best_score:
                    best_score = L2
                    best_residues = residues.copy()
        
        # If we have time, do more random restarts with repair
        while time.time() < deadline - 10.0:
            residues = np.array([random.randint(0, p - 1) for p in primes], dtype=np.int64)
            N = min(50000, max(5000, best_score + 5000))
            residues, L = repair_loop(residues, N, max_iter=2000)
            
            if L > best_score:
                best_score = L
                best_residues = residues.copy()
            
            if best_score >= 10000:
                break
    
    # Final verification and potential extension
    if best_residues is not None and time.time() < deadline - 5.0:
        # Try to extend beyond current best
        for N in [best_score + 1000, best_score + 5000, min(100000, best_score + 20000)]:
            if time.time() > deadline - 3.0:
                break
            if N > 100000:
                N = 100000
            final_score = compute_score(best_residues, N)
            if final_score > best_score:
                best_score = final_score
    
    # Ensure we have a valid result
    if best_residues is None:
        best_residues = np.array([1 % p for p in primes], dtype=np.int64)
    
    return best_residues.tolist()

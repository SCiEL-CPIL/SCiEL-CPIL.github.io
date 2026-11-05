import time
import random
import math

N = 5000
START_TIME = time.time()
TIME_LIMIT = 300.0

def _primes(limit):
    sieve = bytearray(b'\x01') * (limit + 1)
    if limit >= 0:
        sieve[0] = 0
    if limit >= 1:
        sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]

PRIMES = _primes(N)

def construct():
    global START_TIME
    start_time = START_TIME
    if start_time == 0:
        start_time = time.time()
        START_TIME = start_time
    
    best_size = 0
    best_set = set()
    
    # Precompute residues: residues[x][i] = x % PRIMES[i] for i where PRIMES[i] <= N
    # But that's N * len(PRIMES) = 5000 * 669 = 3.3M integers, ~26MB. Acceptable.
    # Actually, let's just compute on the fly with a cache.
    
    # Faster admissibility check using bitmask
    # For a set S and prime p, we need to know if all residues 0..p-1 appear.
    # Represent covered residues as a bitmask (Python int, arbitrary precision)
    
    def is_admissible(present_set, size):
        """Check admissibility of present_set (a set of ints) with given size."""
        for p in PRIMES:
            if p > size:
                break
            mask = 0
            for x in present_set:
                mask |= (1 << (x % p))
                if mask == (1 << p) - 1:
                    return False
        return True
    
    # Greedy construction: start empty, add elements that maintain admissibility
    # But we want to MAXIMIZE size, so start full and remove.
    
    # Better: use the insight that for a set of size k, we need to miss at least one
    # residue for each prime p <= k. The "cost" of a prime p is that we must exclude
    # at least one residue class.
    
    # Optimal strategy: For primes p <= k, choose residue classes to exclude that
    # overlap as much as possible (so we exclude fewer total elements).
    
    # Let's try: for a target size k, find the minimum elements to exclude from [0,N)
    # such that for each p <= k, one residue class mod p is empty.
    
    # This is equivalent to: choose a set E of elements to exclude, minimizing |E|,
    # such that for each p <= N-|E|, there exists r with no element in [0,N)\E congruent to r mod p.
    
    # Greedy: iteratively pick the element whose exclusion helps the most primes.
    
    # Alternative simpler approach: 
    # 1. For each prime p <= N, the residue classes have sizes ~N/p.
    # 2. To make residue r mod p empty, we remove ~N/p elements.
    # 3. If we choose residue classes for all primes, the total removed is at most
    #    sum of N/p for primes p <= k, but with overlaps.
    
    # Let's just do a fast greedy with randomization:
    
    num_restarts = 500
    for restart in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        # Start with full set
        present = bytearray(N)
        for i in range(N):
            present[i] = 1
        current_size = N
        
        # Track for each prime p, which residues are covered
        # covered[p] = set of residues that appear in current set
        # For speed, use a list of bytearrays
        covered = {}
        for p in PRIMES:
            if p > N:
                break
            cov = bytearray(p)
            for r in range(p):
                cov[r] = 1  # Initially all covered
            covered[p] = cov
        
        # Greedy removal
        for it in range(5000):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            
            # Find violating primes
            violating = []
            for p in PRIMES:
                if p > current_size:
                    break
                if min(covered[p]) > 0:
                    violating.append(p)
            
            if not violating:
                break
            
            # Pick prime with smallest min residue count to remove
            # We need to know counts, not just coverage
            # Let's track counts instead
            pass
        
        # This approach is getting complicated. Let me just optimize the original.
        
        # Optimized original approach:
        current_set = set(range(N))
        
        for it in range(10000):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            
            k = len(current_set)
            if k == 0:
                break
            
            violating_p = None
            for p in PRIMES:
                if p > k:
                    break
                seen = 0
                for x in current_set:
                    seen |= (1 << (x % p))
                    if seen == (1 << p) - 1:
                        violating_p = p
                        break
                if violating_p:
                    break
            
            if violating_p is None:
                break
            
            p = violating_p
            # Group by residue
            classes = [[] for _ in range(p)]
            for x in current_set:
                classes[x % p].append(x)
            
            # Find smallest class
            min_idx = min(range(p), key=lambda i: len(classes[i]))
            
            # Remove that class
            to_remove = set(classes[min_idx])
            current_set -= to_remove
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
        
        # Add back phase
        candidates = [x for x in range(N) if x not in best_set]
        random.shuffle(candidates)
        for x in candidates[:1000]:
            if time.time() - start_time > TIME_LIMIT - 3:
                break
            new_set = best_set | {x}
            new_k = len(new_set)
            ok = True
            for p in PRIMES:
                if p > new_k:
                    break
                seen = 0
                for val in new_set:
                    seen |= (1 << (val % p))
                    if seen == (1 << p) - 1:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                best_set.add(x)
                best_size += 1
        
        if len(best_set) > best_size:
            best_size = len(best_set)
    
    return list(best_set)

construct()

import time
import random
import math

N = 5000
START_TIME = time.time()
TIME_LIMIT = 300.0  # 300s deadline, keep margin

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

def _is_admissible(points, primes_list):
    """
    Check if the set of points is admissible.
    Admissible: for every prime p <= len(points), some residue class mod p is missed.
    """
    k = len(points)
    if k == 0:
        return True
    
    for p in primes_list:
        if p > k:
            break
        
        seen = bytearray(p)
        count = 0
        for v in points:
            r = v % p
            if not seen[r]:
                seen[r] = 1
                count += 1
                if count == p:
                    return False
    return True

def construct():
    global START_TIME
    start_time = START_TIME
    if start_time == 0:
        start_time = time.time()
        START_TIME = start_time
    
    # Precompute primes up to N (5000)
    primes = _primes(N)
    
    best_set = set()
    best_size = 0
    
    # Precompute for each prime p, the list of numbers in each residue class
    residue_classes = {}
    for p in primes:
        if p > N:
            break
        classes = [[] for _ in range(p)]
        for x in range(N):
            classes[x % p].append(x)
        residue_classes[p] = classes
    
    # For each number x, precompute which primes have x in which residue
    # This is expensive, so let's just use the residue_classes
    
    # Greedy with randomization:
    # Start with all numbers, then for each prime p (in order), if all residues are covered,
    # remove the residue class with the fewest elements
    
    # To speed up, maintain a boolean array for which numbers are in the current set
    # and counts for each prime/residue
    
    num_restarts = 2000
    
    for restart in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        # Initialize
        in_set = bytearray(b'\x01') * N  # All numbers start in set
        
        # Count for each prime p, residue r: how many numbers in set have x % p == r
        counts = {}
        for p in primes:
            if p > N:
                break
            counts[p] = [0] * p
            for x in range(N):
                counts[p][x % p] += 1
        
        # Process primes in increasing order
        for p in primes:
            if time.time() - start_time > TIME_LIMIT - 2:
                break
            if p > sum(in_set):
                break
            
            # Check if all residues are covered
            if all(counts[p][r] > 0 for r in range(p)):
                # Find residue with minimum count
                min_count = min(counts[p])
                min_residues = [r for r in range(p) if counts[p][r] == min_count]
                r = random.choice(min_residues)
                
                # Remove all numbers with residue r mod p
                for x in residue_classes[p][r]:
                    if in_set[x]:
                        in_set[x] = 0
                        # Update counts for all primes
                        for q in primes:
                            if q > N:
                                break
                            counts[q][x % q] -= 1
        
        # Build the set
        current_set = set(x for x in range(N) if in_set[x])
        
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)
    
    return list(best_set)

# Initial call to set start time and warm up
construct()

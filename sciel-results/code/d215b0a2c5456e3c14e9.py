import time
import random
import math

N = 5000
START_TIME = time.time()
TIME_LIMIT = 280.0  # 300s deadline, keep margin

# Precompute primes up to N
def _sieve_primes(limit):
    """Sieve of Eratosthenes to find primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = 0
    sieve[1] = 0
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = 0
    return [p for p in range(2, limit + 1) if sieve[p]]

ALL_PRIMES = _sieve_primes(N)

def construct():
    global START_TIME
    start_time = START_TIME
    if start_time == 0:
        start_time = time.time()
        START_TIME = start_time
    
    best_set = set()
    best_size = 0
    
    # Primes we need to check: p <= k where k is the size of the tuple
    # Since k <= N, we need primes up to N
    
    # Strategy:
    # 1. Start with the full set [0, N]
    # 2. For each prime p <= current_size, if all residues mod p are covered,
    #    remove the residue class with the fewest elements
    # 3. Then try to add back elements that don't violate admissibility
    # 4. Repeat with different random seeds / orderings
    
    # Efficient admissibility check using residue counts
    def is_admissible(points, primes_to_check):
        k = len(points)
        for p in primes_to_check:
            if p > k:
                break
            # Check if all p residues are covered
            residues = set(v % p for v in points)
            if len(residues) == p:
                return False
        return True
    
    def find_violation(points, primes_to_check):
        """Return the first prime p where all residues are covered, or None."""
        k = len(points)
        for p in primes_to_check:
            if p > k:
                break
            residues = set(v % p for v in points)
            if len(residues) == p:
                return p
        return None
    
    def greedy_repair(points_set, primes_to_check, random_order=False):
        """
        Greedily remove residue classes to make the set admissible.
        Returns the repaired set.
        """
        current = set(points_set)
        
        max_iter = 100
        for _ in range(max_iter):
            if time.time() - start_time > TIME_LIMIT - 20:
                break
            
            k = len(current)
            if k == 0:
                break
            
            # Find a violating prime
            violating_p = None
            for p in primes_to_check:
                if p > k:
                    break
                residues = set(v % p for v in current)
                if len(residues) == p:
                    violating_p = p
                    break
            
            if violating_p is None:
                break
            
            # Group by residue class mod violating_p
            classes = {}
            for v in current:
                r = v % violating_p
                if r not in classes:
                    classes[r] = []
                classes[r].append(v)
            
            # Remove the class with the fewest elements
            # For diversity, sometimes pick from the top 2 smallest
            sorted_classes = sorted(classes.items(), key=lambda x: len(x[1]))
            if random.random() < 0.6:
                min_class = sorted_classes[0][0]
            else:
                min_class = sorted_classes[1][0] if len(sorted_classes) > 1 else sorted_classes[0][0]
            
            current -= set(classes[min_class])
        
        return current
    
    def local_search_add(current_set, primes_to_check):
        """
        Try to add back elements one by one if admissibility is preserved.
        Returns the improved set.
        """
        current = set(current_set)
        
        # Get all candidates not in current
        candidates = [x for x in range(N) if x not in current]
        random.shuffle(candidates)
        
        added_any = True
        max_passes = 10
        for _ in range(max_passes):
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            
            if not added_any:
                break
            
            added_any = False
            check_count = 0
            for x in candidates:
                if time.time() - start_time > TIME_LIMIT - 5:
                    break
                
                check_count += 1
                if check_count > 3000:
                    break
                
                # Check if current | {x} is admissible
                new_size = len(current) + 1
                valid = True
                for p in primes_to_check:
                    if p > new_size:
                        break
                    rx = x % p
                    # Check if all residues mod p would be covered
                    residues = set(v % p for v in current)
                    if rx not in residues:
                        residues.add(rx)
                    if len(residues) == p:
                        valid = False
                        break
                
                if valid:
                    current.add(x)
                    candidates.remove(x)
                    added_any = True
        
        return current
    
    num_restarts = 80
    
    for restart in range(num_restarts):
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        # Start with full set
        current = set(range(N))
        
        # Greedy repair
        current = greedy_repair(current, ALL_PRIMES)
        
        if len(current) > best_size:
            best_set = set(current)
            best_size = len(current)
        
        # Local search: add back elements
        improved = local_search_add(current, ALL_PRIMES)
        
        if len(improved) > best_size:
            best_set = set(improved)
            best_size = len(improved)
        
        # Try a second local search pass with different randomization
        if len(improved) > 0 and time.time() - start_time < TIME_LIMIT - 15:
            # Shuffle and try again
            improved2 = local_search_add(improved, ALL_PRIMES)
            if len(improved2) > best_size:
                best_set = set(improved2)
                best_size = len(improved2)
    
    return list(best_set)

# Initial call to set start time and warm up
construct()

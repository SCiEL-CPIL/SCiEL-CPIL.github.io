"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def _is_valid_full(els):
    """Full validation of the set."""
    n = len(els)
    sums = set()
    for r in range(1, n + 1):
        for combo in __import__('itertools').combinations(range(n), r):
            s = sum(els[i] for i in combo)
            if s in sums:
                return False
            sums.add(s)
    sum_list = sorted(sums)
    m = len(sum_list)
    for i in range(m):
        for j in range(i + 1, m):
            a, b = sum_list[i], sum_list[j]
            if b % a == 0:
                return False
    return True

def _can_add_fast(x, current_sums):
    """
    Fast check:
    1. New sums (x, s+x) are not in current_sums.
    2. No a in current_sums divides b in new_sums.
    3. No b in new_sums divides a in current_sums (handled by 2 if we check both ways? No, 2 checks a|b. We need b|a too? 
       If b < a, b|a is possible. 
       But b = s+x. If x is large, b is large. 
       Let's check both directions to be safe, or just a|b and rely on b>a usually.
       Actually, if b < a, then a % b == 0 is the check.
    """
    new_sums = [x]
    for s in current_sums:
        ns = s + x
        if ns in current_sums:
            return False
        # Check divisibility with existing
        # a | ns?
        for a in current_sums:
            if ns % a == 0:
                return False
            # a % ns == 0? (ns divides a)
            if a % ns == 0:
                return False
        new_sums.append(ns)
        
    # Check x against existing
    for a in current_sums:
        if x % a == 0:
            return False
        if a % x == 0:
            return False
            
    # Check x against new_sums (s+x)
    # x divides s+x iff x divides s.
    for s in current_sums:
        if s % x == 0:
            return False
        # s+x divides x? Impossible if s>0.
        
    # We skip checking (s1+x) vs (s2+x) for speed.
    return True

def construct():
    start_time = time.time()
    best_set = [31, 32, 34]
    
    # Try to find larger sets
    for target_size in range(11, 5, -1):
        if time.time() - start_time > 55:
            break
        
        found = False
        for _ in range(2000): # More restarts since check is faster
            if time.time() - start_time > 55:
                break
                
            # Candidate pool
            # Using numbers in [100, 256] reduces small sum issues
            pool = list(range(100, 257))
            random.shuffle(pool)
            
            current_sums = set()
            current_set = []
            
            for x in pool:
                if len(current_set) >= target_size:
                    break
                if _can_add_fast(x, current_sums):
                    current_set.append(x)
                    # Update sums
                    new_sums = set()
                    for s in current_sums:
                        new_sums.add(s + x)
                    new_sums.add(x)
                    current_sums.update(new_sums)
            
            if len(current_set) == target_size:
                # Verify with full check
                if _is_valid_full(current_set):
                    return [int(x) for x in current_set]
    
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END

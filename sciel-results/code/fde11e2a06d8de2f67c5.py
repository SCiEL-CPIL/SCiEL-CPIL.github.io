import time
import random

def _score_set_bitset(S):
    """
    Count distinct d > 0 such that a, a+d, a+2d all in S.
    Uses bitset operations for speed.
    S is a set of integers in 1..512.
    """
    if not S:
        return 0
    
    # Create a bitmask where bit i is set if i is in S
    # We shift indices by 1 so 1 is bit 0, 512 is bit 511
    mask = 0
    for x in S:
        mask |= (1 << (x - 1))
    
    distinct_d = 0
    # Max d: a + 2d <= 512 => 2d <= 511 => d <= 255
    for d in range(1, 256):
        # Check if there exists a such that a, a+d, a+2d are in S
        # This is equivalent to: (mask & (mask >> d) & (mask >> 2*d)) != 0
        # Note: Python integers are arbitrary precision, so shifts are safe.
        if (mask & (mask >> d) & (mask >> 2 * d)):
            distinct_d += 1
            
    return distinct_d

def _greedy_construct(n=32, max_val=512):
    """
    Greedily construct a set of n elements maximizing distinct AP differences.
    """
    current_set = set()
    # Pre-compute potential differences? No, just evaluate gain.
    # To speed up, we can keep the current distinct_d set.
    current_d_set = set()
    
    # Initial candidate: all numbers 1..512
    candidates = list(range(1, max_val + 1))
    
    for _ in range(n):
        best_candidate = None
        best_gain = -1
        best_new_d_set = None
        
        for c in candidates:
            # Tentatively add c
            new_set = current_set | {c}
            # Calculate new distinct d set
            # Optimization: We can compute the set of d's for the new set.
            # Since n is small (<=32), we can just use the bitset scorer logic 
            # but return the set of d's to compare gains.
            
            # Let's create a helper to get the set of d's
            mask = 0
            for x in new_set:
                mask |= (1 << (x - 1))
            
            new_d_set = set()
            for d in range(1, 256):
                if (mask & (mask >> d) & (mask >> 2 * d)):
                    new_d_set.add(d)
            
            gain = len(new_d_set - current_d_set)
            
            if gain > best_gain:
                best_gain = gain
                best_candidate = c
                best_new_d_set = new_d_set
        
        current_set.add(best_candidate)
        current_d_set = best_new_d_set
        candidates.remove(best_candidate)
        
    return current_set

def construct():
    # 1. Greedy construction for a strong starting point
    current = _greedy_construct(32, 512)
    current_score = _score_set_bitset(current)
    
    best = set(current)
    best_score = current_score
    
    start_time = time.time()
    time_limit = 110  # seconds
    
    # Simulated Annealing
    temp = 5.0
    cooling = 0.999
    
    random.seed(42)
    
    # Pre-compute list of all numbers for random selection
    all_nums = list(range(1, 513))
    
    while time.time() - start_time < time_limit:
        # Pick a random element to remove
        to_remove = random.choice(list(current))
        
        # Pick a random element to add
        # To avoid generating the list of non-current elements every time (which is O(N)),
        # we can just pick a random number and check if it's in current.
        # But 32/512 is small, so rejection sampling is fast.
        to_add = random.choice(all_nums)
        if to_add in current:
            continue
            
        new_set = set(current)
        new_set.discard(to_remove)
        new_set.add(to_add)
        
        new_score = _score_set_bitset(new_set)
        delta = new_score - current_score
        
        if delta >= 0 or random.random() < pow(2.718281828, delta / temp):
            current = new_set
            current_score = new_score
            if current_score > best_score:
                best = set(current)
                best_score = current_score
        
        temp *= cooling
        if temp < 0.01:
            temp = 0.01
            
    return sorted(best)

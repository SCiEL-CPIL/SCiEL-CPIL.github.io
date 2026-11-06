import random
import itertools
import time

def is_sidon(subset):
    """Check if a tuple of integers is a Sidon set (including diagonals)."""
    sums = set()
    n = len(subset)
    for i in range(n):
        si = subset[i]
        for j in range(i, n):
            s = si + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def max_sidon_size(S):
    """Find the size of the largest Sidon subset of S."""
    n = len(S)
    # We check from largest to smallest
    for k in range(n, 0, -1):
        # Optimization: if k is large, few combinations. If small, many.
        # But we return as soon as we find one.
        for subset in itertools.combinations(S, k):
            if is_sidon(subset):
                return k
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = list(range(12))
    best_score = max_sidon_size(best_set)
    
    # If we already found 3, we can't do better than 1 or 2 (which is impossible for 12 distinct ints? 
    # Actually 1 is always Sidon. 2 is always Sidon? 
    # {a,b} sums: a+a, a+b, b+b. Distinct if a!=b. So 2 is always Sidon.
    # So min possible max sidon size is 2.
    # But for 12 elements, is it possible to have no Sidon subset of size 3?
    # {a,b,c} Sidon if a+a!=b+b!=c+c!=a+b!=a+c!=b+c.
    # a+a=b+b implies a=b. So distinct ints always have distinct diagonals.
    # We just need a+b != a+c (b!=c), a+b != b+c (a!=c), etc.
    # The only collision in a 3-set is a+b = c+c or a+a = b+c or a+b = c+b (a=c).
    # So {a,b,c} is non-Sidon if a+b=2c or a+c=2b or b+c=2a.
    # i.e. if they form an arithmetic progression.
    # So if we avoid all 3-term APs, every 3-subset is Sidon.
    # If we want to MINIMIZE max Sidon size, we WANT 3-term APs.
    # So we want a set with many 3-term APs.
    # If every 3-subset is an AP, then no 3-subset is Sidon.
    # But can every 3-subset be an AP? No.
    # We just need to ensure that for every 4-subset, at least one 3-subset is non-Sidon?
    # No, a 4-set is non-Sidon if ANY pair sum collides.
    # It's easier to just search.
    
    if best_score == 2:
        return best_set

    # Random search
    # Bias towards structured sets?
    # Random sets are likely to have large Sidon subsets (size ~ sqrt(12) * constant).
    # We want to FIND the bad ones.
    
    # Let's try local search starting from range(12)
    current = best_set[:]
    current_score = best_score
    
    for step in range(20000):
        if time.time() > deadline:
            break
        
        # Pick a random index
        idx = random.randint(0, 11)
        old_val = current[idx]
        
        # Try to replace with a value that creates more collisions
        # Heuristic: pick a value close to existing ones to increase AP density?
        # Or just random.
        new_val = random.randint(0, 128)
        if new_val in current:
            continue
            
        current[idx] = new_val
        new_score = max_sidon_size(current)
        
        if new_score < current_score:
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_set = current[:]
                # If we hit 2, we are done
                if best_score == 2:
                    break
        else:
            # Revert
            current[idx] = old_val
            
    return best_set

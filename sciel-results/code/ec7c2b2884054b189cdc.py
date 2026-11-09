import time
import sys
import random

START_TIME = time.time()
DEADLINE = 55.0

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def can_add(x, S, n):
    """
    Check if x can be added to a set represented by S (list of bitmasks for cardinalities 0..n).
    Returns (True, new_S) if valid, (False, None) otherwise.
    """
    new_S = [0] * (n + 2)
    new_S[0] = S[0]
    
    for k in range(1, n + 2):
        old_S_k = S[k] if k <= n else 0
        old_S_k_minus_1 = S[k-1] if k-1 <= n else 0
        
        new_part = old_S_k_minus_1 << x
        new_S[k] = old_S_k | new_part
        
        # Check collision with all previous new_S[j] for j < k
        for j in range(k):
            if new_S[k] & new_S[j]:
                return False, None
                
    return True, new_S

def build_S(A):
    """Build the bitmask representation for a given set A."""
    n = len(A)
    S = [0] * (n + 1)
    S[0] = 1
    for x in A:
        valid, new_S = can_add(x, S, n)
        if not valid:
            return None # Should not happen if A is valid
        S = new_S
        n += 1
    return S

def is_valid(A):
    """Check if a set A is valid."""
    if not A:
        return True
    if len(set(A)) != len(A):
        return False
    for x in A:
        if not isinstance(x, int) or x < 1 or x > 256:
            return False
    S = build_S(A)
    return S is not None

def greedy_construct(order, max_size=60):
    """
    Greedily add elements in the given order.
    """
    S = [0] * (max_size + 1)
    S[0] = 1
    n = 0
    chosen = []
    
    for x in order:
        if n >= max_size:
            break
        if is_done():
            break
            
        valid, new_S = can_add(x, S, n)
        if valid:
            S = new_S
            n += 1
            chosen.append(x)
            
    return chosen

def local_search(init_set, time_limit=5.0):
    """
    Try to improve the set by swapping elements or adding new ones.
    """
    current_set = list(init_set)
    best_set = current_set[:]
    best_size = len(current_set)
    start_time = time.time()
    
    S = build_S(current_set)
    if S is None:
        return best_set

    while time.time() - start_time < time_limit:
        if is_done():
            break
            
        # Try to add a random element not in set
        candidates = [i for i in range(1, 257) if i not in current_set]
        if candidates:
            y = random.choice(candidates)
            valid, new_S = can_add(y, S, len(current_set))
            if valid:
                current_set.append(y)
                S = new_S
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
                continue
        
        # Try swap: remove z, add y
        if not candidates:
            break
            
        y = random.choice(candidates)
        z = random.choice(current_set)
        if z == y:
            continue
            
        # New set: remove z, add y
        new_set = [x for x in current_set if x != z]
        new_set.append(y)
        
        # Check validity of new_set
        if is_valid(new_set):
            current_set = new_set
            S = build_S(current_set)
            # Size is same, but maybe this configuration allows more later?
            # We don't track "potential", so this is a random walk on valid sets of same size.
            pass
            
    return best_set

def construct():
    best_set = []
    
    # 1. Greedy Ascending
    if not is_done():
        s1 = greedy_construct(range(1, 257))
        if len(s1) > len(best_set):
            best_set = s1
            
    # 2. Greedy Descending
    if not is_done():
        s2 = greedy_construct(range(256, 0, -1))
        if len(s2) > len(best_set):
            best_set = s2
            
    # 3. Greedy Randomized (multiple restarts)
    random.seed(42)
    for _ in range(100):
        if is_done():
            break
        order = list(range(1, 257))
        random.shuffle(order)
        s3 = greedy_construct(order)
        if len(s3) > len(best_set):
            best_set = s3
            # If we found a larger set, try to improve it
            if len(best_set) > 9 and not is_done():
                improved = local_search(best_set, time_limit=2.0)
                if len(improved) > len(best_set):
                    best_set = improved

    # 4. Try to extend best_set with local search
    if best_set and len(best_set) < 60 and not is_done():
        improved = local_search(best_set, time_limit=DEADLINE - elapsed() - 5)
        if len(improved) > len(best_set):
            best_set = improved

    # Final validation
    if is_valid(best_set):
        return best_set
    else:
        # Fallback to powers of 2
        return [1, 2, 4, 8, 16, 32, 64, 128]

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}: {res}")

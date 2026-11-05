import time
import random
import sys

def _is_sidon(s):
    """Check if set s is Sidon (all pair sums a+b with a<=b are unique)."""
    if not s:
        return True
    sums = set()
    for i, a in enumerate(s):
        for b in s[i:]:
            s_val = a + b
            if s_val in sums:
                return False
            sums.add(s_val)
    return True

def _diffs(s):
    """Return set of nonzero absolute differences."""
    if len(s) < 2:
        return set()
    d = set()
    lst = sorted(s)
    for i in range(len(lst)):
        for j in range(i+1, len(lst)):
            d.add(lst[j] - lst[i])
    return d

def _validate(A, B):
    """Validate both sets are Sidon and have disjoint nonzero differences."""
    if not A or not B:
        return False
    if not _is_sidon(A):
        return False
    if not _is_sidon(B):
        return False
    dA = _diffs(A)
    dB = _diffs(B)
    if dA & dB:
        return False
    return True

def _can_add_to_set(S, x, other_diffs):
    """Check if x can be added to set S while maintaining Sidon and disjoint diffs with other_diffs."""
    if x in S:
        return False
    # Check Sidon condition: all pair sums a+x for a in S must be new, and x+x must be new
    seen_sums = set()
    for a in S:
        s_val = a + x
        if s_val in seen_sums:
            return False
        seen_sums.add(s_val)
    if x + x in seen_sums:
        return False
    
    # Check difference condition: all |x-a| for a in S must not be in other_diffs
    # Also, since Sidon <=> distinct differences, we don't need to check for duplicate diffs within S
    for a in S:
        d = abs(x - a)
        if d in other_diffs:
            return False
    
    return True

def _get_diffs(S):
    """Return set of nonzero absolute differences for set S."""
    if len(S) < 2:
        return set()
    d = set()
    lst = sorted(S)
    for i in range(len(lst)):
        for j in range(i+1, len(lst)):
            d.add(lst[j] - lst[i])
    return d

def _greedy_construct(candidates, rng, n=256, max_size=64):
    """Greedy construction: iterate through candidates in given order, add to A or B if possible."""
    A = []
    B = []
    diffsA = set()
    diffsB = set()
    
    for x in candidates:
        # Try adding to A first (prioritize larger sets)
        if len(A) < max_size and _can_add_to_set(A, x, diffsB):
            for a in A:
                diffsA.add(abs(x - a))
            A.append(x)
        elif len(B) < max_size and _can_add_to_set(B, x, diffsA):
            for b in B:
                diffsB.add(abs(x - b))
            B.append(x)
    
    return A, B

def _local_search(A, B, rng, n=256, max_size=64, max_iterations=10000):
    """Local search: try to add elements or swap elements to improve the solution."""
    A = list(A)
    B = list(B)
    dA = _get_diffs(A)
    dB = _get_diffs(B)
    
    for _ in range(max_iterations):
        improved = False
        
        # Try adding a random element to A or B
        if rng.random() < 0.5 and len(A) < max_size:
            for _ in range(20):
                x = rng.randint(1, n)
                if x not in A and _can_add_to_set(A, x, dB):
                    for a in A:
                        dA.add(abs(x - a))
                    A.append(x)
                    improved = True
                    break
        elif len(B) < max_size:
            for _ in range(20):
                x = rng.randint(1, n)
                if x not in B and _can_add_to_set(B, x, dA):
                    for b in B:
                        dB.add(abs(x - b))
                    B.append(x)
                    improved = True
                    break
        
        # Try swapping: remove an element from A and add it to B, or vice versa
        if not improved and rng.random() < 0.3:
            # Try moving an element from A to B
            if A and len(B) < max_size:
                idx = rng.randrange(len(A))
                x = A[idx]
                # Check if removing x from A and adding to B is valid
                A_temp = A[:idx] + A[idx+1:]
                dA_temp = _get_diffs(A_temp)
                if _can_add_to_set(B, x, dA_temp):
                    # Also need to check that new dA_temp doesn't conflict with dB
                    if not (dA_temp & dB):
                        # Update
                        A = A_temp
                        dA = dA_temp
                        for b in B:
                            dB.add(abs(x - b))
                        B.append(x)
                        improved = True
            # Try moving an element from B to A
            elif B and len(A) < max_size:
                idx = rng.randrange(len(B))
                x = B[idx]
                B_temp = B[:idx] + B[idx+1:]
                dB_temp = _get_diffs(B_temp)
                if _can_add_to_set(A, x, dB_temp):
                    if not (dB_temp & dA):
                        B = B_temp
                        dB = dB_temp
                        for a in A:
                            dA.add(abs(x - a))
                        A.append(x)
                        improved = True
        
        # Try replacing: remove one element from A and add a different one
        if not improved and rng.random() < 0.2 and len(A) > 1:
            idx = rng.randrange(len(A))
            x_remove = A[idx]
            A_temp = A[:idx] + A[idx+1:]
            dA_temp = _get_diffs(A_temp)
            for _ in range(10):
                x_add = rng.randint(1, n)
                if x_add not in A_temp and x_add != x_remove:
                    if _can_add_to_set(A_temp, x_add, dB):
                        if not (dA_temp & dB):
                            A = A_temp
                            dA = dA_temp
                            for a in A_temp:
                                dA.add(abs(x_add - a))
                            A.append(x_add)
                            improved = True
                            break
        
        # Try replacing: remove one element from B and add a different one
        if not improved and rng.random() < 0.2 and len(B) > 1:
            idx = rng.randrange(len(B))
            x_remove = B[idx]
            B_temp = B[:idx] + B[idx+1:]
            dB_temp = _get_diffs(B_temp)
            for _ in range(10):
                x_add = rng.randint(1, n)
                if x_add not in B_temp and x_add != x_remove:
                    if _can_add_to_set(B_temp, x_add, dA):
                        if not (dB_temp & dA):
                            B = B_temp
                            dB = dB_temp
                            for b in B_temp:
                                dB.add(abs(x_add - b))
                            B.append(x_add)
                            improved = True
                            break
        
        if not improved:
            # No improvement in this iteration; break early if stuck
            if rng.random() < 0.1:
                break
    
    return A, B

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin
    
    best_score = 0
    best_A = []
    best_B = []
    
    n = 256
    max_size = 64
    
    # We will try multiple random restarts with different candidate orderings
    for seed in range(2000):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        
        # Randomly shuffle the candidate order
        candidates = list(range(1, n + 1))
        rng.shuffle(candidates)
        
        # Greedy phase
        A, B = _greedy_construct(candidates, rng, n, max_size)
        
        score = len(A) + len(B)
        if _validate(A, B) and score > best_score:
            best_score = score
            best_A = list(A)
            best_B = list(B)
        
        # Local Search
        if time.time() < deadline - 5:
            A_ls, B_ls = _local_search(A, B, rng, n, max_size)
            
            if _validate(A_ls, B_ls):
                score_ls = len(A_ls) + len(B_ls)
                if score_ls > best_score:
                    best_score = score_ls
                    best_A = list(A_ls)
                    best_B = list(B_ls)
                    
    # Fallback
    if not best_A or not best_B:
        best_A = [1]
        best_B = [2]
        
    return [best_A, best_B]

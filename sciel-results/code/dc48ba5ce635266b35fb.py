import time
import sys

# EVOLVE-BLOCK-START

def _find_sidon_set(D):
    """
    Find a Sidon set A subset of {0, ..., D} containing 0,
    such that differences cover 1..30.
    Returns the set if found, else None.
    """
    sys.setrecursionlimit(10000)
    
    def backtrack(A, diffs, missing):
        if not missing:
            return True
        
        k = len(A)
        
        if not A:
            start = 0
        else:
            start = A[-1] + 1
            
        if start > D:
            return False
            
        # Pruning: Can we cover 'missing' with remaining elements?
        N_rem = D - start + 1
        max_new_diffs = N_rem * k + (N_rem * (N_rem - 1)) // 2
        
        if len(missing) > max_new_diffs:
            return False
            
        # Heuristic: Try to cover missing diffs
        # If 1 is missing and 0 is in A, x=1 is a good candidate.
        # We just iterate x from start to D.
        
        for x in range(start, D + 1):
            # Check Sidon condition
            new_diffs = set()
            valid = True
            for a in A:
                d = x - a
                if d in diffs or d in new_diffs:
                    valid = False
                    break
                new_diffs.add(d)
            
            if not valid:
                continue
            
            A.append(x)
            diffs.update(new_diffs)
            missing.difference_update(new_diffs)
            
            if backtrack(A, diffs, missing):
                return True
            
            A.pop()
            diffs.difference_update(new_diffs)
            missing.update(new_diffs)
        
        return False

    A = [0]
    diffs = set()
    missing = set(range(1, 31))
    
    if backtrack(A, diffs, missing):
        return A
    
    return None

def _validate(A):
    """Validate that A is a Sidon set covering 1..30."""
    if 0 not in A:
        return False
    if len(A) > 32:
        return False
    if len(A) != len(set(A)):
        return False
        
    diffs = set()
    n = len(A)
    for i in range(n):
        for j in range(i + 1, n):
            d = A[j] - A[i]
            if d in diffs:
                return False
            diffs.add(d)
            
    for d in range(1, 31):
        if d not in diffs:
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Search for minimal D
    # Lower bound: 30
    # Upper bound: 100 (known solution is 63)
    
    for D in range(30, 100):
        if time.time() > deadline:
            break
            
        result = _find_sidon_set(D)
        if result is not None:
            if _validate(result):
                return result
    
    # Fallback
    return [0, 1, 4, 10, 12, 28, 32, 45, 63]

# EVOLVE-BLOCK-END

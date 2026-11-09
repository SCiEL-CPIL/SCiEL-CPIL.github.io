import time
import random
import sys

START_TIME = time.time()
DEADLINE = 55.0  # Keep a safety margin from the 60s limit

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def validate(A):
    if not A:
        return True
    if len(set(A)) != len(A):
        return False
    for x in A:
        if not isinstance(x, int) or x < 1 or x > 256:
            return False
    n = len(A)
    S = [set() for _ in range(n + 1)]
    S[0].add(0)
    for i in range(n):
        x = A[i]
        for j in range(i, -1, -1):
            if not S[j]:
                continue
            for s in S[j]:
                S[j+1].add(s + x)
    for j in range(n + 1):
        for k in range(j + 1, n + 1):
            if S[j] & S[k]:
                return False
    return True

def try_find_k(K, time_limit=5.0):
    """
    Attempts to find a set of size K using randomized greedy search.
    """
    start_time = time.time()
    candidates = list(range(1, 257))
    
    for _ in range(10000): # Max iterations
        if time.time() - start_time > time_limit:
            return None
        if is_done():
            return None
            
        random.shuffle(candidates)
        chosen = []
        S = [set() for _ in range(K + 1)]
        S[0].add(0)
        
        success = True
        for c in candidates:
            if len(chosen) == K:
                break
                
            # Check if c can be added
            valid = True
            new_sums = []
            current_k = len(chosen)
            
            # We are adding to cardinality current_k + 1
            # Sources are S[current_k]
            for s in S[current_k]:
                ns = s + c
                new_sums.append(ns)
                # Check collision with S[0..current_k]
                for j in range(current_k + 1):
                    if ns in S[j]:
                        valid = False
                        break
                if not valid:
                    break
                    
            if valid:
                chosen.append(c)
                for ns in new_sums:
                    S[current_k + 1].add(ns)
            else:
                # If we can't add c, we might skip it or fail?
                # Greedy: just skip c and try next candidate?
                # But we need exactly K elements.
                # If we skip too many, we might not reach K.
                # For a simple greedy, if we can't add, we just move to next candidate.
                # But this might not fill K slots.
                pass
        
        if len(chosen) == K:
            # Verify
            if validate(chosen):
                return chosen
                
    return None

def construct():
    # Base solution: Powers of 2, size 9
    base_solution = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    # Try to find size 10
    if not is_done():
        res = try_find_k(10, time_limit=10.0)
        if res is not None:
            return res
            
    # Try to find size 11? Unlikely.
    
    return base_solution

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}: {res}")

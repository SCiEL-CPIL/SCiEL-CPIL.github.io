import time
import sys

START_TIME = time.time()
DEADLINE = 55.0

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

def backtrack_solve(K, max_time=45.0):
    """
    Backtracking search with forward checking to find a set of size K.
    S[j] = set of all subset sums of cardinality j from chosen elements.
    """
    start_time = time.time()
    # S[0] = {0} always
    S = [set() for _ in range(K + 1)]
    S[0].add(0)
    chosen = []
    
    # Precompute all possible sums for each cardinality given the current set
    # We'll use a recursive backtracking approach
    
    def can_add(x, current_k):
        """Check if x can be added when we have current_k elements."""
        # New sums at cardinality current_k+1: {s + x for s in S[current_k]}
        new_sums = set()
        for s in S[current_k]:
            ns = s + x
            if ns in new_sums:
                # Internal collision within new sums - shouldn't happen since x>0 and s are distinct
                # Actually this can't happen: if s1 != s2, then s1+x != s2+x
                pass
            new_sums.add(ns)
        
        # Check collision with all existing S[0..current_k]
        for j in range(current_k + 1):
            if new_sums & S[j]:
                return False, None
        
        # Also need to check: after adding x, the new S[j+1] for j < current_k 
        # will change, but those are handled when we actually add x.
        # For forward checking, we just need to verify no immediate collision.
        return True, new_sums
    
    def solve(pos, current_k):
        """Try to build a set of size K. current_k elements chosen so far."""
        if time.time() - start_time > max_time:
            return None
        if is_done():
            return None
        
        if current_k == K:
            return chosen[:]
        
        # Try each candidate from current position to end
        # Heuristic: try candidates in order of "safety" - fewer potential collisions
        # For simplicity, try in increasing order, but we could sort by heuristic
        
        for i in range(pos, 257):
            x = i
            if x in chosen:
                continue
            
            valid, new_sums = can_add(x, current_k)
            if not valid:
                continue
            
            # Add x
            chosen.append(x)
            old_S = [set(s) for s in S]
            
            # Update S: new S[j+1] = old S[j+1] union {s + x for s in old S[j]}
            # But we only care about up to cardinality K
            # Actually, we need to update all S[j] for j from current_k down to 0
            # S[current_k+1] gets new_sums added
            # S[j] for j <= current_k remain the same (they don't involve x yet in terms of being "complete")
            # Wait - no. S[j] is the set of ALL subset sums of cardinality j from the chosen elements.
            # When we add x, for each j from 0 to current_k:
            #   new S[j+1] = old S[j+1] union {s + x : s in old S[j]}
            #   S[j] stays the same for j <= current_k
            # But S[current_k+1] is new
            
            # Let's just update properly
            for j in range(current_k + 1):
                if S[j]:
                    for s in list(S[j]):  # iterate over copy
                        ns = s + x
                        S[j+1].add(ns)
            
            result = solve(i + 1, current_k + 1)
            if result is not None:
                return result
            
            # Backtrack
            chosen.pop()
            for j in range(K + 1):
                S[j] = old_S[j]
        
        return None
    
    result = solve(0, 0)
    if result is not None:
        if validate(result):
            return result
    return None

def construct():
    # Base solution: Powers of 2, size 9
    base_solution = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    
    # Try to find larger sets using backtracking
    # Start from size 10 and go up
    for K in range(10, 13):
        if is_done():
            break
        time_left = DEADLINE - elapsed()
        if time_left < 5:
            break
        res = backtrack_solve(K, max_time=min(time_left - 2, 20.0))
        if res is not None:
            return res
    
    return base_solution

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}: {res}")

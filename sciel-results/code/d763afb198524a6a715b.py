"""Rewritten program for the reciprocal-subsum-free set task using backtracking and sub-sum set pruning."""
import time
from fractions import Fraction

_START = None
DEADLINE = 85.0  # Keep a 5s safety margin

def _deadline_hit():
    global _START
    if _START is None:
        return False
    return (time.time() - _START) > DEADLINE

def construct():
    global _START
    _START = time.time()
    
    N = 24
    candidates = list(range(1, N + 1))
    
    # We will use a backtracking search.
    # State: current set S (as a list), current sub-sum set Sigma (as a set of Fractions)
    # We want to maximize len(S).
    
    best_set = []
    
    # Pre-compute reciprocals
    recips = {i: Fraction(1, i) for i in candidates}
    
    # Order candidates: 
    # Hypothesis: Larger numbers (smaller reciprocals) are "cheaper" to include in terms of creating conflicts?
    # Or maybe smaller numbers are more critical?
    # Let's try descending order (24..1) first, as it might allow packing more small-value items.
    # Actually, let's try a few orders or a smart heuristic.
    # For now, let's use descending order.
    order = sorted(candidates, reverse=True)
    
    # Greedy initial solution to set a good lower bound
    def greedy_solve(order):
        S = []
        Sigma = {Fraction(0, 1)}
        for y in order:
            ry = recips[y]
            valid = True
            # Check 1: ry in Sigma
            if ry in Sigma:
                valid = False
            else:
                # Check 2: for all x in S, if rx > ry, (rx - ry) not in Sigma
                for x in S:
                    rx = recips[x]
                    if rx > ry:
                        if (rx - ry) in Sigma:
                            valid = False
                            break
            if valid:
                S.append(y)
                # Update Sigma
                new_sums = set()
                for s in Sigma:
                    new_sums.add(s + ry)
                Sigma = Sigma.union(new_sums)
        return S

    # Try greedy with different orders to find a good initial bound
    init_orders = [
        list(range(N, 0, -1)),
        list(range(1, N + 1)),
        list(range(N // 2, 0, -1)) + list(range(N, N // 2 - 1, -1)), # Mixed
    ]
    
    best_size = 0
    for o in init_orders:
        S_g = greedy_solve(o)
        if len(S_g) > best_size:
            best_size = len(S_g)
            best_set = S_g
            
    # Backtracking search
    # We'll use a recursive function.
    # To avoid deep recursion issues or overhead, we can implement it iteratively or with a limit.
    # N=24 is small enough for recursion depth.
    
    def backtrack(idx, current_S, Sigma):
        nonlocal best_set, best_size
        
        if _deadline_hit():
            return
            
        # Prune: if we can't beat best_size, stop
        if len(current_S) + (N - idx) <= best_size:
            return
            
        if idx == N:
            if len(current_S) > best_size:
                best_size = len(current_S)
                best_set = current_S[:]
            return
        
        y = order[idx]
        ry = recips[y]
        
        # Option 1: Include y
        # Check validity
        valid = True
        if ry in Sigma:
            valid = False
        else:
            for x in current_S:
                rx = recips[x]
                if rx > ry:
                    if (rx - ry) in Sigma:
                        valid = False
                        break
        
        if valid:
            # Add y
            new_S = current_S + [y]
            # Update Sigma
            # Optimization: Only add new sums if they are distinct?
            # Sigma is a set, so duplicates are handled.
            new_sums = set()
            for s in Sigma:
                new_sums.add(s + ry)
            new_Sigma = Sigma.union(new_sums)
            
            backtrack(idx + 1, new_S, new_Sigma)
            
        # Option 2: Exclude y
        backtrack(idx + 1, current_S, Sigma)

    # Start backtracking
    # To improve pruning, we can sort the order dynamically or use a different static order.
    # Let's stick with the descending order for the main search, as it often works well for packing.
    # We can also try to re-order based on "criticality" but that's complex.
    
    # One issue: The greedy solution might have found a set of size 19.
    # Backtracking will explore the space.
    # To speed up, we can use the greedy solution to guide the search?
    # Or just run the backtracking.
    
    # Let's add a timeout check in the loop
    backtrack(0, [], {Fraction(0, 1)})
    
    if not best_set:
        best_set = [24]
        
    return best_set

if __name__ == "__main__":
    # For local testing
    res = construct()
    print(f"Best set: {res}, Size: {len(res)}")

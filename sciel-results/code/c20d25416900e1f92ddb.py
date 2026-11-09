import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    n = 128
    best_set = set(range(n//2 + 1, n + 1)) # Initial seed solution
    best_size = len(best_set)
    
    # Greedy construction function
    def greedy_randomized():
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        current_set = set()
        
        for c in candidates:
            # Check if adding c creates a violation
            # Violation exists if there are a, b in current_set such that:
            # 1. a + b in current_set
            # 2. a + c in current_set
            # 3. b + c in current_set
            
            # To be efficient, we can check pairs (a, b) in current_set where a+b is in current_set.
            # If such a pair exists, we must ensure a+c and b+c are NOT in current_set.
            # Wait, the condition is: "Forbid THREE DISTINCT members a,b,c for which a+b, a+c and b+c are also selected."
            # This implies a, b, c are in the set, AND a+b, a+c, b+c are in the set.
            
            # Let S be the current set. We want to add c.
            # A violation is formed if there exist a, b in S such that:
            # a != b, a != c, b != c (distinctness)
            # a+b in S, a+c in S, b+c in S
            
            # Note: a+c and b+c must be in S. Since c is being added, a+c and b+c must already be in S.
            # So, for all pairs (a, b) in S such that a+b in S:
            #   If a+c in S and b+c in S, then adding c creates a violation with (a, b, c).
            
            # Optimization:
            # Precompute list of "active pairs" (a, b) in S where a+b in S.
            # For each such pair, if (a+c in S) and (b+c in S), reject c.
            
            can_add = True
            # Iterate through pairs in current_set
            # To speed this up, we can maintain a list of pairs (a, b) such that a+b in S.
            # However, updating this list on insertion is O(|S|).
            # Given |S| <= 128, O(|S|^2) check per insertion is 128^2 * 128 = 2M ops.
            # With 100 restarts, 200M ops. Python might be slow.
            # Let's try to optimize the check.
            
            # Alternative check:
            # For a fixed c, a violation exists if there are a, b in S \ {c} such that:
            # a+b in S, a+c in S, b+c in S.
            # Let X = {x in S | x+c in S}. Note x != c.
            # If a, b in X, and a+b in S, and a != b, then violation.
            # So, we just need to check if there is a pair (a, b) in X such that a+b in S.
            # This is equivalent to checking if the set X contains a "sum-free" pair with respect to S?
            # No, it's checking if X is sum-free within S?
            # Specifically: Does X contain two distinct elements a, b such that a+b in S?
            
            # Let's compute X.
            # X = {x in S | x != c and x+c in S}
            
            # If |X| is small, this is fast.
            # If |X| is large, we check pairs in X.
            
            # Let's implement this check.
            
            X = []
            for x in current_set:
                if x != c and (x + c) in current_set:
                    X.append(x)
            
            # Now check if any pair in X sums to an element in S.
            # Also need a != b.
            # And a, b, c distinct. Since x != c is enforced, and a,b from X, they are distinct from c.
            # Do we need a != b? Yes, "THREE DISTINCT members".
            
            # Sort X? Not necessary for set lookup.
            # Check all pairs in X.
            # To avoid O(|X|^2), we can do:
            # For each a in X:
            #   For each b in X with b > a:
            #     if a+b in S: violation.
            
            # If |X| is up to 128, 128^2/2 = 8k checks.
            # 128 candidates * 8k checks = 1M checks per greedy run.
            # 100 runs = 100M checks.
            # Each check is a set lookup (O(1)).
            # Python 100M simple ops might take ~10-20s.
            # This fits in 50s.
            
            for i in range(len(X)):
                a = X[i]
                for j in range(i + 1, len(X)):
                    b = X[j]
                    if (a + b) in current_set:
                        can_add = False
                        break
                if not can_add:
                    break
            
            if can_add:
                current_set.add(c)
        
        return current_set

    # Run multiple random restarts
    num_restarts = 500
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        s = greedy_randomized()
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            # If we reach a very high number, we might stop early?
            # Max possible is 128.
            if best_size == 128:
                break

    # Convert to list of ints
    return [int(x) for x in best_set]

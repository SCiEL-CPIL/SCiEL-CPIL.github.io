"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    # Key insight: For distinct x, y in X (subset of 1..63):
    # - If x+y >= 64, then x+y in S, and there exists a in S with x+a, y+a in S. Violation.
    # - If x+y < 64 and x+y in X, same violation.
    # So X must be sum-free (distinct pair sums not in X) AND max pair sum < 64.
    
    # This means: for all distinct x,y in X, x+y < 64 and x+y not in X.
    # Equivalently, X is a subset of {1,...,63} where no two distinct elements sum to >= 64,
    # and no two distinct elements sum to an element of X.
    
    # The constraint x+y < 64 for all distinct x,y in X means:
    # If we sort X as x_1 < x_2 < ... < x_k, then x_{k-1} + x_k < 64.
    # This bounds the largest elements.
    
    # Strategy: Use bitmask-based greedy with multiple random orderings.
    # Universe: {1, ..., 63}
    # For each candidate x, check if adding it to current X violates constraints.
    
    # Precompute for each x in 1..63, the set of y such that x+y >= 64 (conflict via S)
    # and x+y in {1..63} (potential sum-free conflict if x+y also in X)
    
    # Conflict condition for pair (x,y), x != y:
    # 1. x + y >= 64: always conflict (since x+y in S)
    # 2. x + y < 64 and x + y in X: conflict (sum-free violation)
    
    # So when adding x to X, for each y in X:
    # - If x + y >= 64: conflict
    # - If x + y < 64 and x + y in X: conflict
    
    # Also need to check triples within X (Case 1), but let's verify:
    # For distinct x,y,z in X, violation if x+y, x+z, y+z all in S∪X.
    # If x+y < 64, x+y in S∪X iff x+y in X.
    # If x+y >= 64, x+y in S.
    # But if any pair sum >= 64, that pair already conflicts.
    # So if X satisfies the pair constraints, all pair sums are < 64 and not in X.
    # Then for triples: x+y < 64 and not in X, so x+y not in S∪X. No triple violation.
    # So pair constraints are sufficient!
    
    def can_add(x, X_set, X_bitmask):
        """Check if x can be added to X."""
        for y in X_set:
            s = x + y
            if s >= 64:
                return False
            if s in X_set:
                return False
        return True
    
    best_X = set()
    
    # Try multiple random orderings and greedy
    num_trials = 50
    for trial in range(num_trials):
        if time.time() > deadline - 5:
            break
        
        candidates = list(range(1, 64))
        random.shuffle(candidates)
        
        current_X = set()
        for x in candidates:
            if can_add(x, current_X, 0):
                current_X.add(x)
        
        if len(current_X) > len(best_X):
            best_X = current_X.copy()
            if time.time() > deadline - 10:
                break
    
    # Also try sorted orders (increasing, decreasing)
    for order in [list(range(1, 64)), list(range(63, 0, -1))]:
        if time.time() > deadline - 5:
            break
        current_X = set()
        for x in order:
            if can_add(x, current_X, 0):
                current_X.add(x)
        if len(current_X) > len(best_X):
            best_X = current_X.copy()
    
    # Local search: try to improve by swapping
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        best_copy = best_X.copy()
        for x in list(best_copy):
            # Try removing x and adding some y not in best_X
            best_copy.remove(x)
            for y in range(1, 64):
                if y in best_copy:
                    continue
                if can_add(y, best_copy, 0):
                    new_X = best_copy | {y}
                    if len(new_X) > len(best_X):
                        best_X = new_X
                        improved = True
                        break
                    best_copy.add(y)
                else:
                    best_copy.add(y)
            best_copy.discard(y)
            best_copy.add(x)
    
    # Combine with S = {64, ..., 128}
    S = set(range(64, 129))
    result = list(S | best_X)
    result.sort()
    
    # Final validation
    def is_valid(result_set):
        rs = set(result_set)
        for i in range(len(result_set)):
            for j in range(i+1, len(result_set)):
                for k in range(j+1, len(result_set)):
                    a, b, c = result_set[i], result_set[j], result_set[k]
                    if (a+b in rs) and (a+c in rs) and (b+c in rs):
                        return False
        return True
    
    if not is_valid(result):
        # Fallback: just S
        result = list(S)
        result.sort()
    
    return result

# EVOLVE-BLOCK-END

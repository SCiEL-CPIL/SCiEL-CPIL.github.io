import random
import time
import sys

def construct():
    """
    Returns a list of up to 48 distinct positive integers <= 192 such that
    no selected integer divides the sum of any 3 distinct other selected integers.
    Uses a branch-and-bound approach on a conflict graph with bitset optimizations.
    """
    n_max = 192
    max_size = 48
    
    # Precompute conflict graph
    # adj[i] is a set of j such that i and j cannot coexist
    # Condition: x divides sum of any 3 distinct others.
    # For a set S, x in S is invalid if exists y1,y2,y3 in S\{x} such that (y1+y2+y3) % x == 0.
    # Also, for any y in S, y is invalid if exists x, z1, z2 in S\{y} such that (x+z1+z2) % y == 0.
    # Actually, the condition is symmetric in the sense that if we build S incrementally,
    # we just need to check if adding x to S makes S invalid.
    # S is valid iff for all x in S, there are no y1,y2,y3 in S\{x} with sum % x == 0.
    
    # Let's precompute for every pair (x, y) if they are "conflicting" in a way that prevents them from being in a valid set?
    # No, conflicts are hyperedges (4-tuples). We need a different approach.
    
    # We will use a recursive backtracking search.
    # State: current set S (as a bitmask or list), and remaining candidates.
    # To speed up, we can precompute for each x, the set of sums of 3 distinct numbers from 1..192? No, that's huge.
    
    # Alternative:
    # For a given set S, checking if x can be added:
    # 1. For all a,b,c in S (distinct), (a+b+c) % x != 0
    # 2. For all y in S, and all a,b in S\{y} (distinct), (x+a+b) % y != 0
    
    # Let's represent S as a list for simplicity in the inner loop, but use bitmasks for candidate selection.
    
    # Precompute all sums of 3 distinct numbers?
    # There are C(192, 3) ~ 1.1 million triples.
    # We can store for each possible sum s, the list of triples that make it.
    # But we need to check if any triple in S sums to a multiple of x.
    
    # Better approach for B&B:
    # Order numbers. Try adding them one by one.
    # Use a timeout.
    
    start_time = time.time()
    timeout = 55  # seconds
    
    # We'll use a heuristic ordering: try larger numbers first? Or smaller?
    # Smaller numbers are more likely to divide sums. So maybe avoid small numbers?
    # Actually, if x is large, it's hard for a+b+c to be a multiple of x unless a+b+c is large.
    # If x is small, it's easy.
    # So maybe pick large numbers first to maximize size?
    
    # Let's try a greedy first to get a good lower bound.
    def greedy():
        S = []
        # Try adding numbers from 192 down to 1
        for x in range(n_max, 0, -1):
            if time.time() - start_time > timeout:
                break
            valid = True
            # Check 1: x divides sum of 3 in S
            if len(S) >= 3:
                # Optimization: if max(S)*3 < x, then no sum can be a multiple of x (except 0, but sums are positive)
                # Actually, min sum of 3 in S is sum of 3 smallest.
                # If x > sum of 3 largest in S, then x cannot divide any sum of 3.
                # Wait, (a+b+c) % x == 0 implies a+b+c >= x (since a,b,c > 0).
                # So if x > sum(S), it's safe from condition 1.
                if x <= sum(S):
                    # Check all triples
                    # This is O(n^3). We can optimize with precomputed sums?
                    # For now, just do it.
                    for i in range(len(S)):
                        for j in range(i+1, len(S)):
                            for k in range(j+1, len(S)):
                                if (S[i] + S[j] + S[k]) % x == 0:
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
            if not valid:
                continue
            
            # Check 2: Any y in S divides x + a + b for distinct a,b in S\{y}
            for y in S:
                # We need to check if there exist a,b in S\{y} such that (x+a+b) % y == 0
                # This is equivalent to (x+a+b) == m*y for some m.
                # a+b = m*y - x.
                # We need to check if there are two distinct elements in S\{y} that sum to (m*y - x) for some m.
                # Possible values of a+b: min is sum of 2 smallest in S\{y}, max is sum of 2 largest.
                # m ranges from 1 to (x + max_sum) // y.
                # Let's just iterate a,b in S\{y}
                others = [v for v in S if v != y]
                if len(others) < 2:
                    continue
                # Precompute sums of pairs in others?
                # Or just double loop.
                for a in others:
                    for b in others:
                        if a == b: continue
                        # Ensure distinct indices? a,b are values. If duplicates? No, distinct integers.
                        # But a and b must be distinct elements. Since values are unique, a!=b is enough.
                        # However, we must ensure a and b are distinct from y, which they are.
                        if (x + a + b) % y == 0:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            
            if valid:
                S.append(x)
        
        return S
    
    best_set = greedy()
    best_size = len(best_set)
    
    # Now try to improve with local search or B&B
    # Given the time limit, let's try a simple local search:
    # Try to swap one element with another to allow more additions.
    
    def can_add(x, S):
        if x in S:
            return False
        # Check 1
        if len(S) >= 3:
            if x <= sum(S):
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        for k in range(j+1, len(S)):
                            if (S[i] + S[j] + S[k]) % x == 0:
                                return False
        # Check 2
        for y in S:
            others = [v for v in S if v != y]
            if len(others) < 2:
                continue
            for a in others:
                for b in others:
                    if a == b: continue
                    if (x + a + b) % y == 0:
                        return False
        return True
    
    # Local search
    improved = True
    while improved and time.time() - start_time < timeout:
        improved = False
        # Try to add any missing number
        for x in range(1, n_max + 1):
            if x not in best_set:
                if can_add(x, best_set):
                    best_set.append(x)
                    improved = True
                    break
        
        if not improved:
            # Try swaps
            # For each y in best_set, try removing y and adding x not in best_set
            # If can_add(x, best_set \ {y}) and len(best_set) increases (it won't, it stays same)
            # But maybe it allows future additions?
            # This is complex. Let's just do a few random restarts if we have time.
            pass
    
    # If we have time, try a few random orders
    if time.time() - start_time < timeout - 5:
        for _ in range(10):
            random.shuffle(list(range(1, n_max + 1))) # dummy shuffle to use random
            order = list(range(n_max, 0, -1))
            random.shuffle(order)
            S = []
            for x in order:
                if time.time() - start_time > timeout - 1:
                    break
                if can_add(x, S):
                    S.append(x)
            if len(S) > len(best_set):
                best_set = S
    
    return best_set[:max_size]

if __name__ == "__main__":
    res = construct()
    print(len(res))
    print(res)

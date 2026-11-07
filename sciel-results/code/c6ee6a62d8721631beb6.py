import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 55.0
    n = 128
    best_size = 0
    best_set = set(range(65, 129))  # seed: upper half
    
    def is_valid_addition(x, S):
        """Check if x can be added to S without creating a forbidden triple."""
        # For each pair (a,b) in S, check if x would create a triple with a,b where
        # all three pair sums are in S ∪ {x}
        # Triple (a, b, x) is forbidden if a+b, a+x, b+x are all in S ∪ {x}
        # Since a+b must be in S (x is new), and a+x, b+x must be in S ∪ {x}
        # But a+x > x and b+x > x (since a,b >= 1), so a+x and b+x must be in S
        # So: for any a,b in S with a != b, if a+b in S and a+x in S and b+x in S, then forbidden
        # Also need to check triples where x is one of the pair-sum values... 
        # Actually, let me think more carefully.
        
        # Forbidden: exist distinct a,b,c in S∪{x} such that a+b, a+c, b+c all in S∪{x}
        # Case 1: x is one of a,b,c. WLOG c=x. Then need a+b, a+x, b+x all in S∪{x}.
        #   a+b must be in S (since a+b != x generally, but could be... no, a+b > x if a,b > 0 and we're adding x)
        #   Actually a+b could equal x if a+b=x. But a,b < x since they're in S and x is being added.
        #   So a+b < 2x. If a+b == x, then a+b is x which is being added. Hmm, this gets complex.
        #   
        # Let me just check all triples involving x directly.
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i+1, len(S_list)):
                a = S_list[i]
                b = S_list[j]
                # Check triple (a, b, x)
                s_ab = a + b
                s_ax = a + x
                s_bx = b + x
                # All three must be in S ∪ {x}
                if s_ab in S or s_ab == x:
                    if s_ax in S or s_ax == x:
                        if s_bx in S or s_bx == x:
                            # Check they're all distinct from a,b,x? No, the constraint is just that they're in the set
                            # a,b,x are distinct (x not in S, a != b)
                            return False
        return True
    
    def try_greedy(order):
        S = set()
        for x in order:
            if is_valid_addition(x, S):
                S.add(x)
        return S
    
    full_range = list(range(1, n + 1))
    
    # Strategy 1: descending greedy
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
        if attempt % 50 == 0:
            S = try_greedy(range(n, 0, -1))
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
        elif attempt % 50 == 1:
            S = try_greedy(range(1, n + 1))
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
        else:
            perm = full_range[:]
            random.shuffle(perm)
            S = try_greedy(perm)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S.copy()
    
    # Local improvement: try swapping or adding/removing
    for _ in range(500):
        if time.time() > deadline - 3:
            break
        S = set(best_set)
        improved = False
        # Try to add missing elements
        for x in range(1, n + 1):
            if x not in S and is_valid_addition(x, S):
                S.add(x)
                improved = True
        # Try removing one element to allow adding two others
        if not improved and len(S) > best_size - 3:
            removed = None
            added_count = 0
            for y in list(S):
                S_copy = set(S)
                S_copy.discard(y)
                added = 0
                for x in range(1, n + 1):
                    if x not in S_copy and is_valid_addition(x, S_copy):
                        S_copy.add(x)
                        added += 1
                if added > 1:
                    S = S_copy
                    removed = y
                    improved = True
                    break
        if improved and len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    # Final: ensure we have at least the seed result
    seed_set = set(range(65, 129))
    if len(seed_set) > best_size:
        best_set = seed_set
        best_size = len(seed_set)
    
    return list(best_set)

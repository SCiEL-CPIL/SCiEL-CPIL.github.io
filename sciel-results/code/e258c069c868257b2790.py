import time
import random
import itertools

# Task: Find a nonempty subset of [1, 180] with no THREE DISTINCT elements a,b,c
# satisfying 1/a = 1/b + 1/c. Maximize cardinality.
# 1/a = 1/b + 1/c  <=>  bc = a(b+c)

def construct():
    deadline = time.time() + 55  # Safety margin for 60s limit
    
    N = 180
    universe = list(range(1, N + 1))
    
    # Precompute for each element x, the set of pairs (u,v) that would form a conflict
    # conflict_pairs[x] = set of frozenset({u,v}) such that {x,u,v} satisfies bc=a(b+c)
    # with x,u,v distinct
    conflict_pairs = {x: set() for x in range(1, N + 1)}
    
    # For each triple (a,b,c) with a<b<c, check if bc = a(b+c)
    # Actually, we need to check all permutations of roles
    # For distinct x,u,v, check:
    #   uv = x(u+v)  [x is 'a', u,v are 'b','c']
    #   xv = u(x+v)  [u is 'a', x,v are 'b','c']
    #   xu = v(x+u)  [v is 'a', x,u are 'b','c']
    
    # Precompute: for each pair (u,v) with u<v, which x values conflict?
    # If uv = x(u+v), then x = uv/(u+v). Must be integer.
    # If xv = u(x+v) => xv = ux + uv => xv - ux = uv => x(v-u) = uv => x = uv/(v-u)
    # If xu = v(x+u) => xu = vx + vu => xu - vx = vu => x(u-v) = vu => x = vu/(u-v) = -vu/(v-u)
    #   Since u<v, v-u>0, so x = vu/(u-v) is negative. So this case never gives positive x.
    #   Wait: xu = v(x+u) => xu = vx + vu => xu - vx = vu => x(u-v) = vu
    #   If u < v, then u-v < 0, so x = vu/(u-v) < 0. Not in our range.
    #   If u > v, then u-v > 0, so x = vu/(u-v) > 0. But we assumed u<v for the pair.
    #   So for pair (u,v) with u<v, the third role assignment gives no valid x.
    
    # So for pair (u,v) with u<v:
    #   Role 1: x = uv/(u+v) if integer
    #   Role 2: x = uv/(v-u) if integer
    #   Role 3: no valid positive x (as shown above)
    
    # Let's verify Role 3 more carefully:
    # v is 'a', x and u are 'b','c'. Equation: xu = v(x+u)
    # xu = vx + vu
    # xu - vx = vu
    # x(u - v) = vu
    # x = vu / (u - v)
    # If u < v: u - v < 0, so x < 0. Invalid.
    # If u > v: u - v > 0, so x = vu/(u-v). But we're iterating u<v, so this doesn't apply.
    # So yes, Role 3 gives no valid x for u<v.
    
    # Therefore, for each pair (u,v) with u<v, at most 2 x values conflict.
    
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            # Role 1: x = uv/(u+v)
            num1 = u * v
            den1 = u + v
            if num1 % den1 == 0:
                x1 = num1 // den1
                if 1 <= x1 <= N and x1 != u and x1 != v:
                    conflict_pairs[x1].add(frozenset((u, v)))
            
            # Role 2: x = uv/(v-u)
            num2 = u * v
            den2 = v - u
            if den2 > 0 and num2 % den2 == 0:
                x2 = num2 // den2
                if 1 <= x2 <= N and x2 != u and x2 != v:
                    conflict_pairs[x2].add(frozenset((u, v)))
    
    # Now is_conflict is O(1) per pair check
    def is_conflict_with_addition(S, x):
        """Check if adding x to S creates a conflict."""
        for pair in conflict_pairs[x]:
            u, v = tuple(pair)
            if u in S and v in S:
                return True
        return False
    
    # Count conflicts for each element (for ordering heuristic)
    conflict_count = {x: len(conflict_pairs[x]) for x in range(1, N + 1)}
    
    best_set = set()
    best_size = 0
    
    def greedy_construct(order):
        S = set()
        for x in order:
            if not is_conflict_with_addition(S, x):
                S.add(x)
        return S
    
    def local_search(S):
        """Try to improve S by swapping elements."""
        S = set(S)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try adding each element not in S
            for x in range(1, N + 1):
                if x in S:
                    continue
                if is_conflict_with_addition(S, x):
                    continue
                # Try removing one element to make room? No, just add if possible.
                # Actually, if no conflict, just add it.
                S.add(x)
                improved = True
                break  # Restart the loop to try more additions
            if not improved:
                # Try swaps: remove y, add x
                for x in range(1, N + 1):
                    if x in S:
                        continue
                    if not is_conflict_with_addition(S, x):
                        # Can add directly, handled above
                        continue
                    # Need to remove something. Try removing each y in S.
                    for y in list(S):
                        S.remove(y)
                        if not is_conflict_with_addition(S, x):
                            S.add(x)
                            improved = True
                            break
                        S.add(y)  # Restore
                    if improved:
                        break
        return S
    
    # Initial orderings
    orderings = []
    # Descending (large numbers first)
    orderings.append(sorted(universe, reverse=True))
    # Ascending
    orderings.append(sorted(universe))
    # By conflict count (fewest conflicts first)
    orderings.append(sorted(universe, key=lambda x: conflict_count[x]))
    # By conflict count (most conflicts first)
    orderings.append(sorted(universe, key=lambda x: -conflict_count[x]))
    # Random
    for _ in range(3):
        orderings.append(random.sample(universe, N))
    
    for order in orderings:
        S = greedy_construct(order)
        S = local_search(S)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
        
        if time.time() > deadline - 10:
            break
    
    # Randomized restarts with local search
    while time.time() < deadline - 5:
        order = random.sample(universe, N)
        S = greedy_construct(order)
        S = local_search(S)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    # Final polish: try to add any remaining elements
    for x in range(1, N + 1):
        if x not in best_set and not is_conflict_with_addition(best_set, x):
            best_set.add(x)
    
    return list(best_set)

import time
import random
import itertools

def construct():
    deadline = time.time() + 55  # Safety margin for 60s limit
    
    N = 180
    universe = list(range(1, N + 1))
    
    # Precompute for each x, the list of pairs (u,v) such that {x,u,v} is a forbidden triple.
    # For each x, a pair (u,v) conflicts with x if any of:
    #   u*v == x*(u+v)   [x plays role a]
    #   x*v == u*(x+v)   [u plays role a]
    #   x*u == v*(x+u)   [v plays role a]
    # We store for each x a set of frozenset({u,v}) pairs that conflict with x.
    
    conflict_pairs = [set() for _ in range(N + 1)]  # 1-indexed
    
    for a in range(1, N + 1):
        for b in range(1, N + 1):
            if a == b:
                continue
            for c in range(1, N + 1):
                if a == c or b == c:
                    continue
                # Check if (a,b,c) satisfies bc = a(b+c)
                if b * c == a * (b + c):
                    # Triple {a,b,c} is forbidden
                    conflict_pairs[a].add(frozenset((b, c)))
                    conflict_pairs[b].add(frozenset((a, c)))
                    conflict_pairs[c].add(frozenset((a, b)))
    
    # Also build reverse: for each pair {u,v}, which x values conflict?
    # This helps for local search.
    pair_to_x = {}
    for x in range(1, N + 1):
        for pair in conflict_pairs[x]:
            if pair not in pair_to_x:
                pair_to_x[pair] = set()
            pair_to_x[pair].add(x)
    
    def can_add(S, x):
        """Check if x can be added to S without creating a forbidden triple."""
        for pair in conflict_pairs[x]:
            u, v = tuple(pair)
            if u in S and v in S:
                return False
        return True
    
    def greedy_build(order):
        S = set()
        for x in order:
            if can_add(S, x):
                S.add(x)
        return S
    
    def local_search(S, max_iters=200):
        """Try to improve S by removing one element and adding one or more others."""
        S = set(S)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            # Try removing each element and see if we can add more
            for remove_x in list(S):
                S_trial = S - {remove_x}
                # Try to add back elements not in S_trial
                added = 0
                for add_x in range(1, N + 1):
                    if add_x not in S_trial and add_x != remove_x:
                        if can_add(S_trial, add_x):
                            S_trial.add(add_x)
                            added += 1
                if len(S_trial) > len(S):
                    S = S_trial
                    improved = True
                    break
        return S
    
    best_set = set()
    best_size = 0
    
    # Phase 1: Multiple greedy builds with different orderings
    orderings = [
        sorted(universe, reverse=True),
        sorted(universe),
        sorted(universe, key=lambda x: x % 7),
        sorted(universe, key=lambda x: x % 13),
        sorted(universe, key=lambda x: -x // 10),
    ]
    
    for order in orderings:
        if time.time() > deadline - 10:
            break
        S = greedy_build(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    # Phase 2: Randomized restarts
    while time.time() < deadline - 15:
        order = random.sample(universe, N)
        S = greedy_build(order)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
    
    # Phase 3: Local search on best set and some random sets
    if time.time() < deadline - 5:
        S_improved = local_search(best_set, max_iters=500)
        if len(S_improved) > best_size:
            best_size = len(S_improved)
            best_set = S_improved
    
    # More local search attempts on random greedy results
    while time.time() < deadline - 2:
        order = random.sample(universe, N)
        S = greedy_build(order)
        S_improved = local_search(S, max_iters=200)
        if len(S_improved) > best_size:
            best_size = len(S_improved)
            best_set = S_improved
    
    return list(best_set)

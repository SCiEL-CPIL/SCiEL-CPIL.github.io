import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit for safety
    
    N = 128
    # 1. Precompute bad triples
    # bad[i] is a set of (j, k) such that {i, j, k} is bad
    # Actually, we need to know if {x, y, z} is bad quickly.
    # Let's create a 3D boolean array or a set of frozensets.
    # Given N=128, a set of frozensets is fine.
    
    def is_bad(a, b, c):
        # Check if for all primes p, top two exponents are equal
        # Equivalent: for all p, not (one is strictly greater than the other two)
        # We can just check lcm(a,b) == lcm(a,c) == lcm(b,c)
        from math import lcm
        l_ab = lcm(a, b)
        l_ac = lcm(a, c)
        l_bc = lcm(b, c)
        return l_ab == l_ac == l_bc

    # Precompute all bad triples
    bad_triples = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                if is_bad(a, b, c):
                    bad_triples.add(frozenset((a, b, c)))
    
    # For fast lookup in search, we can store conflicts per pair
    # conflict_pair[x][y] = set of z such that {x,y,z} is bad
    # But we need to handle x,y,z order.
    # Let's just use the set of frozensets and check membership?
    # Membership in set of frozensets is O(1).
    
    # We want Max Independent Set.
    # Use Branch and Bound.
    
    best_size = 0
    best_set = []
    
    # Initial lower bound using greedy
    def greedy():
        selected = []
        available = set(range(1, N + 1))
        # Sort by degree (number of bad triples involving the number)
        degrees = {i: 0 for i in range(1, N + 1)}
        for t in bad_triples:
            for x in t:
                degrees[x] += 1
        
        # Greedy: pick node with lowest degree? Or highest?
        # For MIS, usually pick node, remove neighbors.
        # Here "neighbors" are complex.
        # Let's just pick 1..N and check.
        for x in range(1, N + 1):
            if all(not (frozenset((x, a, b)) in bad_triples) for a, b in combinations(selected, 2)):
                selected.append(x)
        return selected

    init_set = greedy()
    best_size = len(init_set)
    best_set = init_set
    
    # Search
    # State: current_selected (list), available (set)
    # To prune: if len(current) + len(available) <= best_size, return.
    
    # We need a fast way to check if adding x to current is valid.
    # It is valid if no pair (a,b) in current forms a bad triple with x.
    
    def search(current, available):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return

        if not available:
            if len(current) > best_size:
                best_size = len(current)
                best_set = current[:]
            return

        # Upper bound
        if len(current) + len(available) <= best_size:
            return

        # Heuristic: pick a candidate from available
        # We can pick the one with smallest degree in the induced subgraph?
        # Or just iterate.
        
        # To speed up, we can branch on a specific element.
        # Let's pick the first element in available (sorted) to be deterministic.
        # Or better, pick an element that is in many bad triples within available?
        
        # Simple branching:
        # Option 1: Exclude x
        # Option 2: Include x
        
        # Let's pick x = min(available)
        x = min(available)
        
        # Option 1: Exclude x
        search(current, available - {x})
        
        # Option 2: Include x
        # Check if x can be added to current
        valid = True
        for a, b in combinations(current, 2):
            if frozenset((x, a, b)) in bad_triples:
                valid = False
                break
        
        if valid:
            new_current = current + [x]
            # New available: remove x, and remove any y such that {x, y, z} is bad for some z in new_current?
            # No, if we include x, we must ensure no future picks y,z form bad triple with x and any existing z?
            # Actually, the constraint is on TRIPLES.
            # If we pick x, we must not have any pair (y,z) in the final set such that {x,y,z} is bad.
            # This means if we pick x, we must remove all pairs (y,z) from consideration?
            # This is hard to represent in a simple "available set".
            
            # Alternative: Just check validity when adding to current.
            # So, new_available = available - {x}
            # But we must also ensure that no future pair (y,z) from new_available forms a bad triple with x and ANY element in new_current?
            # No, the bad triple is {x, y, z}. All three must be in the set.
            # So if we pick x, we just can't pick any pair (y,z) that is "conflicting" with x.
            # This effectively removes edges (y,z) from the "allowed pairs" graph.
            # But our "available" set doesn't track pairs.
            
            # This suggests a simple "available set" is not enough for efficient pruning of pairs.
            # However, N=128 is small. We can just recurse and check validity at each step.
            # The bottleneck is the number of recursive calls.
            
            # To prune better:
            # If we pick x, any y in available such that there exists z in available where {x,y,z} is bad
            # does NOT mean y is removed. It means we can't pick BOTH y and z.
            # This is a 2-SAT-like constraint on the remaining set.
            
            # Given the time limit, a simple DFS with the validity check `all(not ...)` might be too slow if the branching factor is high.
            # But `best_size` will prune a lot.
            
            # Let's try to improve the heuristic.
            # Pick x that is in the most bad triples within `available`?
            
            new_available = available - {x}
            search(new_current, new_available)
            
    # Start search with empty set and all numbers
    # To make it faster, we can start with the greedy solution?
    # No, greedy is a lower bound. Search starts from empty.
    
    # We can speed up by ordering the search.
    # Let's just run the search.
    
    # Optimization: Precompute conflict sets for pairs?
    # conflict[x][y] = set of z such that {x,y,z} is bad.
    # Then when we have current, and we want to add x:
    # We check if any (a,b) in current is in conflict[x]?
    # No, we check if {x,a,b} is bad.
    
    # Let's just run the search.
    search([], set(range(1, N + 1)))
    
    return best_set

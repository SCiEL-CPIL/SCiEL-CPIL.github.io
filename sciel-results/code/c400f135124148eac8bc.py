# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    Uses randomized greedy initialization with multi-start local search (swap and add/remove moves).
    """
    m = 205
    deadline = time.time() + 620  # safety margin before 630s kill

    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    forbidden_list = sorted(forbidden)
    
    # Build adjacency: adj[i] = set of j such that (i-j) mod m is in forbidden
    # i.e., i and j cannot both be in the set
    adj = [set() for _ in range(m)]
    for i in range(m):
        for f in forbidden_list:
            j = (i + f) % m
            adj[i].add(j)
            adj[j].add(i)  # symmetric
    
    # Convert to bitmasks for fast operations
    adj_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        adj_mask[i] = mask

    def is_valid_set(S):
        """Check if set S (as a list) is valid (no forbidden differences)."""
        S_set = set(S)
        for a in S:
            for b in S:
                if a == b:
                    continue
                d = (a - b) % m
                if d in forbidden:
                    return False
        return True

    def greedy_init(order):
        """Greedy construction following given order."""
        selected = []
        selected_set = set()
        for candidate in order:
            ok = True
            for s in selected_set:
                d = (candidate - s) % m
                if d in forbidden:
                    ok = False
                    break
            if ok:
                selected.append(candidate)
                selected_set.add(candidate)
        return selected

    def local_search(S, max_iters=50000):
        """Local search with swap moves: remove one, add another if possible."""
        S_set = set(S)
        best = list(S)
        best_size = len(best)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            # Try all swap moves: remove r, add a
            for r in best:
                S_set.remove(r)
                # Find best candidate to add
                best_add = None
                for a in range(m):
                    if a in S_set:
                        continue
                    ok = True
                    for s in S_set:
                        d = (a - s) % m
                        if d in forbidden:
                            ok = False
                            break
                    if ok:
                        best_add = a
                        break
                if best_add is not None:
                    S_set.add(best_add)
                    best = list(S_set)
                    improved = True
                    break
                else:
                    S_set.add(r)
            
            if not improved:
                # Try add-only moves
                added = False
                for a in range(m):
                    if a in S_set:
                        continue
                    ok = True
                    for s in S_set:
                        d = (a - s) % m
                        if d in forbidden:
                            ok = False
                            break
                    if ok:
                        S_set.add(a)
                        best = list(S_set)
                        added = True
                        break
                if not added:
                    break
            
            if len(best) > best_size:
                best_size = len(best)
        
        return best

    # Multi-start local search
    best_solution = None
    best_size = 0
    
    # First, try a few deterministic orderings
    orders = [
        list(range(m)),
        list(range(m, 0, -1)),
        list(range(0, m, 2)) + list(range(1, m, 2)),
        list(range(1, m, 2)) + list(range(0, m, 2)),
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        S = greedy_init(order)
        S = local_search(S, max_iters=20000)
        if len(S) > best_size:
            best_size = len(S)
            best_solution = S
    
    # Now randomized restarts
    rng = random.Random(42)
    restarts = 0
    while time.time() < deadline - 10:
        restarts += 1
        # Random permutation
        order = list(range(m))
        rng.shuffle(order)
        S = greedy_init(order)
        S = local_search(S, max_iters=15000)
        if len(S) > best_size:
            best_size = len(S)
            best_solution = S
        
        # Also try: start from best, remove a random element, re-add greedily
        if best_solution is not None and len(best_solution) > 3:
            if time.time() > deadline:
                break
            S2 = list(best_solution)
            # Remove a random element
            idx = rng.randint(0, len(S2) - 1)
            removed = S2.pop(idx)
            # Try to add back different elements
            for a in range(m):
                if a in S2:
                    continue
                ok = True
                for s in S2:
                    d = (a - s) % m
                    if d in forbidden:
                        ok = False
                        break
                if ok:
                    S2.append(a)
                    break
            S2 = local_search(S2, max_iters=10000)
            if len(S2) > best_size:
                best_size = len(S2)
                best_solution = S2
    
    if best_solution is None:
        best_solution = [0]
    
    return np.array(sorted(best_solution), dtype=np.int64)


# EVOLVE-BLOCK-END

import time
import random
from math import lcm
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 128

    # Pre-compute all bad triples and conflict structures
    conflict_map = defaultdict(set)  # conflict_map[v] = set of frozenset({u,w}) such that {v,u,w} is bad
    bad_triples = []
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm(a, b)
            for c in range(b + 1, N + 1):
                if lcm(a, c) == lab and lcm(b, c) == lab:
                    triple = frozenset((a, b, c))
                    bad_triples.append(triple)
                    conflict_map[a].add(frozenset((b, c)))
                    conflict_map[b].add(frozenset((a, c)))
                    conflict_map[c].add(frozenset((a, b)))
    
    if time.time() > deadline:
        return list(range(64, 129))
    
    # Precompute conflict count for each element (degree)
    degrees = {v: len(conflict_map[v]) for v in range(1, N + 1)}
    
    # Fast membership check helper
    def can_add(v, S):
        for pair in conflict_map[v]:
            u, w = tuple(pair)
            if u in S and w in S:
                return False
        return True
    
    # Count bad triples in set S using precomputed list
    def count_bad(S):
        S_set = S if isinstance(S, set) else set(S)
        count = 0
        for t in bad_triples:
            if t[0] in S_set and t[1] in S_set and t[2] in S_set:
                count += 1
        return count
    
    # Greedy initialization with ordering
    def greedy_init(order=None):
        S = set()
        if order is None:
            order = sorted(range(1, N + 1), key=lambda x: degrees[x])
        for v in order:
            if can_add(v, S):
                S.add(v)
        return S
    
    # Local search: try to improve set S
    def local_search(S, max_iters=10000, deadline_check=None):
        S = set(S)
        best_S = set(S)
        best_size = len(S)
        
        # Try to add elements first
        def try_add_all(S):
            improved = True
            while improved:
                improved = False
                for v in range(1, N + 1):
                    if v not in S and can_add(v, S):
                        S.add(v)
                        improved = True
            return S
        
        S = try_add_all(S)
        best_S = set(S)
        best_size = len(S)
        
        # Now do swap-based local search
        import math
        def exp_safe(x):
            if x > 50: return 1.0
            if x < -50: return 0.0
            return math.exp(x)
        
        temp = 1.0
        cooling = 0.9995
        no_improve = 0
        max_no_improve = 500
        
        for i in range(max_iters):
            if deadline_check is not None and time.time() > deadline_check:
                break
            if no_improve > max_no_improve:
                break
            
            # Try all possible single additions
            added = False
            for v in range(1, N + 1):
                if v not in S and can_add(v, S):
                    S.add(v)
                    added = True
                    if len(S) > best_size:
                        best_S = set(S)
                        best_size = len(S)
                        no_improve = 0
                    break
            
            if added:
                continue
            
            # Try swap: remove one, add one
            # Pick a random element to remove
            if S:
                v_remove = random.choice(list(S))
                S.discard(v_remove)
                
                # Now try to add any element
                added_swap = False
                for v_add in range(1, N + 1):
                    if v_add not in S and can_add(v_add, S):
                        S.add(v_add)
                        added_swap = True
                        if len(S) > best_size:
                            best_S = set(S)
                            best_size = len(S)
                            no_improve = 0
                        break
                
                if not added_swap:
                    # Put v_remove back
                    S.add(v_remove)
                else:
                    # Check if we should keep the swap
                    # If size is same or better, keep it
                    pass
            else:
                no_improve += 1
            
            temp *= cooling
        
        # Final add-all
        best_S = try_add_all(best_S)
        
        return best_S
    
    # Run greedy + local search with multiple strategies
    random.seed(42)
    
    best_set = greedy_init()
    best_size = len(best_set)
    
    # Local search on greedy result
    if time.time() < deadline - 15:
        improved = local_search(best_set, max_iters=5000, deadline_check=deadline - 10)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
    
    # Multiple diversified restarts
    for restart in range(30):
        if time.time() > deadline - 8:
            break
        
        # Diversified initialization
        S_init = set()
        order = list(range(1, N + 1))
        random.shuffle(order)
        # Sort by degree + noise for diversity
        order.sort(key=lambda x: degrees[x] + random.random() * 10)
        
        for v in order:
            if can_add(v, S_init):
                S_init.add(v)
        
        # Local search
        S_improved = local_search(S_init, max_iters=3000, deadline_check=deadline - 8)
        
        if len(S_improved) > best_size:
            best_set = S_improved
            best_size = len(best_set)
    
    # Final polish: exhaustive add
    def try_add_more(S):
        improved = True
        while improved:
            improved = False
            for v in range(1, N + 1):
                if v not in S and can_add(v, S):
                    S.add(v)
                    improved = True
        return S
    
    best_set = try_add_more(best_set)
    
    return list(best_set)

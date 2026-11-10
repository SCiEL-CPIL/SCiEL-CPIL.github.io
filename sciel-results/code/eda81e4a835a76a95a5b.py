import random
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    # 1. Precomputation
    # Generate all triples
    all_triples = []
    triple_indices = [] # (a,b,c)
    for a in range(16):
        for b in range(a + 1, 16):
            for c in range(b + 1, 16):
                all_triples.append((a, b, c))
                triple_indices.append((1 << a) | (1 << b) | (1 << c))
    
    n_triples = len(all_triples) # 560
    full_mask = (1 << 16) - 1
    
    # Precompute compatibility: compat[i] is a set (or list) of indices j such that triple i and j share <= 1 point
    # Using bitsets for fast intersection checks later? 
    # For move generation, we need to find edges compatible with the current set.
    # It's faster to maintain a 'blocked' set or check on the fly.
    
    # Precompute supersets for each triple mask
    # supersets[i] is a list of all masks M such that triple_indices[i] is a subset of M
    supersets = []
    for i in range(n_triples):
        t_mask = triple_indices[i]
        # Remaining bits
        rem = full_mask ^ t_mask
        subs = [rem]
        # Generate all submasks of rem
        sub = rem
        while True:
            subs.append(sub)
            if sub == 0:
                break
            sub = (sub - 1) & rem
        # The supersets of t_mask are t_mask | s for s in subs
        # Actually, we just need the indices.
        sup_list = [t_mask | s for s in subs]
        supersets.append(sup_list)
        
    # Precompute popcounts for all masks
    popcounts = np.array([m.bit_count() for m in range(1 << 16)], dtype=np.uint8)
    
    # State
    # count[M] = number of selected triples contained in M
    count = np.zeros(1 << 16, dtype=np.uint16)
    current_edges = set()
    
    def add_edge(i):
        current_edges.add(i)
        count[supersets[i]] += 1
        
    def remove_edge(i):
        current_edges.remove(i)
        count[supersets[i]] -= 1
        
    def get_alpha():
        # Find max popcount of M where count[M] == 0
        valid = count == 0
        if not valid.any():
            return 0
        return int(np.max(popcounts[valid]))
        
    def is_compatible(i, edges_set):
        # Check if triple i shares <= 1 point with all triples in edges_set
        # This is O(|edges_set|). 
        # For speed, we can precompute pair conflicts.
        # But with 560 triples, checking against ~30 edges is fast.
        t_i = triple_indices[i]
        for j in edges_set:
            t_j = triple_indices[j]
            if (t_i & t_j).bit_count() > 1:
                return False
        return True
        
    # Optimized compatibility check using bitsets of edges?
    # Let's precompute a conflict matrix or lists.
    # conflicts[i] = set of j such that i and j share > 1 point.
    conflicts = [set() for _ in range(n_triples)]
    for i in range(n_triples):
        ti = triple_indices[i]
        for j in range(i + 1, n_triples):
            tj = triple_indices[j]
            if (ti & tj).bit_count() > 1:
                conflicts[i].add(j)
                conflicts[j].add(i)
                
    def get_compatible_candidates(edges_set):
        # Returns list of edge indices not in edges_set and compatible with all in edges_set
        # Start with all edges, remove those in edges_set, remove those in union of conflicts
        # This is O(n_triples) per call if we use sets.
        # 560 is small.
        candidates = []
        for i in range(n_triples):
            if i in edges_set:
                continue
            # Check conflicts
            ok = True
            for j in edges_set:
                if j in conflicts[i]:
                    ok = False
                    break
            if ok:
                candidates.append(i)
        return candidates

    # Greedy Construction with Lookahead (Minimize Alpha)
    def greedy_construct():
        edges = set()
        # We want to build a system that minimizes alpha.
        # Strategy: At each step, pick the edge that minimizes the alpha of the resulting system.
        # This is expensive but we only do it ~30 times.
        
        current_edges_set = set()
        
        # Initialize count to 0
        count[:] = 0
        
        while True:
            candidates = get_compatible_candidates(current_edges_set)
            if not candidates:
                break
            
            best_edge = None
            best_alpha = 17
            
            # To save time, if candidates > 100, sample a subset?
            # Or just evaluate all. 500 candidates * 0.1ms = 50ms. 30 steps = 1.5s. OK.
            
            for e in candidates:
                # Simulate adding e
                add_edge(e)
                alpha = get_alpha()
                remove_edge(e)
                
                if alpha < best_alpha:
                    best_alpha = alpha
                    best_edge = e
                    if best_alpha == 8: # Theoretical lower bound? 
                        # For v=16, alpha >= 8? 
                        # If we find 8, we can stop? 
                        # Actually, lower bound is ceil(16/2)=8? No.
                        # Just keep going if fast.
                        pass
                        
            if best_edge is not None:
                add_edge(best_edge)
                current_edges_set.add(best_edge)
            else:
                break
                
        return current_edges_set, get_alpha()

    # Local Search (Simulated Annealing)
    def local_search(initial_edges, initial_alpha, max_time):
        current_edges = set(initial_edges)
        current_alpha = initial_alpha
        best_edges = set(current_edges)
        best_alpha = current_alpha
        
        # Sync count array
        count[:] = 0
        for e in current_edges:
            add_edge(e)
            
        T = 2.0
        T_min = 0.05
        cooling = 0.999
        
        rng = random.Random()
        
        while time.time() < max_time:
            if T < T_min:
                T = T_min
                
            # Move: Swap
            if not current_edges:
                # If empty, just add a random edge
                candidates = get_compatible_candidates(current_edges)
                if not candidates: break
                e_new = rng.choice(candidates)
                add_edge(e_new)
                current_edges.add(e_new)
                current_alpha = get_alpha()
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_edges = set(current_edges)
                T *= cooling
                continue
                
            # Pick an edge to remove
            e_old = rng.choice(list(current_edges))
            
            # Find candidates to add
            # Temporarily remove e_old to check compatibility?
            # No, e_old is in current_edges.
            # Candidates must be compatible with (current_edges - {e_old})
            temp_edges = current_edges - {e_old}
            candidates = get_compatible_candidates(temp_edges)
            
            if not candidates:
                # Try removing e_old without adding? (Decrease size)
                # Usually we want to keep size roughly constant or increase.
                # Removing reduces edges, likely increases alpha.
                # Skip this move type for now, focus on swaps.
                T *= cooling
                continue
            
            e_new = rng.choice(candidates)
            
            # Perform Swap
            remove_edge(e_old)
            current_edges.remove(e_old)
            
            add_edge(e_new)
            current_edges.add(e_new)
            
            new_alpha = get_alpha()
            
            delta = new_alpha - current_alpha
            
            if delta <= 0 or rng.random() < (2.718 ** (-delta / T)):
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_edges = set(current_edges)
            else:
                # Reject, revert
                remove_edge(e_new)
                current_edges.remove(e_new)
                
                add_edge(e_old)
                current_edges.add(e_old)
                
            T *= cooling
            
        return best_edges, best_alpha

    # Main
    best_overall_alpha = 16
    best_overall_edges = set()
    
    # Phase 1: Greedy Init
    init_edges, init_alpha = greedy_construct()
    if init_alpha < best_overall_alpha:
        best_overall_alpha = init_alpha
        best_overall_edges = init_edges
        
    # Phase 2: Local Search
    # Run multiple SA instances
    time_limit_per_sa = 20.0
    num_sa = int((deadline - time.time() - 5) / time_limit_per_sa)
    if num_sa < 1:
        num_sa = 1
        
    for i in range(num_sa):
        if time.time() > deadline - 5:
            break
            
        # Start from best or a random greedy
        if i == 0 or random.random() < 0.5:
            start_edges = set(best_overall_edges)
            start_alpha = best_overall_alpha
        else:
            start_edges, start_alpha = greedy_construct()
            
        max_t = deadline - 5 - time.time()
        if max_t < 2: break
        
        ls_edges, ls_alpha = local_search(start_edges, start_alpha, time.time() + max_t)
        
        if ls_alpha < best_overall_alpha:
            best_overall_alpha = ls_alpha
            best_overall_edges = ls_edges

    return [list(all_triples[i]) for i in sorted(best_overall_edges)]

# EVOLVE-BLOCK-END

"""Direct structural approach for equal-LCM-triple-free subset of [1,128]."""
import time
import random
from math import gcd
from itertools import combinations

def construct():
    deadline = time.time() + 80
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute LCM table
    lcm_table = {}
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcm_table[(i, j)] = i * j // gcd(i, j)
    
    # Find all bad triples
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[(a, b)]
            for c in range(b + 1, N + 1):
                if lcm_table[(a, c)] == lab and lcm_table[(b, c)] == lab:
                    bad_triples.append((a, b, c))
    
    # Build conflict structure: for each element, list of (y,z) pairs
    conflict_pairs = {x: [] for x in range(1, N + 1)}
    for (a, b, c) in bad_triples:
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))
    
    # Convert to frozensets for faster membership
    conflict_pairs_fs = {x: [frozenset(p) for p in conflict_pairs[x]] for x in range(1, N + 1)}
    
    def can_add(x, S):
        """Check if x can be added to S without creating a bad triple."""
        for yz in conflict_pairs_fs[x]:
            if yz[0] in S and yz[1] in S:
                return False
        return True
    
    def can_add_fast(x, S_set):
        """Faster check using precomputed sets."""
        for y, z in conflict_pairs[x]:
            if y in S_set and z in S_set:
                return False
        return True
    
    def greedy_matching_bound(candidates, S):
        """Compute upper bound on how many more elements can be added.
        Uses a greedy matching on the conflict hypergraph: each edge (y,z) 
        that is 'active' (both in candidates) can block at most one additional element.
        Upper bound = len(candidates) - min_edges_needed
        where min_edges_needed is a lower bound on how many elements must be excluded.
        """
        if not candidates:
            return 0
        
        # Simple bound: for each candidate x, count how many conflict pairs are fully in candidates
        # The number of elements we MUST exclude is at least the size of a minimum vertex cover
        # of the hypergraph, which is at least the size of a maximum matching.
        
        # Greedy matching: repeatedly pick a conflict edge (y,z) where both y,z in candidates,
        # then remove one of them from consideration.
        cand_set = set(candidates)
        matched = 0
        used = set()
        
        # Collect all active edges
        active_edges = []
        for x in candidates:
            for y, z in conflict_pairs[x]:
                if y in cand_set and z in cand_set and y != z:
                    edge = (min(y,z), max(y,z))
                    if edge not in used:
                        active_edges.append(edge)
                        used.add(edge)
        
        # Greedy matching on the graph formed by active edges
        matched_vertices = set()
        for u, v in active_edges:
            if u not in matched_vertices and v not in matched_vertices:
                matched_vertices.add(u)
                matched_vertices.add(v)
                matched += 1
        
        # Each matched edge means at least one of its endpoints must be excluded
        # So we can add at most len(candidates) - matched elements
        return len(candidates) - matched
    
    def branch_and_bound(S, candidates, best_size, best_set):
        """Branch and bound search for maximum valid subset."""
        if time.time() > deadline - 10:
            return best_size, best_set
        
        if not candidates:
            if len(S) > best_size:
                best_size = len(S)
                best_set = set(S)
            return best_size, best_set
        
        # Upper bound using matching
        ub = len(S) + greedy_matching_bound(candidates, S)
        if ub <= best_size:
            return best_size, best_set
        
        # Pick the most constrained candidate
        best_x = None
        best_conflict_count = -1
        for x in candidates:
            count = 0
            for y, z in conflict_pairs[x]:
                if y in candidates and z in candidates:
                    count += 1
            if count > best_conflict_count:
                best_conflict_count = count
                best_x = x
        
        if best_x is None:
            best_x = next(iter(candidates))
        
        # Branch 1: Include best_x
        if can_add_fast(best_x, S):
            S_new = S | {best_x}
            candidates_new = set()
            for x in candidates:
                if x != best_x and can_add_fast(x, S_new):
                    candidates_new.add(x)
            best_size, best_set = branch_and_bound(S_new, candidates_new, best_size, best_set)
        
        # Branch 2: Exclude best_x
        candidates_new = set(candidates)
        candidates_new.discard(best_x)
        best_size, best_set = branch_and_bound(S, candidates_new, best_size, best_set)
        
        return best_size, best_set
    
    def greedy_fill(S, order=None):
        """Fill S by adding elements in given order (or natural order)."""
        if order is None:
            order = elements
        for x in order:
            if x not in S and can_add_fast(x, S):
                S.add(x)
        return S
    
    def local_search(S, max_rounds=50):
        """Try to improve S via additions and 1-out-1-in swaps."""
        S = set(S)
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and time.time() < deadline - 5:
            improved = False
            rounds += 1
            # Try additions
            for x in elements:
                if x not in S and can_add_fast(x, S):
                    S.add(x)
                    improved = True
            # Try swaps: remove x, try adding y
            for x in list(S):
                S.remove(x)
                best_y = None
                for y in elements:
                    if y != x and y not in S and can_add_fast(y, S):
                        S.add(y)
                        # Try to fill more
                        for z in elements:
                            if z not in S and can_add_fast(z, S):
                                S.add(z)
                        best_y = y
                        break
                if best_y is not None:
                    improved = True
                else:
                    S.add(x)
        return S
    
    best = set()
    best_size = 0
    
    # Strategy 1: Greedy with different orderings
    triple_count = {x: len(conflict_pairs[x]) for x in range(1, N + 1)}
    
    orderings = [
        sorted(elements, key=lambda x: (triple_count[x], x)),
        sorted(elements, key=lambda x: (triple_count[x], -x)),
        sorted(elements, key=lambda x: x),
        sorted(elements, key=lambda x: -x),
        sorted(elements, key=lambda x: (triple_count[x], x % 2, x)),
        sorted(elements, key=lambda x: (triple_count[x], -(x % 2), x)),
    ]
    
    for order in orderings:
        if time.time() > deadline - 25:
            break
        S = set()
        greedy_fill(S, order)
        S = local_search(S, max_rounds=40)
        if len(S) > best_size:
            best_size = len(S)
            best = set(S)
    
    # Strategy 2: Branch and bound from a good starting point
    if time.time() < deadline - 20:
        S_start = set(best)
        candidates = set()
        for x in elements:
            if x not in S_start and can_add_fast(x, S_start):
                candidates.add(x)
        
        if candidates:
            best_size, best = branch_and_bound(S_start, candidates, best_size, best)
    
    # Strategy 3: Randomized greedy with local search
    random.seed(42)
    num_starts = 200
    for start in range(num_starts):
        if time.time() > deadline - 15:
            break
        
        order = elements[:]
        random.shuffle(order)
        
        S = set()
        greedy_fill(S, order)
        S = local_search(S, max_rounds=25)
        
        if len(S) > best_size:
            best_size = len(S)
            best = set(S)
    
    # Strategy 4: Another B&B pass from the best found
    if time.time() < deadline - 10:
        S_start = set(best)
        candidates = set()
        for x in elements:
            if x not in S_start and can_add_fast(x, S_start):
                candidates.add(x)
        if candidates:
            best_size, best = branch_and_bound(S_start, candidates, best_size, best)
    
    # Final fill
    if time.time() < deadline - 3:
        greedy_fill(best)
    
    if not best:
        best = {1}
    
    return list(best)

"""Rewritten program using greedy minimum-degree set packing on the sunflower conflict hypergraph."""
from itertools import combinations

def construct():
    n = 16
    k = 4
    all_sets = list(combinations(range(1, n + 1), k))
    # Precompute set objects for fast intersection
    set_objs = [frozenset(s) for s in all_sets]
    num_sets = len(all_sets)
    
    # We need to find a large independent set in the 3-uniform hypergraph
    # where an edge is a triple (i, j, k) that forms a sunflower.
    # Sunflower condition: three distinct sets A, B, C such that
    # A ∩ B = A ∩ C = B ∩ C.
    
    # Strategy: Greedy algorithm that picks the set with minimum "conflict degree"
    # where conflict degree is the number of pairs (a, b) already chosen such that
    # adding the candidate would form a sunflower with some pair, or more precisely,
    # we track which pairs of chosen sets would be "completed" by a candidate.
    
    # Actually, let's think differently. A triple (i, j, k) is forbidden if
    # set_objs[i] & set_objs[j] == set_objs[i] & set_objs[k] == set_objs[j] & set_objs[k].
    
    # For greedy: maintain a set of chosen indices.
    # For each candidate c not yet chosen, we need to check if there exist
    # two chosen sets a, b such that (a, b, c) is a sunflower.
    # That is: set_objs[a] & set_objs[b] == set_objs[a] & set_objs[c] == set_objs[b] & set_objs[c].
    
    # To speed up, we can precompute for each pair (a, b) the intersection I_ab = A ∩ B.
    # Then (a, b, c) is a sunflower iff I_ab == I_ac == I_bc.
    # Note I_ac = A ∩ C, I_bc = B ∩ C.
    
    # Alternative: for each candidate c, and for each pair (a, b) in chosen,
    # check if A∩B == A∩C == B∩C.
    
    # Let's precompute intersections between all pairs for quick lookup.
    # There are C(1820, 2) ≈ 1.66M pairs, which is feasible.
    
    # Actually, num_sets = C(16,4) = 1820.
    # Pairs: 1820*1819/2 ≈ 1.66M. Storing frozenset intersections is fine.
    
    # Precompute pair intersections
    pair_int = {}
    for i in range(num_sets):
        si = set_objs[i]
        for j in range(i + 1, num_sets):
            sj = set_objs[j]
            inter = si & sj
            pair_int[(i, j)] = inter
    
    # Now greedy:
    chosen = []
    chosen_set = set()
    
    # For efficiency, we can maintain for each candidate c a count of how many
    # pairs (a, b) in chosen would form a sunflower with c.
    # When we add a new set x to chosen, we need to update counts for all
    # candidates c: for each a in chosen (before adding x), check if (a, x, c)
    # is a sunflower.
    
    # Let's maintain conflict_count[c] = number of pairs (a, b) in chosen such that
    # (a, b, c) is a sunflower. Initially all zero.
    
    conflict_count = [0] * num_sets
    available = set(range(num_sets))
    
    while available:
        # Find candidate with minimum conflict_count
        best_c = None
        best_count = float('inf')
        for c in available:
            if conflict_count[c] < best_count:
                best_count = conflict_count[c]
                best_c = c
            if best_count == 0:
                break
        
        if best_c is None:
            break
        
        # Add best_c to chosen
        chosen.append(best_c)
        available.discard(best_c)
        
        # Update conflict counts for all remaining candidates
        # For each a in chosen (excluding best_c), check if (a, best_c, c) is a sunflower
        for a_idx in chosen[:-1]:
            ia = set_objs[a_idx]
            ib = set_objs[best_c]
            iab = ia & ib
            # For candidate c, we need iac == iab and ibc == iab
            # iac = ia & set_objs[c], ibc = ib & set_objs[c]
            for c in available:
                sc = set_objs[c]
                if ia & sc == iab and ib & sc == iab:
                    conflict_count[c] += 1
    
    return [sorted(set_objs[i]) for i in chosen]

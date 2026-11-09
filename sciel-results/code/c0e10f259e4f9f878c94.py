"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    n = 16
    points = list(range(n))
    
    # Generate all C(16,3) = 560 triples
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)
    
    # Map each point to list of triples containing it
    point_to_triples = [[] for _ in range(n)]
    for t_idx, (a, b, c) in enumerate(all_triples):
        point_to_triples[a].append(t_idx)
        point_to_triples[b].append(t_idx)
        point_to_triples[c].append(t_idx)
    
    # Map each pair of points to list of triples containing both
    pair_to_triples = {}
    for t_idx, (a, b, c) in enumerate(all_triples):
        for p1, p2 in [(a, b), (a, c), (b, c)]:
            pair_key = (min(p1, p2), max(p1, p2))
            if pair_key not in pair_to_triples:
                pair_to_triples[pair_key] = []
            pair_to_triples[pair_key].append(t_idx)
    
    # For each triple, compute the set of "conflicting" triples that would form a K(2,2,2)
    # A K(2,2,2) is formed by 3 disjoint pairs (a1,a2), (b1,b2), (c1,c2)
    # The 8 transversal triples are {ai, bj, ck} for all i,j,k in {0,1}
    # Two triples conflict if they can be part of the same K(2,2,2) where all 8 transversals are present
    
    # Actually, we need: no 3 disjoint pairs have all 8 transversal triples present
    # So we need to check if adding a new triple completes any K(2,2,2)
    
    # Precompute for each triple, the list of other triples that share exactly 2 points
    # (since in a K(2,2,2), any two of the 8 triples share exactly 2 points)
    
    # For a given triple (a, b, c), the 7 other triples in a K(2,2,2) that contains it
    # are determined by choosing a partner for each point and then combining
    # If triple = (a, b, c), then the other 7 are:
    # (a', b, c), (a, b', c), (a, b, c'), 
    # (a', b', c), (a', b, c'), (a, b', c'),
    # (a', b', c')
    # where a' != a, b' != b, c' != c, and {a,a'}, {b,b'}, {c,c'} are the 3 disjoint pairs
    
    # So for triple (a, b, c), the conflicting triples are:
    # For each a' in points\{a,b,c}, each b' in points\{a,b,c}, each c' in points\{a,b,c}:
    # The 7 triples: (a',b,c), (a,b',c), (a,b,c'), (a',b',c), (a',b,c'), (a,b',c'), (a',b',c')
    # But we only need to check if ALL 8 are present, so we need to track which of these are in our set
    
    # Better approach: for each triple, precompute the list of "completions"
    # A completion is a set of 7 other triples such that together with this triple they form a K(2,2,2)
    
    # For triple t = (a, b, c), a K(2,2,2) containing t is determined by choosing:
    # a' (the other point in the pair with a), b' (other point in pair with b), c' (other point in pair with c)
    # where a', b', c' are all distinct and not in {a, b, c}
    # The 8 triples are: (a,b,c), (a',b,c), (a,b',c), (a,b,c'), (a',b',c), (a',b,c'), (a,b',c'), (a',b',c')
    
    # So for each triple and each valid (a', b', c'), we get a K(2,2,2)
    # We need to ensure that no K(2,2,2) has all 8 triples in our set
    
    # Precompute: for each triple t_idx, list of K(2,2,2) IDs it belongs to
    # And for each K(2,2,2) ID, the 8 triples that form it
    
    k222_list = []  # list of (tuple of 8 triple indices)
    triple_to_k222s = [[] for _ in range(num_triples)]
    
    # Generate all K(2,2,2) configurations
    # Choose 3 disjoint pairs from 16 points
    # Then the 8 transversal triples must all be valid (which they are since points are distinct)
    
    # Number of ways to choose 3 disjoint pairs from 16 points:
    # C(16,2) * C(14,2) * C(12,2) / 6 = 120 * 91 * 66 / 6 = 116280
    # But we need to be careful about order - each set of 3 pairs is counted 6 times
    
    # Let's enumerate more efficiently
    # For each unordered set of 6 distinct points, we can form C(6,2)*C(4,2)*C(2,2)/6 ways... 
    # Actually, let's just iterate over all possible K(2,2,2)s
    
    # A K(2,2,2) is defined by 3 unordered pairs. Let's enumerate by choosing 6 distinct points,
    # then partitioning them into 3 unordered pairs.
    # Number of partitions of 6 points into 3 pairs: 15 (3!! = 3*1*1 = 3... wait)
    # Actually: (6-1)!! = 5!! = 5*3*1 = 15
    
    # So total K(2,2,2)s: C(16,6) * 15 = 8008 * 15 = 120120
    
    # For each such K(2,2,2), we need to check if all 8 transversal triples are in our set
    # But for precomputation, we just need to list which K(2,2,2)s each triple belongs to
    
    # Let's build this mapping
    # For each set of 6 points, for each partition into 3 pairs, form the 8 triples
    
    # This might be slow. Let's use a different approach:
    # For each triple (a,b,c), the K(2,2,2)s it belongs to are determined by choosing
    # a' from remaining 13 points, b' from remaining 12 (excluding a'), c' from remaining 11 (excluding a',b')
    # But this overcounts because the same K(2,2,2) can be reached multiple ways
    
    # Actually, for a fixed triple (a,b,c), each K(2,2,2) containing it is uniquely determined by
    # the choice of a', b', c' (one from each "position"). So there are 13*12*11 = 1716 K(2,2,2)s
    # containing any given triple. But many of these will have duplicate triples or invalid ones.
    
    # Let's just precompute the conflict structure directly:
    # Two triples conflict if they share exactly 2 points (i.e., they differ in exactly one point)
    # But that's not quite right - we need to check if adding a triple completes a K(2,2,2)
    
    # Simpler approach: use a greedy algorithm with backtracking
    # Or use the fact that we just need to maximize the number of triples with no complete K(2,2,2)
    
    # Let's use a different strategy: 
    # Start with all 560 triples, then remove triples to eliminate all K(2,2,2)s
    # This is a hitting set problem
    
    # Actually, let's use a greedy approach:
    # Randomly order the triples, and add each one if it doesn't complete any K(2,2,2)
    # To check if adding triple t completes a K(2,2,2), we need to check all K(2,2,2)s that contain t
    # and see if the other 7 triples are already in our set
    
    # Precompute for each triple t, the list of K(2,2,2) "patterns" it participates in
    # A pattern is the set of 7 other triples needed to complete the K(2,2,2)
    
    # For triple (a, b, c), for each choice of a' != a, b' != b, c' != c, all distinct and not in {a,b,c}:
    # The 7 other triples are:
    # (a', b, c), (a, b', c), (a, b, c'), (a', b', c), (a', b, c'), (a, b', c'), (a', b', c')
    # We need to map these to triple indices
    
    # Let's create a lookup: triple -> index
    triple_index = {}
    for i, t in enumerate(all_triples):
        triple_index[t] = i
    
    # For each triple, store the list of "completion sets" (each is a frozenset of 7 triple indices)
    triple_completions = [[] for _ in range(num_triples)]
    
    for t_idx, (a, b, c) in enumerate(all_triples):
        remaining = [x for x in range(n) if x not in (a, b, c)]
        for a_prime in remaining:
            for b_prime in remaining:
                if b_prime == a_prime:
                    continue
                for c_prime in remaining:
                    if c_prime in (a_prime, b_prime):
                        continue
                    # The 7 other triples
                    others = [
                        (a_prime, b, c),
                        (a, b_prime, c),
                        (a, b, c_prime),
                        (a_prime, b_prime, c),
                        (a_prime, b, c_prime),
                        (a, b_prime, c_prime),
                        (a_prime, b_prime, c_prime)
                    ]
                    # Sort each to match the canonical form
                    others_sorted = [tuple(sorted(t)) for t in others]
                    # Check if all exist in our triple_index
                    if all(o in triple_index for o in others_sorted):
                        completion = frozenset(triple_index[o] for o in others_sorted)
                        triple_completions[t_idx].append(completion)
    
    # Now do greedy with random restarts
    best_size = 0
    best_set = None
    
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        random.seed(restart * 12345 + 67890)
        
        # Random order of triples
        order = list(range(num_triples))
        random.shuffle(order)
        
        in_set = [False] * num_triples
        completion_counts = [0] * num_triples  # For each triple, how many completions are fully satisfied
        # Actually, let's track for each completion frozenset, how many of its members are in the set
        # But completions are per-triple, so let's track differently
        
        # For each triple t, for each completion C of t, track how many of the 7 triples in C are in our set
        # When we add triple t, for each completion C of t, we check if all 7 are in the set
        # If so, we can't add t
        
        # Simpler: when considering adding triple t, check all completions of t
        # For each completion C, count how many of the 7 triples are already in_set
        # If any completion has all 7 in_set, then adding t would complete a K(2,2,2)
        
        current_size = 0
        for t_idx in order:
            can_add = True
            for completion in triple_completions[t_idx]:
                # Check if all 7 triples in completion are in_set
                all_in = True
                for other_idx in completion:
                    if not in_set[other_idx]:
                        all_in = False
                        break
                if all_in:
                    can_add = False
                    break
            
            if can_add:
                in_set[t_idx] = True
                current_size += 1
        
        if current_size > best_size:
            best_size = current_size
            best_set = in_set.copy()
    
    # Convert to list of triples
    result = []
    for i, in_set_flag in enumerate(best_set):
        if in_set_flag:
            result.append(list(all_triples[i]))
    
    return result
# EVOLVE-BLOCK-END

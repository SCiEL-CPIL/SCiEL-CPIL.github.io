"""Triple systems on 16 points without K(2,2,2) - local search with precomputed conflicts."""
import time
import random
from itertools import combinations

def construct():
    n = 16
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # All possible triples
    all_triples = list(combinations(range(n), 3))
    triple_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # For each 6-point set, compute the 8 transversal triples of K(2,2,2)
    # K(2,2,2) has 3 parts of size 2. For 6 points {a,b,c,d,e,f},
    # there are 15 ways to partition into 3 pairs.
    # For each partition, the 8 transversals are all triples picking one from each part.
    
    # Precompute: for each 6-set, list of 15 partitions, each with 8 transversal triple indices
    six_sets = list(combinations(range(n), 6))
    
    # For each 6-set, store list of lists of 8 triple indices (one per partition)
    # conflict_count[i] = number of 6-sets where triple i is one of the 8 transversals
    # Actually, we need: for each 6-set, which triples are "bad" if all 8 are present
    
    # Better approach: for each 6-set, precompute all 15*8 = 120 triple indices
    # But we only care about the 8-transversal groups.
    
    # Let's precompute for each 6-set: list of 15 groups of 8 triple indices
    # When we add/remove a triple, we update counts for each 6-set it belongs to
    
    # For each triple, which 6-sets contain it? (up to C(13,3)=286 six-sets)
    # For each 6-set containing triple t, which partition(s) include t as a transversal?
    
    # This is getting complex. Let's use a different approach:
    # For each 6-set, precompute the 15 groups of 8 triples.
    # Maintain a count for each 6-set: how many of its 15 groups have all 8 triples present.
    # Actually, we need: no group has all 8 present.
    # So for each 6-set and each of its 15 groups, maintain a count of how many of the 8 triples are in our system.
    # If any count reaches 8, it's invalid.
    
    # Total 6-sets: C(16,6) = 8008
    # Each has 15 groups of 8 triples.
    # Total group-triple memberships: 8008 * 15 * 8 = 960,960
    
    # Precompute: for each 6-set index, list of 15 groups, each group is a tuple of 8 triple indices
    six_set_idx = {s: i for i, s in enumerate(six_sets)}
    six_set_triples = []  # six_set_triples[i] = list of 15 groups, each group = tuple of 8 triple indices
    
    # For each 6-set, generate all 15 pair partitions and their 8 transversals
    for s in six_sets:
        a, b, c, d, e, f = s
        groups = []
        # 15 ways to partition 6 elements into 3 pairs
        # Pair (a,b): remaining {c,d,e,f} -> 3 ways to pair: (c,d)(e,f), (c,e)(d,f), (c,f)(d,e)
        # Pair (a,c): remaining {b,d,e,f} -> 3 ways
        # Pair (a,d): remaining {b,c,e,f} -> 3 ways
        # Pair (a,e): remaining {b,c,d,f} -> 3 ways
        # Pair (a,f): remaining {b,c,d,e} -> 3 ways
        # That's 15 total
        
        def get_transversals(p1, p2, p3):
            """Given 3 pairs, return 8 transversals (one element from each pair)."""
            result = []
            for x in p1:
                for y in p2:
                    for z in p3:
                        t = tuple(sorted((x, y, z)))
                        result.append(t)
            return result
        
        # Generate all 15 partitions
        pairs_list = [
            [(a,b),(c,d),(e,f)],
            [(a,b),(c,e),(d,f)],
            [(a,b),(c,f),(d,e)],
            [(a,c),(b,d),(e,f)],
            [(a,c),(b,e),(d,f)],
            [(a,c),(b,f),(d,e)],
            [(a,d),(b,c),(e,f)],
            [(a,d),(b,e),(c,f)],
            [(a,d),(b,f),(c,e)],
            [(a,e),(b,c),(d,f)],
            [(a,e),(b,d),(c,f)],
            [(a,e),(b,f),(c,d)],
            [(a,f),(b,c),(d,e)],
            [(a,f),(b,d),(c,e)],
            [(a,f),(b,e),(c,d)],
        ]
        
        for p1, p2, p3 in pairs_list:
            trans = get_transversals(p1, p2, p3)
            # Convert to triple indices
            group = tuple(triple_idx[t] for t in trans)
            groups.append(group)
        
        six_set_triples.append(groups)
    
    # Now for local search:
    # State: set of active triples (by index)
    # For each 6-set i and each group g (0..14), maintain count[i][g] = number of active triples in group g
    # A configuration is "saturated" if count[i][g] == 8 for any i, g
    
    # For each triple t, precompute which (six_set, group) pairs it belongs to
    # triple_to_conflicts[t] = list of (six_set_idx, group_idx)
    
    triple_to_conflicts = [[] for _ in range(num_triples)]
    for si, groups in enumerate(six_set_triples):
        for gi, group in enumerate(groups):
            for t in group:
                triple_to_conflicts[t].append((si, gi))
    
    # Initialize with a greedy construction
    active = [False] * num_triples
    count_6s = [[0]*15 for _ in range(len(six_sets))]
    
    # Greedy: add triples in random order, skip if it would saturate any group
    random.seed(42)
    order = list(range(num_triples))
    random.shuffle(order)
    
    for t in order:
        # Check if adding t would cause any group to reach 8
        can_add = True
        for si, gi in triple_to_conflicts[t]:
            if count_6s[si][gi] == 7:
                can_add = False
                break
        if can_add:
            active[t] = True
            for si, gi in triple_to_conflicts[t]:
                count_6s[si][gi] += 1
    
    best_count = sum(active)
    best_active = active[:]
    
    # Local search: try to add inactive triples, or swap
    while time.time() < deadline:
        improved = False
        # Try adding any inactive triple
        for t in range(num_triples):
            if active[t]:
                continue
            can_add = True
            for si, gi in triple_to_conflicts[t]:
                if count_6s[si][gi] == 7:
                    can_add = False
                    break
            if can_add:
                active[t] = True
                for si, gi in triple_to_conflicts[t]:
                    count_6s[si][gi] += 1
                improved = True
                if sum(active) > best_count:
                    best_count = sum(active)
                    best_active = active[:]
        
        if not improved:
            # Try swaps: remove one active, add one inactive
            # Sample a few swaps
            for _ in range(100):
                if time.time() >= deadline:
                    break
                t_remove = random.choice([i for i in range(num_triples) if active[i]])
                t_add = random.choice([i for i in range(num_triples) if not active[i]])
                
                # Check if removing t_remove and adding t_add keeps validity
                # First check t_add alone (after removing t_remove)
                # Simulate: decrement counts for t_remove, increment for t_add
                # Check no group reaches 8
                
                temp_counts = {}
                valid = True
                for si, gi in triple_to_conflicts[t_remove]:
                    temp_counts[(si, gi)] = temp_counts.get((si, gi), 0) - 1
                for si, gi in triple_to_conflicts[t_add]:
                    temp_counts[(si, gi)] = temp_counts.get((si, gi), 0) + 1
                    if count_6s[si][gi] + temp_counts[(si, gi)] >= 8:
                        valid = False
                        break
                
                if valid:
                    # Apply the swap
                    for si, gi in triple_to_conflicts[t_remove]:
                        count_6s[si][gi] -= 1
                    for si, gi in triple_to_conflicts[t_add]:
                        count_6s[si][gi] += 1
                    active[t_remove] = False
                    active[t_add] = True
                    improved = True
        
        if not improved:
            break
    
    # Return best
    result = [list(all_triples[i]) for i in range(num_triples) if best_active[i]]
    return result

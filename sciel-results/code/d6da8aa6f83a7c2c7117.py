"""Max K(2,2,2)-free triple system on 16 points via greedy + local search."""
import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_all = len(all_triples)
    
    # Precompute index for O(1) lookup
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # For each pair of points, precompute all triples containing that pair
    # This helps in checking conflicts
    pair_to_triples = {}
    for i in range(N):
        for j in range(i+1, N):
            pair_to_triples[(i,j)] = []
    
    for t in all_triples:
        a, b, c = t
        pair_to_triples[(a,b)].append(t)
        pair_to_triples[(a,c)].append(t)
        pair_to_triples[(b,c)].append(t)
    
    # A K(2,2,2) is formed by 3 disjoint pairs (a,b), (c,d), (e,f)
    # The 8 transversal triples are: (a,c,e), (a,c,f), (a,d,e), (a,d,f),
    #                                (b,c,e), (b,c,f), (b,d,e), (b,d,f)
    # We need to ensure NOT all 8 are present.
    
    # Precompute all possible K(2,2,2) configurations
    # A K(2,2,2) is defined by 3 disjoint pairs.
    # Number of ways to choose 3 disjoint pairs from 16 points:
    # Choose 6 points, then partition into 3 pairs: C(16,6) * 15 = 8008 * 15 = 120120
    # For each such set of 3 pairs, the 8 transversal triples are determined.
    
    # Let's precompute for each triple, which K(2,2,2)s it participates in
    # Actually, let's think differently. When we add a triple, we need to check
    # if it completes any K(2,2,2).
    
    # Alternative approach: for each triple t = (a,b,c), a K(2,2,2) containing t
    # has t as one of the 8 transversal triples. So t uses one point from each pair.
    # The 3 pairs are (a, a'), (b, b'), (c, c') where a' != a, b' != b, c' != c,
    # and {a,a',b,b',c,c'} are all distinct.
    # The other 7 triples in the K(2,2,2) are the other combinations.
    
    # For efficiency, let's use a different strategy:
    # Maintain a set of present triples.
    # When checking if adding triple t would create a K(2,2,2):
    # t = (a,b,c). For each choice of a' in remaining points (not a,b,c),
    # b' in remaining (not a,b,c,a'), c' in remaining (not a,b,c,a',b'):
    # Check if all 8 transversal triples of pairs (a,a'),(b,b'),(c,c') are present or being added.
    
    # This is expensive. Let's use a smarter approach.
    
    # Actually, let's precompute for each triple, the list of "potential conflicts"
    # A conflict for triple t=(a,b,c) is a set of 7 other triples such that
    # if all 8 are present, we have a K(2,2,2).
    
    # For t=(a,b,c), the K(2,2,2)s it can be part of:
    # Choose a' from {0..15}\{a,b,c}, b' from {0..15}\{a,b,c,a'}, c' from {0..15}\{a,b,c,a',b'}
    # The 8 triples are: (a,b,c), (a,b,c'), (a,b',c), (a,b',c'), (a',b,c), (a',b,c'), (a',b',c), (a',b',c')
    
    # Precompute for each triple, the list of (a',b',c') triples that form a K(2,2,2) with it
    triple_conflicts = [[] for _ in range(num_all)]
    
    for idx, t in enumerate(all_triples):
        a, b, c = t
        remaining = [x for x in range(N) if x != a and x != b and x != c]
        for a_prime in remaining:
            for b_prime in remaining:
                if b_prime == a_prime:
                    continue
                for c_prime in remaining:
                    if c_prime == a_prime or c_prime == b_prime:
                        continue
                    # Check all 8 triples
                    eight = [
                        (a, b, c),
                        (a, b, c_prime),
                        (a, b_prime, c),
                        (a, b_prime, c_prime),
                        (a_prime, b, c),
                        (a_prime, b, c_prime),
                        (a_prime, b_prime, c),
                        (a_prime, b_prime, c_prime)
                    ]
                    # All must be valid triples (distinct elements)
                    valid = True
                    for e in eight:
                        if len(set(e)) != 3:
                            valid = False
                            break
                    if not valid:
                        continue
                    # Store the other 7 triples
                    others = []
                    for e in eight:
                        if e != t:
                            others.append(triple_to_idx[e])
                    triple_conflicts[idx].append(others)
    
    # Now greedy construction
    present = set()
    present_idx = set()
    
    def can_add(triple_idx):
        """Check if adding triple_idx would create a K(2,2,2)."""
        for others in triple_conflicts[triple_idx]:
            # Check if all others are already present
            all_present = True
            for o in others:
                if o not in present_idx:
                    all_present = False
                    break
            if all_present:
                return False
        return True
    
    # Greedy: add triples in a good order
    # Sort by some heuristic - maybe by degree or random
    order = list(range(num_all))
    random.seed(42)
    random.shuffle(order)
    
    for idx in order:
        if can_add(idx):
            present_idx.add(idx)
            present.add(all_triples[idx])
    
    # Local search: try to add more triples by removing some
    def count_triples():
        return len(present_idx)
    
    best_count = count_triples()
    
    # Local search with swaps: remove 1, try to add 2
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each triple and see if we can add more
        for remove_idx in list(present_idx):
            if time.time() > deadline:
                break
            # Temporarily remove
            present_idx.discard(remove_idx)
            present.discard(all_triples[remove_idx])
            
            # Try to add triples that were previously blocked
            added = []
            for add_idx in order:
                if add_idx in present_idx:
                    continue
                if can_add(add_idx):
                    present_idx.add(add_idx)
                    present.add(all_triples[add_idx])
                    added.append(add_idx)
            
            if len(added) > 1:  # Net gain
                improved = True
                if count_triples() > best_count:
                    best_count = count_triples()
            else:
                # Revert
                present_idx.discard(remove_idx)
                present.add(all_triples[remove_idx])
                for a in added:
                    present_idx.discard(a)
                    present.discard(all_triples[a])
    
    # Try random restarts if time allows
    restarts = 0
    while time.time() < deadline - 5 and restarts < 5:
        restarts += 1
        # Reset
        present_idx.clear()
        present.clear()
        
        # Different random seed
        seed = restarts * 1000 + 42
        random.seed(seed)
        order = list(range(num_all))
        random.shuffle(order)
        
        for idx in order:
            if can_add(idx):
                present_idx.add(idx)
                present.add(all_triples[idx])
        
        # Quick local search
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            improved = False
            for remove_idx in list(present_idx):
                if time.time() > deadline - 5:
                    break
                present_idx.discard(remove_idx)
                present.discard(all_triples[remove_idx])
                
                added = []
                for add_idx in order:
                    if add_idx in present_idx:
                        continue
                    if can_add(add_idx):
                        present_idx.add(add_idx)
                        present.add(all_triples[add_idx])
                        added.append(add_idx)
                
                if len(added) > 1:
                    improved = True
                    if count_triples() > best_count:
                        best_count = count_triples()
                else:
                    present_idx.discard(remove_idx)
                    present.add(all_triples[remove_idx])
                    for a in added:
                        present_idx.discard(a)
                        present.discard(all_triples[a])
                if not improved:
                    break
    
    return [list(t) for t in present]


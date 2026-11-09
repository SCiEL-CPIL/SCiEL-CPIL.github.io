"""Construct a linear triple system on 16 points with small independence number."""
import itertools
from functools import lru_cache

# EVOLVE-BLOCK-START

def _independence_number(triples):
    """Compute the independence number of the hypergraph."""
    n = 16
    triple_set = set(frozenset(t) for t in triples)
    
    # Optimization: Check subsets from largest size down
    # If we find an independent set of size k, we can stop if we are just checking existence
    # But we need the max size.
    # Brute force over all masks is 2^16 = 65536.
    # For each mask, check if it contains any triple.
    # We can precompute for each triple the masks it is contained in? No.
    # We can iterate subsets by size.
    
    max_indep = 0
    # Iterate over all subsets
    # To speed up, we can use the fact that if a set is independent, all its subsets are.
    # We want the largest.
    
    # Faster check:
    # For a given subset mask, it is independent if no triple is a subset of it.
    # We can precompute a bitset of "forbidden" masks?
    # Or, iterate masks and check against triples.
    
    # Given n=16, 65536 * (num_triples) might be slow if num_triples is large (~40-100).
    # 65536 * 100 = 6.5M operations. Feasible in Python? Maybe.
    # Let's optimize the check.
    
    # Convert triples to bitmasks
    triple_masks = []
    for t in triple_set:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks.append(mask)
    
    # For each subset mask, check if any triple_mask is a subset of it.
    # (triple_mask & subset_mask) == triple_mask
    
    # We can iterate subsets in decreasing size order to find max early?
    # But we need the max.
    
    # Let's just iterate all.
    for mask in range(1 << n):
        popcount = bin(mask).count('1')
        if popcount <= max_indep:
            continue
            
        is_independent = True
        for tm in triple_masks:
            if (tm & mask) == tm:
                is_independent = False
                break
        
        if is_independent:
            max_indep = popcount
    
    return max_indep

def _generate_triples():
    """
    Construct a linear triple system using an algebraic core and deterministic greedy completion.
    Core: All triples {x, y, x+y} in GF(2)^4 (40 triples).
    Completion: Greedy addition of remaining triples in deterministic order.
    """
    n = 16
    points = list(range(n))
    
    # 1. Generate Algebraic Core: {x, y, x+y}
    core_triples = []
    for x in points:
        for y in points:
            if x == y or x == 0 or y == 0:
                continue
            z = x ^ y
            if z == x or z == y or z == 0:
                continue
            t = tuple(sorted((x, y, z)))
            if t not in core_triples:
                core_triples.append(t)
                
    # Ensure linearity of core (it should be linear by property)
    # Pairs used in core
    pair_count = {}
    triples = []
    for t in core_triples:
        a, b, c = t
        pairs = [(a, b), (a, c), (b, c)]
        ok = True
        for p in pairs:
            if p in pair_count:
                ok = False
                break
        if ok:
            triples.append(t)
            for p in pairs:
                pair_count[p] = 1
                
    # 2. Generate all other triples
    all_triples = list(itertools.combinations(points, 3))
    remaining = [t for t in all_triples if t not in triples]
    
    # 3. Deterministic Greedy Completion
    # We try a few deterministic orderings and pick the best
    orderings = [
        sorted(remaining),  # Lexicographic
        sorted(remaining, reverse=True),
        sorted(remaining, key=lambda t: sum(t)),
        sorted(remaining, key=lambda t: max(t) - min(t)),
    ]
    
    best_triples = triples
    best_indep = float('inf')
    
    for order in orderings:
        current_triples = list(triples)
        current_pairs = dict(pair_count)
        
        for t in order:
            a, b, c = t
            pairs = [(a, b), (a, c), (b, c)]
            ok = True
            for p in pairs:
                if p in current_pairs:
                    ok = False
                    break
            
            if ok:
                current_triples.append(t)
                for p in pairs:
                    current_pairs[p] = 1
        
        # Evaluate
        indep = _independence_number(current_triples)
        if indep < best_indep:
            best_indep = indep
            best_triples = current_triples
        
        if best_indep <= 5:
            break
            
    return best_triples

def construct():
    triples = _generate_triples()
    # Convert to list of lists
    return [list(t) for t in triples]

# EVOLVE-BLOCK-END

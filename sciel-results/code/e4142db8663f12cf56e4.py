import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 16
    k = 4
    
    # Generate all 4-subsets as bitmasks
    # We store them as (mask, tuple_of_elements) for easy conversion back
    all_masks = []
    all_tuples = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)
        all_tuples.append(combo)
        
    total = len(all_masks)
    
    # Map mask to index (though we might just use indices)
    # We need to evaluate different orderings
    
    def run_greedy(order_indices):
        family_masks = []
        for idx in order_indices:
            if time.time() > deadline - 2:
                break
            c_mask = all_masks[idx]
            
            # Check against all pairs in family
            compatible = True
            # Optimization: If family is large, this is the bottleneck.
            # We can break early.
            for i in range(len(family_masks)):
                a_mask = family_masks[i]
                # Precompute a & c? No, b changes.
                ac = a_mask & c_mask
                for j in range(i + 1, len(family_masks)):
                    b_mask = family_masks[j]
                    ab = a_mask & b_mask
                    bc = b_mask & c_mask
                    if ab == ac == bc:
                        compatible = False
                        break
                if not compatible:
                    break
            
            if compatible:
                family_masks.append(c_mask)
                
        return family_masks

    best_family = []
    
    # 1. Standard Lexicographic Order
    # Indices are already in lex order of combinations(range(16), 4)
    order = list(range(total))
    fam = run_greedy(order)
    if len(fam) > len(best_family):
        best_family = fam[:]
        
    # 2. Reverse Lexicographic Order
    if time.time() < deadline - 5:
        order_rev = list(reversed(range(total)))
        fam = run_greedy(order_rev)
        if len(fam) > len(best_family):
            best_family = fam[:]

    # 3. Random Permutations of the Universe
    # Permuting the universe elements changes the "lexicographic" structure of the subsets.
    # We generate a random permutation of 0..15, then sort the masks based on this permutation.
    
    # To sort masks based on a permutation perm:
    # For a mask m, get its elements. Sort them by perm[element].
    # This effectively relabels the elements.
    
    for attempt in range(10):
        if time.time() > deadline - 5:
            break
        
        random.seed(42 + attempt)
        perm = list(range(n))
        random.shuffle(perm)
        
        # Create a sorting key for each mask
        # We want to order the candidates.
        # Key: tuple of (perm[x] for x in sorted_elements_of_mask)
        
        # Precompute keys for this permutation
        keys = []
        for idx in range(total):
            mask = all_masks[idx]
            elems = []
            for i in range(n):
                if mask & (1 << i):
                    elems.append(perm[i])
            # elems is already sorted by original index, but we want to sort by perm value?
            # No, the "lexicographic order" on the permuted universe means we treat the permuted values as the new labels.
            # So we sort the elements by their new labels (perm[i]).
            # The subset is defined by the set of original indices.
            # The "lexicographic" order of the subset *in the permuted space* is determined by the sorted tuple of perm[i] for i in subset.
            elems_sorted_by_perm = sorted(elems)
            keys.append((tuple(elems_sorted_by_perm), idx))
            
        # Sort indices by these keys
        keys.sort()
        order_perm = [idx for _, idx in keys]
        
        fam = run_greedy(order_perm)
        if len(fam) > len(best_family):
            best_family = fam[:]
            
    # Convert best_family masks back to lists of 1-based integers
    result = []
    for mask in best_family:
        elements = []
        for i in range(n):
            if mask & (1 << i):
                elements.append(i + 1)
        result.append(sorted(elements))
        
    return result

import itertools
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    # Points 0..15
    N = 16
    
    # Precompute all triples as bitmasks and their constituent points
    all_triples = []
    for comb in itertools.combinations(range(N), 3):
        mask = (1 << comb[0]) | (1 << comb[1]) | (1 << comb[2])
        all_triples.append((mask, comb))
        
    # Precompute all 8-subsets as bitmasks
    all_8_subsets = []
    for comb in itertools.combinations(range(N), 8):
        mask = 0
        for p in comb:
            mask |= (1 << p)
        all_8_subsets.append(mask)
        
    def count_independent_8_sets(triple_masks):
        """
        Returns the number of 8-subsets that do not contain any triple from triple_masks.
        triple_masks: list of integers (bitmasks)
        """
        count = 0
        for sub_mask in all_8_subsets:
            is_independent = True
            for t_mask in triple_masks:
                if (sub_mask & t_mask) == t_mask:
                    is_independent = False
                    break
            if is_independent:
                count += 1
        return count

    def is_linear(triple_masks, new_mask):
        """
        Check if adding new_mask to triple_masks maintains linearity.
        """
        for t_mask in triple_masks:
            if (new_mask & t_mask).bit_count() > 1:
                return False
        return True

    def generate_initial_packing():
        """
        Greedy random construction of a linear triple system.
        """
        triples = []
        triple_masks = []
        available_indices = list(range(len(all_triples)))
        random.shuffle(available_indices)
        
        for idx in available_indices:
            mask, comb = all_triples[idx]
            if is_linear(triple_masks, mask):
                triples.append(list(comb))
                triple_masks.append(mask)
                
        return triples, triple_masks

    # Initial solution
    triples, triple_masks = generate_initial_packing()
    current_count = count_independent_8_sets(triple_masks)
    best_count = current_count
    best_triples = triples[:]
    
    if current_count == 0:
        return best_triples

    # Simulated Annealing / Local Search
    # We want to minimize current_count.
    # Move: Remove a random triple, add a random compatible triple.
    
    temp = 1.0
    cooling_rate = 0.99
    
    # To speed up is_linear check, we can keep a set of pair masks?
    # No, just iterating over ~20 triples is fast enough.
    
    steps = 0
    while time.time() < deadline:
        steps += 1
        if steps % 1000 == 0:
            # Print progress if needed, but avoid I/O overhead
            pass
            
        # Pick a triple to remove
        if not triple_masks:
            break
            
        remove_idx = random.randint(0, len(triple_masks) - 1)
        removed_mask = triple_masks[remove_idx]
        
        # Try to find a replacement
        # We can pick a random triple from all_triples
        # To be more efficient, we could pick from a pool of "promising" triples, 
        # but random is fine for small N.
        
        # We want to replace the removed triple with a new one.
        # The new triple must be linear with the remaining triples.
        
        candidate_mask = None
        # Try a few random candidates to find a compatible one
        for _ in range(50):
            cand_idx = random.randint(0, len(all_triples) - 1)
            cand_mask, cand_comb = all_triples[cand_idx]
            
            # It shouldn't be the same as the one we removed (optional, but good)
            if cand_mask == removed_mask:
                continue
                
            # Check linearity with remaining triples
            # triple_masks without removed_mask
            # We can just check all, and if it conflicts with removed_mask, it doesn't matter 
            # because removed_mask is gone.
            # Wait, is_linear checks against the list. 
            # We need to check against triple_masks excluding remove_idx.
            
            compatible = True
            for i, t_mask in enumerate(triple_masks):
                if i == remove_idx:
                    continue
                if (cand_mask & t_mask).bit_count() > 1:
                    compatible = False
                    break
            
            if compatible:
                candidate_mask = cand_mask
                candidate_comb = cand_comb
                break
                
        if candidate_mask is None:
            # No replacement found, try another removal or skip
            continue
            
        # Perform the move
        # Remove old
        triple_masks[remove_idx] = candidate_mask
        # Update triples list for return
        # We need to keep the list of triples consistent with masks.
        # Let's maintain a list of triples (lists of points) and a list of masks.
        # When we swap, we update both.
        
        # Find the index in triples corresponding to remove_idx
        # Since we only swap, the index is the same.
        # But we need to know the points for the new mask.
        # We stored candidate_comb.
        
        old_comb = triples[remove_idx]
        triples[remove_idx] = list(candidate_comb)
        
        new_count = count_independent_8_sets(triple_masks)
        
        if new_count < best_count:
            best_count = new_count
            best_triples = triples[:]
            if best_count == 0:
                return best_triples
                
        # Acceptance criterion
        delta = new_count - current_count
        if delta < 0:
            current_count = new_count
        else:
            # Accept with probability exp(-delta / temp)
            # Since delta is integer >= 0
            if random.random() < pow(temp, -delta):
                current_count = new_count
            else:
                # Reject move: revert
                triple_masks[remove_idx] = removed_mask
                triples[remove_idx] = old_comb
        
        temp *= cooling_rate
        if temp < 0.01:
            temp = 0.01

    return best_triples

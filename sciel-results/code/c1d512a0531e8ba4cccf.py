import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin for 90s limit
    
    n = 12
    m = 40
    N = 1 << n  # 4096
    
    # Precompute popcount for all masks
    popcount = np.zeros(N, dtype=np.int8)
    for mask in range(1, N):
        popcount[mask] = popcount[mask >> 1] + (mask & 1)
    
    # Precompute all masks as numpy array
    all_masks = np.arange(N, dtype=np.int32)
    
    def are_comparable(a, b):
        """Check if a and b are comparable (one is subset of other)."""
        return (a & b) == a or (a & b) == b

    def score_family(masks):
        """Count comparable pairs in the family."""
        count = 0
        m = len(masks)
        for i in range(m):
            mi = masks[i]
            for j in range(i+1, m):
                mj = masks[j]
                if are_comparable(mi, mj):
                    count += 1
        return count

    def greedy_initial():
        """
        Construct a family by greedily adding the mask that maximizes
        the number of comparable pairs with the current family.
        """
        masks = []
        mask_set = set()
        
        masks.append(0)
        mask_set.add(0)
        
        while len(masks) < m:
            best_mask = -1
            best_score = -1
            for candidate in range(N):
                if candidate in mask_set:
                    continue
                
                gain = 0
                for existing in masks:
                    if are_comparable(candidate, existing):
                        gain += 1
                
                if gain > best_score:
                    best_score = gain
                    best_mask = candidate
                    
                if best_score == len(masks):
                    break
            
            if best_mask == -1:
                break
                
            masks.append(best_mask)
            mask_set.add(best_mask)
            
            if time.time() > deadline - 10:
                break

        return masks

    def chain_based_initial():
        """
        Build a family by constructing chains in the Boolean lattice.
        A chain is a sequence of sets where each is a subset of the next.
        Multiple chains give many comparable pairs within each chain.
        """
        masks = []
        mask_set = set()
        
        # Start with empty set and full set
        masks.append(0)
        mask_set.add(0)
        
        # Build chains by extending from existing sets
        while len(masks) < m:
            if time.time() > deadline - 15:
                break
            
            # Pick a random existing set to extend from
            base_idx = random.randint(0, len(masks)-1)
            base = masks[base_idx]
            
            # Find a superset of base not already in the set
            best_candidate = -1
            best_gain = -1
            
            # Try to find a superset that is comparable with many existing sets
            # Sample candidates
            for _ in range(200):
                # Add 1-3 random bits to base
                new_mask = base
                flips = random.randint(1, 3)
                bits_to_add = random.sample(range(n), flips)
                for b in bits_to_add:
                    new_mask |= (1 << b)
                
                if new_mask in mask_set:
                    continue
                if new_mask == base:
                    continue
                
                # Check it's actually a superset (it should be by construction)
                gain = 0
                for existing in masks:
                    if are_comparable(new_mask, existing):
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = new_mask
            
            if best_candidate == -1:
                # Fallback: try random
                for _ in range(50):
                    cand = random.randint(0, N-1)
                    if cand not in mask_set:
                        best_candidate = cand
                        break
            
            if best_candidate != -1:
                masks.append(best_candidate)
                mask_set.add(best_candidate)
            else:
                break
        
        return masks

    def local_search(masks, max_iterations=20000):
        """Improve the family using local search with NumPy-vectorized gains."""
        masks = list(masks)
        current_score = score_family(masks)
        best_score = current_score
        best_masks = list(masks)
        
        no_improve_count = 0
        iter_count = 0
        mask_set = set(masks)
        
        # Precompute comparability matrix for current family
        # comp[i][j] = 1 if masks[i] and masks[j] are comparable
        comp = np.zeros((m, m), dtype=np.int8)
        for i in range(m):
            for j in range(i+1, m):
                if are_comparable(masks[i], masks[j]):
                    comp[i][j] = 1
                    comp[j][i] = 1
        
        while iter_count < max_iterations:
            if time.time() > deadline - 2:
                break
            iter_count += 1
            
            # Pick a random position to modify
            idx = random.randint(0, m-1)
            old_mask = masks[idx]
            
            # Generate a new candidate mask
            r = random.random()
            if r < 0.3:
                new_mask = random.randint(0, N - 1)
            elif r < 0.6:
                new_mask = old_mask
                flips = random.randint(1, 2)
                for _ in range(flips):
                    bit = random.randint(0, n-1)
                    new_mask ^= (1 << bit)
            else:
                other_idx = random.randint(0, m-1)
                if other_idx == idx:
                    other_idx = (other_idx + 1) % m
                base = masks[other_idx]
                new_mask = base
                flips = random.randint(1, 2)
                for _ in range(flips):
                    bit = random.randint(0, n-1)
                    new_mask ^= (1 << bit)
            
            if new_mask == old_mask:
                continue
            if new_mask in mask_set:
                continue
            
            # Evaluate delta using precomputed comp
            # Old comparability: sum of comp[idx, j] for j != idx
            old_comp_count = int(np.sum(comp[idx]))
            
            # New comparability: check new_mask against all other masks
            new_comp_count = 0
            for j in range(m):
                if j == idx:
                    continue
                if are_comparable(new_mask, masks[j]):
                    new_comp_count += 1
            
            delta = new_comp_count - old_comp_count
            
            if delta > 0:
                mask_set.remove(old_mask)
                masks[idx] = new_mask
                mask_set.add(new_mask)
                current_score += delta
                
                # Update comp matrix
                for j in range(m):
                    if j == idx:
                        continue
                    comp[idx][j] = 1 if are_comparable(new_mask, masks[j]) else 0
                    comp[j][idx] = comp[idx][j]
                
                if current_score > best_score:
                    best_score = current_score
                    best_masks = list(masks)
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 1000:
                    no_improve_count = 0
                    # Perturb
                    for _ in range(random.randint(1, 5)):
                        idx2 = random.randint(0, m-1)
                        new_mask = random.randint(0, N - 1)
                        if new_mask not in mask_set:
                            mask_set.remove(masks[idx2])
                            masks[idx2] = new_mask
                            mask_set.add(new_mask)
                    current_score = score_family(masks)
                    # Rebuild comp matrix
                    for i in range(m):
                        for j in range(i+1, m):
                            comp[i][j] = 1 if are_comparable(masks[i], masks[j]) else 0
                            comp[j][i] = comp[i][j]
        
        return best_masks, best_score

    def local_search_vectorized(masks, max_iterations=20000):
        """
        Local search with NumPy-vectorized gain calculation.
        For each candidate, compute comparability with all family members at once.
        """
        masks = list(masks)
        current_score = score_family(masks)
        best_score = current_score
        best_masks = list(masks)
        
        no_improve_count = 0
        iter_count = 0
        mask_set = set(masks)
        
        # Convert to numpy for vectorized operations
        masks_arr = np.array(masks, dtype=np.int32)
        
        while iter_count < max_iterations:
            if time.time() > deadline - 2:
                break
            iter_count += 1
            
            # Pick a random position to modify
            idx = random.randint(0, m-1)
            old_mask = int(masks_arr[idx])
            
            # Generate a new candidate mask
            r = random.random()
            if r < 0.3:
                new_mask = random.randint(0, N - 1)
            elif r < 0.6:
                new_mask = old_mask
                flips = random.randint(1, 2)
                for _ in range(flips):
                    bit = random.randint(0, n-1)
                    new_mask ^= (1 << bit)
            else:
                other_idx = random.randint(0, m-1)
                if other_idx == idx:
                    other_idx = (other_idx + 1) % m
                base = int(masks_arr[other_idx])
                new_mask = base
                flips = random.randint(1, 2)
                for _ in range(flips):
                    bit = random.randint(0, n-1)
                    new_mask ^= (1 << bit)
            
            if new_mask == old_mask:
                continue
            if new_mask in mask_set:
                continue
            
            # Vectorized comparability check
            # For all j != idx, check if new_mask and masks[j] are comparable
            # comparable if (a & b) == a or (a & b) == b
            other_masks = np.delete(masks_arr, idx)
            intersections = new_mask & other_masks
            # comparable if intersection == new_mask (new is subset) or intersection == other (other is subset)
            is_subset_of_other = (intersections == new_mask)
            other_is_subset = (intersections == other_masks)
            new_comp_count = int(np.sum(is_subset_of_other | other_is_subset))
            
            # Old comparability
            old_other_masks = np.delete(masks_arr, idx)
            old_intersections = old_mask & old_other_masks
            old_is_subset = (old_intersections == old_mask)
            old_other_is_subset = (old_intersections == old_other_masks)
            old_comp_count = int(np.sum(old_is_subset | old_other_is_subset))
            
            delta = new_comp_count - old_comp_count
            
            if delta > 0:
                mask_set.remove(old_mask)
                masks_arr[idx] = new_mask
                mask_set.add(new_mask)
                current_score += delta
                
                if current_score > best_score:
                    best_score = current_score
                    best_masks = list(masks_arr)
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 1000:
                    no_improve_count = 0
                    # Perturb
                    for _ in range(random.randint(1, 5)):
                        idx2 = random.randint(0, m-1)
                        new_mask = random.randint(0, N - 1)
                        if new_mask not in mask_set:
                            mask_set.remove(int(masks_arr[idx2]))
                            masks_arr[idx2] = new_mask
                            mask_set.add(new_mask)
                    current_score = score_family(list(masks_arr))
                    # Reset no_improve after perturb
                    no_improve_count = 0
        
        return best_masks, best_score

    best_overall = None
    best_overall_score = -1
    
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        if r == 0:
            init_masks = greedy_initial()
        elif r == 1:
            init_masks = chain_based_initial()
        elif r == 2:
            init_masks = chain_based_initial()
        else:
            # Mix of greedy, chain-based, and random
            choice = random.random()
            if choice < 0.3:
                init_masks = greedy_initial()
            elif choice < 0.6:
                init_masks = chain_based_initial()
            else:
                masks = set()
                while len(masks) < m:
                    masks.add(random.randint(0, N - 1))
                init_masks = list(masks)
        
        # Use vectorized local search
        result_masks, result_score = local_search_vectorized(init_masks, max_iterations=15000)
        
        if result_score > best_overall_score:
            best_overall_score = result_score
            best_overall = result_masks
    
    if best_overall is None:
        best_overall = greedy_initial()
    
    # Convert masks to lists of integers
    result = []
    for mask in best_overall:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
    
    # Ensure exactly m distinct sets
    seen = set()
    final_result = []
    for s in result:
        t = tuple(sorted(s))
        if t not in seen:
            seen.add(t)
            final_result.append(list(s))
    
    # If we don't have enough, add random sets
    while len(final_result) < m:
        size = random.randint(0, n)
        s = sorted(random.sample(range(n), size))
        t = tuple(s)
        if t not in seen:
            seen.add(t)
            final_result.append(s)
    
    # Trim if too many (shouldn't happen)
    final_result = final_result[:m]
    
    # Final type check: ensure all elements are Python int
    for i in range(len(final_result)):
        final_result[i] = [int(x) for x in final_result[i]]
    
    return final_result

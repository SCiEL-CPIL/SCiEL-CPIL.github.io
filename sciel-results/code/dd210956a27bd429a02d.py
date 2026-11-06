import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    n = 16
    k = 4
    full_mask = (1 << n) - 1

    # Generate all 4-element subsets as masks
    all_masks = []
    for c in combinations(range(n), k):
        mask = 0
        for x in c:
            mask |= (1 << x)
        all_masks.append(mask)
    
    num_sets = len(all_masks)
    
    # Precompute indices for ALL masks (0 to 2^16 - 1)
    # This allows O(1) access to indices of any mask
    all_indices = [None] * (1 << n)
    for m in range(1 << n):
        inds = []
        temp = m
        idx = 0
        while temp:
            if temp & 1:
                inds.append(idx)
            temp >>= 1
            idx += 1
        all_indices[m] = inds

    def popcount(x):
        return bin(x).count('1')

    def get_blocked(a, b):
        """
        Returns a list of masks z such that {a, b, z} is a sunflower.
        a, b are distinct 4-sets.
        """
        k_mask = a & b
        f_mask = (a | b) ^ k_mask # Symmetric difference
        avail_mask = full_mask ^ f_mask
        need = k - popcount(k_mask)
        
        if need < 0:
            return []
            
        # We need to choose 'need' elements from avail_mask \ k_mask
        # Actually, z = k_mask | extra.
        # extra must be disjoint from f_mask (which it is, as avail_mask excludes f_mask)
        # extra must be disjoint from k_mask? 
        # z is a set. If z contains k_mask, and we add extra from avail_mask \ k_mask, 
        # then z = k_mask | extra.
        # The elements in extra must not be in k_mask.
        # avail_mask includes k_mask.
        # So we should choose from (avail_mask ^ k_mask).
        
        candidates_mask = avail_mask ^ k_mask
        
        if need == 0:
            # z = k_mask
            # But z must be distinct from a and b.
            # If k_mask == a or k_mask == b, skip.
            # Since a,b are 4-sets and k_mask = a&b, if need=0, |k_mask|=4.
            # Then a subset k_mask and b subset k_mask. Since |a|=4, a=k_mask.
            # So a=b. But a!=b. So need=0 is impossible for distinct a,b.
            return []
            
        avail_inds = all_indices[candidates_mask]
        blocked = []
        # Generate all combinations of 'need' from avail_inds
        for combo in combinations(avail_inds, need):
            z_extra = 0
            for idx in combo:
                z_extra |= (1 << idx)
            z_mask = k_mask | z_extra
            
            # z_mask must be distinct from a and b
            if z_mask != a and z_mask != b:
                blocked.append(z_mask)
                
        return blocked

    def rebuild_forbidden(selected_list):
        """
        Rebuilds the forbidden set for a given list of selected masks.
        """
        forbidden = set()
        n_sel = len(selected_list)
        for i in range(n_sel):
            a = selected_list[i]
            for j in range(i + 1, n_sel):
                b = selected_list[j]
                blocked = get_blocked(a, b)
                forbidden.update(blocked)
        return forbidden

    best_family = []
    best_size = 0
    
    rng = random.Random(42)
    
    # 1. Initial Greedy Passes
    # We run several greedy passes with different shuffles to find a good starting point.
    # Greedy is fast because we only check membership in 'forbidden'.
    
    for attempt in range(10):
        if time.time() > deadline:
            break
            
        order = list(all_masks)
        rng.shuffle(order)
        
        selected = []
        selected_set = set()
        forbidden = set()
        
        for x in order:
            if x in forbidden:
                continue
            if x in selected_set:
                continue
                
            # Add x
            selected.append(x)
            selected_set.add(x)
            
            # Update forbidden
            # For every y in selected (except x), add blocks(x, y)
            # We iterate over selected[:-1]
            for y in selected[:-1]:
                blocks = get_blocked(x, y)
                for z in blocks:
                    forbidden.add(z)
                    
        if len(selected) > best_size:
            best_size = len(selected)
            best_family = selected.copy()
            
        if best_size >= 250:
            break

    # 2. Local Search (Swaps)
    # Try to improve the best family found.
    # Strategy: Remove a random element, rebuild forbidden, try to add as many as possible.
    
    if best_size > 0 and time.time() < deadline - 5:
        current_family = best_family.copy()
        current_size = best_size
        
        # Precompute indices for all_masks for faster lookup? 
        # We have all_masks list.
        
        # We will perform a limited number of swap attempts
        max_swap_attempts = 500
        
        for _ in range(max_swap_attempts):
            if time.time() > deadline - 2:
                break
                
            if len(current_family) == 0:
                break
                
            # Pick a random element to remove
            remove_idx = rng.randrange(len(current_family))
            removed_mask = current_family[remove_idx]
            
            # Remove it
            current_family.pop(remove_idx)
            
            # Rebuild forbidden
            new_forbidden = rebuild_forbidden(current_family)
            
            # Try to add sets that are not in current_family and not in new_forbidden
            # We can iterate through all_masks or just the ones that were previously blocked?
            # Iterating through all_masks is O(1820). 1820 * 500 = 910,000 checks. Fast.
            
            added_any = False
            new_family = current_family.copy()
            new_family_set = set(new_family)
            
            for m in all_masks:
                if m in new_family_set:
                    continue
                if m in new_forbidden:
                    continue
                    
                # We can add m
                new_family.append(m)
                new_family_set.add(m)
                # Update new_forbidden with blocks involving m
                for y in new_family[:-1]:
                    blocks = get_blocked(m, y)
                    new_forbidden.update(blocks)
                    
                added_any = True
                
            if added_any and len(new_family) > current_size:
                # Improvement found
                current_family = new_family
                current_size = len(new_family)
                best_family = current_family.copy()
                best_size = current_size
                
                if best_size >= 250:
                    break
            else:
                # No improvement, revert
                current_family.append(removed_mask)
                # We don't strictly need to sort, but keeping it somewhat ordered helps? 
                # No, set logic handles it.
                
    # Convert back to sorted lists
    result = []
    for m in best_family:
        s = []
        temp = m
        idx = 0
        while temp:
            if temp & 1:
                s.append(idx + 1)
            temp >>= 1
            idx += 1
        result.append(s)
        
    return result

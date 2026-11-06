import random
import time
from itertools import combinations

def construct():
    n = 16
    k = 4
    
    # Generate all k-subsets as bitmasks
    all_masks = []
    mask_to_idx = {}
    for comb in combinations(range(n), k):
        mask = 0
        for i in comb:
            mask |= (1 << i)
        all_masks.append(mask)
        mask_to_idx[mask] = len(all_masks) - 1
        
    N = len(all_masks)
    
    # Precompute bad sets for every pair (i, j)
    # bad_pairs[i][j] is a bitmask of indices k such that {S_i, S_j, S_k} is a sunflower
    # This is memory intensive: 1820 * 1820 bitmasks.
    # 1820 * 1820 * (1820/8) bytes ~ 360 MB. 
    # Let's try to compute on the fly or use a more memory efficient structure if needed.
    # But 360MB might be OK? Or we can compute the bad mask for a pair on the fly.
    
    def get_bad_mask(i, j):
        """
        Returns a bitmask of all sets k such that {S_i, S_j, S_k} forms a sunflower.
        """
        s_i = all_masks[i]
        s_j = all_masks[j]
        core = s_i & s_j
        core_size = bin(core).count('1')
        
        # If core_size > 4, impossible (but k=4 so max intersection is 4)
        # If core_size == 4, s_i == s_j, but i != j, so this won't happen for distinct sets.
        
        # The union of s_i and s_j
        union = s_i | s_j
        # The universe is (1 << n) - 1
        universe = (1 << n) - 1
        # Available elements for the "petals" of k
        # k must contain core.
        # k \ core must be disjoint from (s_i \ core) and (s_j \ core).
        # This is equivalent to k \ core being a subset of universe \ union.
        available = universe ^ union
        
        # We need to choose (k - core_size) elements from 'available'
        need = k - core_size
        
        bad_mask = 0
        
        if need < 0:
            return 0
            
        # Iterate through all combinations of 'need' elements from 'available'
        # available is a bitmask. We need to extract its set bits.
        avail_list = []
        temp = available
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            avail_list.append(idx)
            temp ^= lsb
            
        if len(avail_list) < need:
            return 0
            
        for comb in combinations(avail_list, need):
            petal_mask = 0
            for idx in comb:
                petal_mask |= (1 << idx)
            full_mask = core | petal_mask
            if full_mask in mask_to_idx:
                k_idx = mask_to_idx[full_mask]
                bad_mask |= (1 << k_idx)
                
        return bad_mask

    # Optimization: Precompute bad masks for all pairs?
    # 1820^2 / 2 = 1.6M pairs.
    # Each calculation involves iterating combinations.
    # Average need is around 2-3. Combinations are small.
    # It might take a few seconds to precompute.
    # Let's precompute to speed up the search.
    
    # To save memory, we can store bad masks in a dictionary or list of lists of integers (if sparse)?
    # But the search needs to OR them quickly. Bitmasks are fast.
    # Let's use a list of lists, where bad_pairs[i][j] is the bitmask.
    # But 360MB is risky.
    # Let's compute on the fly in the search, but cache results?
    # Or, notice that the search is backtracking. We can just compute it when needed.
    
    # Let's try a greedy approach first with this strong pruning logic?
    # No, the directive says "new approach". Backtracking is new.
    
    # We will do a randomized backtracking search.
    # To avoid timeout, we limit the search depth or time.
    
    best_family = []
    start_time = time.time()
    max_time = 55.0
    
    # Heuristic ordering: Sort sets by "popularity" or random?
    # Random order is good for diversification.
    
    for attempt in range(50): # Try a few different random orderings
        if time.time() - start_time > max_time:
            break
            
        indices = list(range(N))
        random.shuffle(indices)
        
        # We will use a recursive backtracking function
        # But Python recursion limit might be an issue if depth is high (max 1820).
        # Set recursion limit higher.
        import sys
        sys.setrecursionlimit(10000)
        
        def search(family, blocked_mask, next_idx):
            nonlocal best_family
            if time.time() - start_time > max_time:
                return
                
            if len(family) > len(best_family):
                best_family = family[:]
                
            # Try to add more sets
            # We iterate through the shuffled indices starting from next_idx
            # But we need to ensure we don't re-add sets we've already considered in this branch?
            # Standard backtracking: we pick a set, add it, recurse.
            # To avoid permutations, we enforce an order on the indices chosen.
            # So we only consider indices > last_chosen_index.
            
            last_idx = -1
            if family:
                last_idx = family[-1] # This is not the index in the original list, but the index in the shuffled list?
                # Actually, we should pass the position in the shuffled list to enforce order.
                
            # Let's restructure:
            # search(family_indices_in_shuffled_order, current_pos, blocked_mask)
            # family_indices_in_shuffled_order is a list of positions in the 'indices' list.
            
            # This implementation is getting complex. Let's use a simpler iterative deepening or just a greedy with local search?
            # No, let's stick to a simple recursive backtracking with a depth limit or time check.
            
            # To keep it simple and fast:
            # We try to add sets from the shuffled list.
            # If we add a set, we update blocked_mask.
            # We recurse.
            
            # We need to know which sets are "available" (not blocked and not yet used).
            # Since we enforce order by the shuffled index, we just iterate forward.
            
            for pos in range(next_idx, N):
                s_idx = indices[pos]
                
                # Check if s_idx is blocked
                if (blocked_mask >> s_idx) & 1:
                    continue
                    
                # Try adding s_idx
                new_family = family + [s_idx]
                
                # Calculate new blocked mask
                new_blocked = blocked_mask
                for old_idx in family:
                    # Compute bad mask for pair (s_idx, old_idx)
                    bm = get_bad_mask(s_idx, old_idx)
                    new_blocked |= bm
                    
                # Recurse
                search(new_family, new_blocked, pos + 1)
                
                if time.time() - start_time > max_time:
                    return

        # Start search
        # To improve speed, we can prune if len(family) + (N - next_idx) <= len(best_family)
        # But N is 1820, so this bound is weak.
        
        search([], 0, 0)
        
    # Convert best_family (indices) to lists
    result = []
    for idx in best_family:
        mask = all_masks[idx]
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i + 1)
        result.append(sorted(subset))
        
    return result

"""Larger 4-uniform sunflower-free family via faster conflict checks + smarter local search."""

def construct():
    from itertools import combinations
    import time
    import numpy as np
    from collections import defaultdict
    
    deadline = time.time() + 82  # safety margin
    
    # All 4-subsets of [1,16]
    all_sets = list(combinations(range(1, 17), 4))
    n_total = len(all_sets)  # 1820
    
    def to_mask(s):
        m = 0
        for x in s:
            m |= (1 << (x - 1))
        return m
    
    masks = np.array([to_mask(s) for s in all_sets], dtype=np.int64)
    
    # Precompute: for each mask, the 6 pairwise-intersection masks (or rather, the intersection with each other mask is dynamic)
    # Instead, precompute popcount for all possible 16-bit masks? Too many.
    # Precompute popcount for 4-bit chunks.
    pop4 = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)
    
    def popcount(x):
        if x == 0:
            return 0
        return int(pop4[x & 0xff] + pop4[(x >> 8) & 0xff])
    
    # For a given candidate c and chosen set, we need to check:
    # Does there exist a pair i,j in chosen such that:
    #   (chosen[i] & chosen[j]) == (chosen[i] & c) == (chosen[j] & c)
    # Let t_i = chosen[i] & c. Then we need chosen[i] & chosen[j] == t_i.
    # Note that t_i is a subset of both chosen[i] and c.
    # Also t_i has size <= 4 (since chosen[i] has size 4).
    # So t_i is one of at most C(4,0)+C(4,1)+C(4,2)+C(4,3)+C(4,4) = 16 subsets.
    # But actually t_i = chosen[i] & c, so t_i is determined by chosen[i] and c.
    
    # Key insight: group chosen by their intersection with c. For each group value t,
    # check if any pair in that group has intersection exactly t.
    # For a pair i,j in group t: chosen[i] & chosen[j] == t.
    # Since t = chosen[i] & c = chosen[j] & c, and chosen[i] & chosen[j] is a subset of both.
    # chosen[i] & chosen[j] == t means: the parts of chosen[i] and chosen[j] outside c are disjoint.
    # Let A_i = chosen[i] \ c, A_j = chosen[j] \ c. Then chosen[i] & chosen[j] = t ∪ (A_i & A_j).
    # So we need A_i & A_j == empty, i.e., A_i and A_j are disjoint.
    # A_i has size = 4 - popcount(t). Similarly for A_j.
    
    # So for each group t with size s = popcount(t), we need to check if there exist two
    # elements in the group whose "tails" (chosen[i] \ c) are disjoint.
    # The tails are subsets of [16] \ c, each of size 4 - s.
    
    # Even faster: for a fixed c, precompute for each chosen[i] its tail = chosen[i] & ~c.
    # Then group by t = chosen[i] & c. For each group, check if any two tails are disjoint.
    # Two tails of size d are disjoint iff tail_i & tail_j == 0.
    
    def check_conflict(c_mask, chosen_masks):
        """Returns True if c_mask forms a sunflower with any pair in chosen_masks."""
        n = len(chosen_masks)
        if n < 2:
            return False
        
        # For each chosen, compute t = chosen & c and tail = chosen & ~c
        not_c = (~c_mask) & 0xFFFF  # 16-bit mask
        # Vectorized
        ts = chosen_masks & c_mask
        tails = chosen_masks & not_c
        
        # Group indices by t value
        # Use a dict mapping t -> list of tails
        groups = defaultdict(list)
        for i in range(n):
            groups[ts[i]].append(tails[i])
        
        for t, tail_list in groups.items():
            if len(tail_list) < 2:
                continue
            d = 4 - popcount(t)
            if d == 0:
                # All tails are empty, so any pair works -> conflict
                return True
            # Check if any two tails are disjoint
            # tails are d-bit (actually up to 16-bit but only d bits set)
            # For small groups, O(k^2) check
            k = len(tail_list)
            for a in range(k):
                ta = tail_list[a]
                for b in range(a + 1, k):
                    if (ta & tail_list[b]) == 0:
                        return True
        return False
    
    # Faster check using precomputed: for a given c, and a set of chosen masks,
    # we can use a different approach. But let's first try the above.
    
    # Even better: precompute for each c_mask, for each possible tail value (subset of [16]\c of size d),
    # but that's complex. Let's try a different optimization.
    
    # Alternative: for a fixed c, the condition "chosen[i] & chosen[j] == chosen[i] & c == chosen[j] & c"
    # is equivalent to: let t = chosen[i] & c. Then we need chosen[j] & c == t AND chosen[i] & chosen[j] == t.
    # The second condition means: (chosen[i] \ c) & (chosen[j] \ c) == 0.
    
    # So: for each chosen[i], compute t_i = chosen[i] & c and tail_i = chosen[i] \ c.
    # For a candidate c, we need to check: does there exist i != j such that t_i == t_j and tail_i & tail_j == 0?
    
    # This is what we do above. Let's optimize by using numpy for the groupby.
    
    def check_conflict_fast(c_mask, chosen_masks):
        n = len(chosen_masks)
        if n < 2:
            return False
        
        not_c = (~c_mask) & 0xFFFF
        ts = chosen_masks & c_mask
        tails = chosen_masks & not_c
        
        # Sort by t to group
        order = np.argsort(ts)
        ts_sorted = ts[order]
        tails_sorted = tails[order]
        
        # Find group boundaries
        changes = np.where(np.diff(ts_sorted) != 0)[0] + 1
        starts = np.concatenate([[0], changes])
        ends = np.concatenate([changes, [n]])
        
        for s, e in zip(starts, ends):
            if e - s < 2:
                continue
            t_val = ts_sorted[s]
            d = 4 - popcount(t_val)
            if d == 0:
                return True
            # Check disjointness within group [s, e)
            group_tails = tails_sorted[s:e]
            k = len(group_tails)
            # For small k, brute force
            for a in range(k):
                ta = group_tails[a]
                # Check against all b > a
                rest = group_tails[a+1:]
                if np.any((ta & rest) == 0):
                    return True
        return False
    
    # Greedy build
    def greedy_build(order, deadline):
        chosen_indices = []
        chosen_masks_list = []
        
        for idx in order:
            if time.time() > deadline:
                break
            cm = masks[idx]
            
            if len(chosen_masks_list) < 2:
                chosen_indices.append(idx)
                chosen_masks_list.append(cm)
                continue
            
            chosen_arr = np.array(chosen_masks_list, dtype=np.int64)
            if not check_conflict_fast(cm, chosen_arr):
                chosen_indices.append(idx)
                chosen_masks_list.append(cm)
        
        return chosen_indices, chosen_masks_list
    
    # Local improvement: try 1-out 2-in, 1-out 1-in, 2-out 3-in swaps
    def local_improve(chosen_indices, chosen_masks_list, deadline):
        improved = True
        chosen_set = set(chosen_indices)
        
        while improved and time.time() < deadline - 3:
            improved = False
            n_chosen = len(chosen_indices)
            
            # Try removing one element and adding multiple
            for remove_pos in range(n_chosen):
                if time.time() > deadline - 3:
                    break
                
                remaining_indices = [i for p, i in enumerate(chosen_indices) if p != remove_pos]
                remaining_masks = [m for p, m in enumerate(chosen_masks_list) if p != remove_pos]
                remaining_set = set(remaining_indices)
                
                if len(remaining_masks) < 2:
                    continue
                
                remaining_arr = np.array(remaining_masks, dtype=np.int64)
                
                # Try to add elements not in chosen
                added = []
                for idx in range(n_total):
                    if idx in chosen_set:
                        continue
                    if time.time() > deadline - 4:
                        break
                    cm = masks[idx]
                    if not check_conflict_fast(cm, remaining_arr):
                        added.append(idx)
                
                if len(added) > 1:
                    new_indices = remaining_indices + added
                    new_masks = remaining_masks + [masks[i] for i in added]
                    chosen_indices = new_indices
                    chosen_masks_list = new_masks
                    chosen_set = set(new_indices)
                    improved = True
                    break
            
            # Try 2-out 3-in
            if not improved and n_chosen >= 2:
                # Try removing pairs
                for r1 in range(n_chosen):
                    if time.time() > deadline - 5:
                        break
                    for r2 in range(r1 + 1, n_chosen):
                        if time.time() > deadline - 5:
                            break
                        remaining_indices = [i for p, i in enumerate(chosen_indices) if p != r1 and p != r2]
                        remaining_masks = [m for p, m in enumerate(chosen_masks_list) if p != r1 and p != r2]
                        remaining_set = set(remaining_indices)
                        
                        if len(remaining_masks) < 2:
                            continue
                        
                        remaining_arr = np.array(remaining_masks, dtype=np.int64)
                        
                        added = []
                        for idx in range(n_total):
                            if idx in chosen_set:
                                continue
                            if time.time() > deadline - 6:
                                break
                            cm = masks[idx]
                            if not check_conflict_fast(cm, remaining_arr):
                                added.append(idx)
                        
                        if len(added) >= 3:
                            new_indices = remaining_indices + added
                            new_masks = remaining_masks + [masks[i] for i in added]
                            chosen_indices = new_indices
                            chosen_masks_list = new_masks
                            chosen_set = set(new_indices)
                            improved = True
                            break
                    if improved:
                        break
        
        return chosen_indices, chosen_masks_list
    
    # Element frequency for ordering
    elem_freq = np.zeros(16, dtype=np.int64)
    for m in masks:
        for e in range(16):
            if m & (1 << e):
                elem_freq[e] += 1
    
    # Order sets by rarity of elements
    def set_rarity(i):
        m = masks[i]
        return sum(1.0 / elem_freq[e] for e in range(16) if m & (1 << e))
    
    freq_order = sorted(range(n_total), key=set_rarity)
    
    # Initial greedy
    chosen_indices, chosen_masks_list = greedy_build(freq_order, deadline)
    best_size = len(chosen_indices)
    best_indices = list(chosen_indices)
    
    # Local improvement
    if time.time() < deadline - 10:
        ci, cm = local_improve(chosen_indices, chosen_masks_list, deadline)
        if len(ci) > best_size:
            best_size = len(ci)
            best_indices = list(ci)
    
    # Simulated annealing
    def simulated_annealing(initial_indices, initial_masks, deadline):
        current_indices = list(initial_indices)
        current_masks = list(initial_masks)
        current_size = len(current_indices)
        best_indices_local = list(current_indices)
        best_size_local = current_size
        
        selected_set = set(current_indices)
        rng = np.random.default_rng(123)
        T = 3.0
        cooling = 0.995
        min_T = 0.05
        
        while time.time() < deadline - 5 and T > min_T:
            if rng.random() < 0.7:  # Try add
                unselected = [i for i in range(n_total) if i not in selected_set]
                if not unselected:
                    T *= cooling
                    continue
                idx = unselected[rng.integers(0, len(unselected))]
                cm = masks[idx]
                
                if len(current_masks) < 2 or not check_conflict_fast(cm, np.array(current_masks, dtype=np.int64)):
                    current_indices.append(idx)
                    current_masks.append(cm)
                    selected_set.add(idx)
                    current_size += 1
                    if current_size > best_size_local:
                        best_size_local = current_size
                        best_indices_local = list(current_indices)
            else:  # Try remove
                if current_size == 0:
                    T *= cooling
                    continue
                remove_pos = rng.integers(0, current_size)
                idx = current_indices[remove_pos]
                
                if rng.random() < np.exp(-1.0 / T):
                    current_indices.pop(remove_pos)
                    current_masks.pop(remove_pos)
                    selected_set.remove(idx)
                    current_size -= 1
            
            T *= cooling
        
        return best_indices_local, best_size_local
    
    if time.time() < deadline - 15:
        sa_indices, sa_size = simulated_annealing(best_indices, [masks[i] for i in best_indices], deadline)
        if sa_size > best_size:
            best_size = sa_size
            best_indices = list(sa_indices)
    
    # Random restarts
    rng = np.random.default_rng(42)
    restarts = 0
    while time.time() < deadline - 15 and restarts < 15:
        if restarts % 3 == 0:
            order = rng.permutation(n_total)
        elif restarts % 3 == 1:
            order = freq_order.copy()
            for _ in range(200):
                i, j = rng.integers(0, n_total, 2)
                order[i], order[j] = order[j], order[i]
        else:
            # Weight by rarity
            rarities = np.array([set_rarity(i) for i in range(n_total)])
            weights = rarities / rarities.sum()
            order = rng.choice(n_total, size=n_total, replace=False, p=weights)
        
        chosen_r, masks_r = greedy_build(order, deadline)
        
        if time.time() < deadline - 10 and len(chosen_r) >= best_size - 5:
            ci, cm = local_improve(chosen_r, masks_r, deadline)
            if len(ci) > len(chosen_r):
                chosen_r = ci
        
        if time.time() < deadline - 15 and len(chosen_r) >= best_size - 3:
            sa_r, sa_sz = simulated_annealing(chosen_r, masks_r, deadline)
            if sa_sz > len(chosen_r):
                chosen_r = sa_r
        
        restarts += 1
        if len(chosen_r) > best_size:
            best_size = len(chosen_r)
            best_indices = list(chosen_r)
    
    return [sorted(all_sets[i]) for i in best_indices]

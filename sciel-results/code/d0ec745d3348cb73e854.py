import numpy as np
import time
from itertools import combinations, product
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    D = 11
    
    # Generate all weight-4 supports
    supports = list(combinations(range(D), 4))
    N_sup = len(supports)  # 330
    
    # Precompute support masks (11-bit integers)
    sup_masks = [0] * N_sup
    for i, s in enumerate(supports):
        m = 0
        for idx in s:
            m |= (1 << idx)
        sup_masks[i] = m
        
    # Precompute incompatibility masks
    # incompat[i] is a bitmask of all j such that supports[i] and supports[j] are incompatible
    # Incompatible if intersection size >= 3
    incompat_masks = [0] * N_sup
    for i in range(N_sup):
        mask_i = sup_masks[i]
        # Check all j
        # Optimization: We can iterate j and check popcount
        # 330*330 = 108900 checks. Popcount is fast.
        incompat = 0
        for j in range(N_sup):
            if i == j:
                continue
            inter = mask_i & sup_masks[j]
            # Count bits
            # Python int.bit_count() is available in 3.10+
            if inter.bit_count() >= 3:
                incompat |= (1 << j)
        incompat_masks[i] = incompat

    def greedy_mis_min_degree(available_mask, rng):
        """Greedy MIS using minimum degree heuristic with bitmasks."""
        selected = 0
        avail = available_mask
        
        while avail:
            if time.time() > deadline - 10.0:
                break
            
            # Find node with minimum degree in induced subgraph
            # Degree of u in avail is popcount(avail & incompat_masks[u])
            min_deg = float('inf')
            best_u = -1
            # Iterate over bits in avail
            # To speed up, we can sample or just iterate all if N is small enough
            # 330 is small.
            u = 0
            temp_avail = avail
            while temp_avail:
                # Get lowest set bit
                lsb = temp_avail & -temp_avail
                u = lsb.bit_length() - 1
                deg = (avail & incompat_masks[u]).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    best_u = u
                temp_avail ^= lsb
            
            if best_u == -1:
                break
                
            # Add best_u to selected
            selected |= (1 << best_u)
            
            # Remove best_u and its neighbors from avail
            # Neighbors of best_u are incompat_masks[best_u]
            # We remove (1 << best_u) | incompat_masks[best_u]
            to_remove = (1 << best_u) | incompat_masks[best_u]
            avail &= ~to_remove
            
        return selected

    def local_search_add(selected_mask, rng):
        """Try to add vertices that are compatible with all current selected vertices."""
        selected = selected_mask
        # Check all nodes not in selected
        # A node x is compatible with selected if (incompat_masks[x] & selected) == 0
        # We want to add as many as possible? 
        # If we add x, it might block others.
        # Greedy add: pick one, add, repeat.
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            # Find any x not in selected such that (incompat_masks[x] & selected) == 0
            # To maximize size, maybe pick the one with fewest new conflicts?
            # For now, just pick the first one found.
            for x in range(N_sup):
                if selected & (1 << x):
                    continue
                if (incompat_masks[x] & selected) == 0:
                    selected |= (1 << x)
                    changed = True
                    break
        return selected

    def local_search_1_2_swap(selected_mask, rng):
        """Improve MIS using 1-2 swaps: remove one vertex, add two compatible vertices."""
        selected = selected_mask
        improved = True
        
        while improved and time.time() < deadline - 5.0:
            improved = False
            
            # Iterate through selected vertices
            sel_list = []
            temp = selected
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                sel_list.append(v)
                temp ^= lsb
                
            for v in sel_list:
                if time.time() > deadline - 5.0:
                    break
                
                # If we remove v, what is the new constraint?
                # Remaining selected: selected_without_v = selected ^ (1 << v)
                # We need to find two vertices u1, u2 such that:
                # 1. u1, u2 not in selected_without_v
                # 2. u1, u2 compatible with all in selected_without_v
                # 3. u1 compatible with u2
                # 4. u1, u2 != v (they are not in selected anyway, so this is covered by 1)
                
                remaining = selected ^ (1 << v)
                
                # Find candidates compatible with remaining
                # cand is valid if (incompat_masks[cand] & remaining) == 0
                candidates = []
                for x in range(N_sup):
                    if remaining & (1 << x):
                        continue
                    if x == v:
                        continue
                    if (incompat_masks[x] & remaining) == 0:
                        candidates.append(x)
                        
                # Find a pair (u1, u2) in candidates that are compatible
                n_cand = len(candidates)
                found = False
                for i in range(n_cand):
                    u1 = candidates[i]
                    for j in range(i + 1, n_cand):
                        u2 = candidates[j]
                        if (incompat_masks[u1] & (1 << u2)) == 0:
                            # Swap
                            selected ^= (1 << v) # Remove v
                            selected |= (1 << u1)
                            selected |= (1 << u2)
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if improved:
                    break # Restart outer loop
        
        return selected

    def local_search_2_2_swap(selected_mask, rng):
        """Improve MIS using 2-2 swaps."""
        selected = selected_mask
        improved = True
        
        while improved and time.time() < deadline - 5.0:
            improved = False
            sel_list = []
            temp = selected
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                sel_list.append(v)
                temp ^= lsb
                
            n_sel = len(sel_list)
            for i in range(n_sel):
                if time.time() > deadline - 5.0:
                    break
                v1 = sel_list[i]
                for j in range(i + 1, n_sel):
                    if time.time() > deadline - 5.0:
                        break
                    v2 = sel_list[j]
                    
                    remaining = selected ^ (1 << v1) ^ (1 << v2)
                    
                    candidates = []
                    for x in range(N_sup):
                        if remaining & (1 << x):
                            continue
                        if x == v1 or x == v2:
                            continue
                        if (incompat_masks[x] & remaining) == 0:
                            candidates.append(x)
                            
                    n_cand = len(candidates)
                    found = False
                    for a in range(n_cand):
                        u1 = candidates[a]
                        for b in range(a + 1, n_cand):
                            u2 = candidates[b]
                            if (incompat_masks[u1] & (1 << u2)) == 0:
                                selected ^= (1 << v1)
                                selected ^= (1 << v2)
                                selected |= (1 << u1)
                                selected |= (1 << u2)
                                improved = True
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if improved:
                    break
        
        return selected

    best_mask = 0
    best_size = 0
    
    rng = np.random.default_rng(42)
    
    # Multiple restarts
    # With bitmask speedup, we can do many more restarts
    n_restarts = 1000 
    full_mask = (1 << N_sup) - 1
    
    for r in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        if r == 0:
            iset = greedy_mis_min_degree(full_mask, rng)
        else:
            # Perturbed greedy: start with random subset removed
            # Or just run greedy on full mask, but the min-degree heuristic is deterministic?
            # No, if there are ties, we can randomize.
            # The current greedy_mis_min_degree picks the *first* min-degree node found in iteration order.
            # To make it stochastic, we should shuffle the iteration or pick randomly among min-degree nodes.
            
            # Let's modify greedy to pick randomly among min-degree nodes.
            # For now, let's just use the deterministic one and rely on the 1-2 swaps to escape local optima.
            # Or, we can randomize the order of nodes in the "find min degree" loop.
            
            iset = greedy_mis_min_degree(full_mask, rng)
            iset = local_search_add(iset, rng)
            iset = local_search_1_2_swap(iset, rng)
            iset = local_search_2_2_swap(iset, rng)
            iset = local_search_1_2_swap(iset, rng)
            
        size = iset.bit_count()
        if size > best_size:
            best_size = size
            best_mask = iset

    # Final polish
    if time.time() < deadline - 5.0:
        best_mask = local_search_add(best_mask, rng)
        best_mask = local_search_1_2_swap(best_mask, rng)
        best_mask = local_search_2_2_swap(best_mask, rng)
        best_mask = local_search_1_2_swap(best_mask, rng)
        best_mask = local_search_add(best_mask, rng)

    # Generate vectors
    vectors = []
    temp = best_mask
    while temp:
        lsb = temp & -temp
        i = lsb.bit_length() - 1
        temp ^= lsb
        sup = supports[i]
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(sup, signs):
                v[idx] = s
            vectors.append(v)
            
    if not vectors:
        vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                v = np.zeros(D, dtype=np.int8)
                v[i] = 1
                v[j] = 1
                vectors.append(v)
                
    return np.array(vectors, dtype=np.int8)

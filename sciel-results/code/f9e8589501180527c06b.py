# EVOLVE-BLOCK-START
import time
import random
import itertools

P = {'n': 12, 'm': 40}

def _count_comparable_pairs(sets):
    m = len(sets)
    cnt = 0
    for i in range(m):
        si = sets[i]
        si_set = set(si) if si else None
        for j in range(i + 1, m):
            sj = sets[j]
            if not si and not sj:
                continue
            if not si:
                cnt += 1
            elif not sj:
                cnt += 1
            else:
                if len(si) < len(sj):
                    s_j_set = set(sj)
                    if all(x in s_j_set for x in si):
                        cnt += 1
                elif len(sj) < len(si):
                    s_i_set = si_set
                    if all(x in s_i_set for x in sj):
                        cnt += 1
    return cnt

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    n = P['n']
    m = P['m']
    total_masks = 1 << n  # 4096
    
    # Precompute comparability using bitsets
    # comp[i] is a bitmask where bit j is set if i and j are comparable (i != j)
    comp_bitset = [0] * total_masks
    for i in range(total_masks):
        ci = 0
        for j in range(total_masks):
            if i == j:
                continue
            if (i & j) == i or (i & j) == j:
                ci |= (1 << j)
        comp_bitset[i] = ci
    
    # Precompute popcount for all masks
    popcount = [0] * total_masks
    for mask in range(total_masks):
        popcount[mask] = mask.bit_count()
    
    # Precompute list of masks by size
    masks_by_size = [[] for _ in range(n + 1)]
    for mask in range(total_masks):
        masks_by_size[popcount[mask]].append(mask)
    
    def score_of_selection(sel):
        """Compute score for a selection (list of m mask IDs)."""
        cnt = 0
        for i in range(m):
            for j in range(i + 1, m):
                if (sel[i] & sel[j]) == sel[i] or (sel[i] & sel[j]) == sel[j]:
                    cnt += 1
        return cnt
    
    def greedy_initial(seed_rng):
        """Greedy construction starting from a structured seed."""
        selected = []
        selected_set = set()
        
        # Start with empty set and some singletons
        selected.append(0)
        selected_set.add(0)
        
        # Add some singletons
        for _ in range(5):
            b = seed_rng.randint(0, n - 1)
            mask = 1 << b
            if mask not in selected_set:
                selected.append(mask)
                selected_set.add(mask)
        
        # Greedily add masks that maximize comparability with current selection
        while len(selected) < m:
            best_mask = -1
            best_score = -1
            for k in range(total_masks):
                if k in selected_set:
                    continue
                # Count comparable with current selected
                s = 0
                for sel in selected:
                    if (k & sel) == k or (k & sel) == sel:
                        s += 1
                if s > best_score:
                    best_score = s
                    best_mask = k
            selected.append(best_mask)
            selected_set.add(best_mask)
        
        return selected
    
    def local_search(selected, rng, time_limit):
        """Local search with 1-swap and 2-swap moves using precomputed bitsets."""
        sel = list(selected)
        sel_set = set(sel)
        current_score = score_of_selection(sel)
        best_score = current_score
        best_sel = list(sel)
        
        # Precompute for each selected position, the loss if removed
        def compute_losses(sel_list):
            losses = [0] * len(sel_list)
            for i in range(len(sel_list)):
                for j in range(len(sel_list)):
                    if i == j:
                        continue
                    if (sel_list[i] & sel_list[j]) == sel_list[i] or (sel_list[i] & sel_list[j]) == sel_list[j]:
                        losses[i] += 1
            return losses
        
        losses = compute_losses(sel)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # 1-swap moves: try all positions, find best improvement
            best_delta = 0
            best_idx = -1
            best_new = -1
            
            for idx in range(m):
                if time.time() >= time_limit:
                    break
                old = sel[idx]
                loss = losses[idx]
                
                # Try all non-selected masks
                for new in range(total_masks):
                    if new in sel_set:
                        continue
                    gain = 0
                    for j in range(m):
                        if j == idx:
                            continue
                        if (new & sel[j]) == new or (new & sel[j]) == sel[j]:
                            gain += 1
                    delta = gain - loss
                    if delta > best_delta:
                        best_delta = delta
                        best_idx = idx
                        best_new = new
            
            if best_delta > 0:
                # Apply the best move
                old = sel[best_idx]
                sel_set.remove(old)
                sel_set.add(best_new)
                sel[best_idx] = best_new
                current_score += best_delta
                losses = compute_losses(sel)
                if current_score > best_score:
                    best_score = current_score
                    best_sel = list(sel)
                improved = True
        
        # 2-swap moves
        for _ in range(30):
            if time.time() >= time_limit:
                break
            idx1 = rng.randint(0, m - 1)
            idx2 = rng.randint(0, m - 1)
            if idx1 == idx2:
                continue
            old1 = sel[idx1]
            old2 = sel[idx2]
            
            # Pick two random non-selected
            candidates = [x for x in range(total_masks) if x not in sel_set]
            if len(candidates) < 2:
                break
            new1, new2 = rng.sample(candidates, 2)
            
            # Compute delta
            loss = 0
            for j in range(m):
                if j == idx1 or j == idx2:
                    continue
                if (old1 & sel[j]) == old1 or (old1 & sel[j]) == sel[j]:
                    loss += 1
                if (old2 & sel[j]) == old2 or (old2 & sel[j]) == old2:
                    loss += 1
            if (old1 & old2) == old1 or (old1 & old2) == old2:
                loss += 1
            
            gain = 0
            for j in range(m):
                if j == idx1 or j == idx2:
                    continue
                if (new1 & sel[j]) == new1 or (new1 & sel[j]) == sel[j]:
                    gain += 1
                if (new2 & sel[j]) == new2 or (new2 & sel[j]) == sel[j]:
                    gain += 1
            if (new1 & new2) == new1 or (new1 & new2) == new2:
                gain += 1
            
            delta = gain - loss
            if delta > 0:
                sel_set.remove(old1)
                sel_set.remove(old2)
                sel_set.add(new1)
                sel_set.add(new2)
                sel[idx1] = new1
                sel[idx2] = new2
                current_score += delta
                losses = compute_losses(sel)
                if current_score > best_score:
                    best_score = current_score
                    best_sel = list(sel)
        
        return best_sel, best_score
    
    # Multiple restarts with different strategies
    best_overall_score = -1
    best_overall_sel = None
    
    rng = random.Random(42)
    
    # Strategy 1: Structured chain initialization
    structured = []
    structured.append(0)  # empty set
    for b in range(n):
        structured.append(1 << b)  # all singletons
    pairs = list(itertools.combinations(range(n), 2))
    rng.shuffle(pairs)
    for p in pairs[:27]:
        structured.append((1 << p[0]) | (1 << p[1]))
    
    if len(structured) > m:
        structured = structured[:m]
    while len(structured) < m:
        k = rng.randint(0, total_masks - 1)
        while k in structured:
            k = rng.randint(0, total_masks - 1)
        structured.append(k)
    
    s1, sc1 = local_search(structured, rng, deadline - 5)
    if sc1 > best_overall_score:
        best_overall_score = sc1
        best_overall_sel = s1
    
    # Strategy 2: Greedy initialization
    init2 = greedy_initial(rng)
    s2, sc2 = local_search(init2, rng, deadline - 5)
    if sc2 > best_overall_score:
        best_overall_score = sc2
        best_overall_sel = s2
    
    # Strategy 3: More restarts
    restart_count = 0
    while time.time() < deadline - 3 and restart_count < 15:
        init = greedy_initial(rng)
        s, sc = local_search(init, rng, deadline - 3)
        if sc > best_overall_score:
            best_overall_score = sc
            best_overall_sel = s
        restart_count += 1
    
    if best_overall_sel is None:
        best_overall_sel = list(range(m))
    
    # Convert to lists
    result = []
    for mask in best_overall_sel:
        result.append([b for b in range(n) if (mask >> b) & 1])
    
    final_result = []
    for sl in result:
        final_result.append([int(x) for x in sl])
    
    return final_result
# EVOLVE-BLOCK-END

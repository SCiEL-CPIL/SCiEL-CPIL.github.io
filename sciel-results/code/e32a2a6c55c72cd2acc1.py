import time
import random
import math

P = {'n': 12, 'm': 40}

def _count_comparable_pairs(sets):
    m = len(sets)
    cnt = 0
    for i in range(m):
        si = sets[i]
        for j in range(i + 1, m):
            sj = sets[j]
            if si and sj:
                if len(si) < len(sj):
                    is_sub = True
                    s_j_set = set(sj)
                    for x in si:
                        if x not in s_j_set:
                            is_sub = False
                            break
                    if is_sub:
                        cnt += 1
                elif len(sj) < len(si):
                    is_sub = True
                    s_i_set = set(si)
                    for x in sj:
                        if x not in s_i_set:
                            is_sub = False
                            break
                    if is_sub:
                        cnt += 1
            elif si and not sj:
                cnt += 1
            elif not si and sj:
                cnt += 1
    return cnt

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    n = P['n']
    m = P['m']
    total_masks = 1 << n
    full_mask = total_masks - 1
    
    # Precompute comparability bitsets
    # comp[i] is an integer where bit j is 1 if mask i and mask j are comparable (i != j)
    comp = [0] * total_masks
    for i in range(total_masks):
        ci = 0
        for j in range(total_masks):
            if i != j and ((i & j) == i or (i & j) == j):
                ci |= (1 << j)
        comp[i] = ci
    
    def greedy_init(randomize_ties=False):
        """Greedy construction: add mask that maximizes comparability with current set."""
        selected = []
        selected_mask = 0
        for _ in range(m):
            best_mask = -1
            best_score = -1
            best_tie = 0
            for k in range(total_masks):
                if (selected_mask >> k) & 1:
                    continue
                score = (comp[k] & selected_mask).bit_count()
                if randomize_ties and score == best_score:
                    best_tie += 1
                    if best_tie == 1:
                        best_mask = k
                elif score > best_score:
                    best_score = score
                    best_mask = k
                    best_tie = 0
            selected.append(best_mask)
            selected_mask |= (1 << best_mask)
        return selected, selected_mask
    
    def compute_score(selected, selected_mask):
        score = 0
        for s in selected:
            score += (comp[s] & selected_mask).bit_count()
        return score // 2
    
    def hill_climb_polish(selected, selected_mask, current_score, time_limit):
        """Intensive hill climbing with 1-swap and occasional 2-swap moves."""
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1-swap pass
            for idx in range(m):
                if time.time() >= time_limit:
                    break
                old_mask = selected[idx]
                sel_without_old = selected_mask & ~(1 << old_mask)
                loss = (comp[old_mask] & sel_without_old).bit_count()
                
                best_new = -1
                best_delta = 0
                for cand in range(total_masks):
                    if (selected_mask >> cand) & 1:
                        continue
                    gain = (comp[cand] & sel_without_old).bit_count()
                    delta = gain - loss
                    if delta > best_delta:
                        best_delta = delta
                        best_new = cand
                
                if best_new != -1 and best_delta > 0:
                    selected[idx] = best_new
                    selected_mask = sel_without_old | (1 << best_new)
                    current_score += best_delta
                    improved = True
            
            # 2-swap pass (less frequent, for escaping local optima)
            if time.time() < time_limit and not improved:
                for idx1 in range(m):
                    if time.time() >= time_limit:
                        break
                    for idx2 in range(idx1 + 1, m):
                        if time.time() >= time_limit:
                            break
                        old1 = selected[idx1]
                        old2 = selected[idx2]
                        if old1 == old2:
                            continue
                        
                        sel_without_both = selected_mask & ~(1 << old1) & ~(1 << old2)
                        
                        loss1 = (comp[old1] & sel_without_both).bit_count()
                        loss2 = (comp[old2] & sel_without_both).bit_count()
                        pair_loss = 1 if ((comp[old1] >> old2) & 1) else 0
                        total_loss = loss1 + loss2 + pair_loss
                        
                        best_new1 = -1
                        best_new2 = -1
                        best_delta = 0
                        
                        # Try all pairs of non-selected masks
                        for cand1 in range(total_masks):
                            if (selected_mask >> cand1) & 1:
                                continue
                            gain1 = (comp[cand1] & sel_without_both).bit_count()
                            for cand2 in range(total_masks):
                                if (selected_mask >> cand2) & 1 or cand2 == cand1:
                                    continue
                                gain2 = (comp[cand2] & sel_without_both).bit_count()
                                pair_gain = 1 if ((comp[cand1] >> cand2) & 1) else 0
                                total_gain = gain1 + gain2 + pair_gain
                                delta = total_gain - total_loss
                                if delta > best_delta:
                                    best_delta = delta
                                    best_new1 = cand1
                                    best_new2 = cand2
                        
                        if best_new1 != -1 and best_delta > 0:
                            selected[idx1] = best_new1
                            selected[idx2] = best_new2
                            selected_mask = sel_without_both | (1 << best_new1) | (1 << best_new2)
                            current_score += best_delta
                            improved = True
                            break
                    if improved:
                        break
        
        return selected, selected_mask, current_score
    
    # Multiple greedy restarts with different random seeds
    best_score = -1
    best_selection = []
    
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            selected, selected_mask = greedy_init(randomize_ties=False)
        else:
            random.seed(restart * 12345 + 67890)
            selected, selected_mask = greedy_init(randomize_ties=True)
        
        current_score = compute_score(selected, selected_mask)
        
        # Simulated annealing phase
        temperature = 5.0
        cooling_rate = 0.995
        min_temp = 0.05
        no_improve = 0
        max_no_improve = 3000
        
        while temperature > min_temp and time.time() < deadline - 10.0:
            # 1-Swap Move
            idx = random.randint(0, m - 1)
            old_mask = selected[idx]
            sel_without_old = selected_mask & ~(1 << old_mask)
            loss = (comp[old_mask] & sel_without_old).bit_count()
            
            # Pick random new_mask not in selected
            new_mask = -1
            attempts = 0
            while new_mask == -1 and attempts < 200:
                cand = random.randint(0, total_masks - 1)
                if not ((selected_mask >> cand) & 1):
                    new_mask = cand
                attempts += 1
            
            if new_mask != -1:
                gain = (comp[new_mask] & sel_without_old).bit_count()
                delta = gain - loss
                
                if delta > 0 or random.random() < math.exp(delta / max(temperature, 0.1)):
                    selected[idx] = new_mask
                    selected_mask = sel_without_old | (1 << new_mask)
                    current_score += delta
                    no_improve = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_selection = list(selected)
                else:
                    no_improve += 1
            
            # 2-Swap Move (occasionally)
            if random.random() < 0.15 and time.time() < deadline - 10.0:
                idx1 = random.randint(0, m - 1)
                idx2 = random.randint(0, m - 1)
                if idx1 == idx2:
                    idx2 = (idx2 + 1) % m
                
                old1 = selected[idx1]
                old2 = selected[idx2]
                if old1 == old2:
                    temperature *= cooling_rate
                    no_improve += 1
                    continue
                
                sel_without_both = selected_mask & ~(1 << old1) & ~(1 << old2)
                loss1 = (comp[old1] & sel_without_both).bit_count()
                loss2 = (comp[old2] & sel_without_both).bit_count()
                pair_loss = 1 if ((comp[old1] >> old2) & 1) else 0
                total_loss = loss1 + loss2 + pair_loss
                
                new1 = -1
                new2 = -1
                attempts = 0
                while (new1 == -1 or new2 == -1) and attempts < 400:
                    cand = random.randint(0, total_masks - 1)
                    if not ((selected_mask >> cand) & 1) and cand != old1 and cand != old2:
                        if new1 == -1:
                            new1 = cand
                        elif cand != new1:
                            new2 = cand
                    attempts += 1
                
                if new1 != -1 and new2 != -1:
                    gain1 = (comp[new1] & sel_without_both).bit_count()
                    gain2 = (comp[new2] & sel_without_both).bit_count()
                    pair_gain = 1 if ((comp[new1] >> new2) & 1) else 0
                    total_gain = gain1 + gain2 + pair_gain
                    delta = total_gain - total_loss
                    
                    if delta > 0 or random.random() < math.exp(delta / max(temperature, 0.1)):
                        selected[idx1] = new1
                        selected[idx2] = new2
                        selected_mask = sel_without_both | (1 << new1) | (1 << new2)
                        current_score += delta
                        no_improve = 0
                        if current_score > best_score:
                            best_score = current_score
                            best_selection = list(selected)
                    else:
                        no_improve += 1
            
            temperature *= cooling_rate
            
            if no_improve > max_no_improve:
                break
        
        # Final hill climbing polish
        if time.time() < deadline:
            selected, selected_mask, current_score = hill_climb_polish(
                list(selected), selected_mask, current_score, deadline
            )
            if current_score > best_score:
                best_score = current_score
                best_selection = list(selected)
    
    # Construct final result
    result = []
    for mask in best_selection:
        result.append([int(b) for b in range(n) if (mask >> b) & 1])
    
    return result

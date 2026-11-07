import time
import random
import math

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Precompute comparability
    # comp[i][j] = 1 if i and j are comparable (strict subset or superset)
    comp = [[0] * total_masks for _ in range(total_masks)]
    comp_list = [[] for _ in range(total_masks)]
    
    # Optimization: Use symmetry
    for i in range(total_masks):
        for j in range(i + 1, total_masks):
            if (i & ~j) == 0 or (j & ~i) == 0:
                comp[i][j] = 1
                comp[j][i] = 1
                comp_list[i].append(j)
                comp_list[j].append(i)

    def mask_to_list(mask):
        return [b for b in range(n) if (mask >> b) & 1]

    def list_to_mask(lst):
        mask = 0
        for b in lst:
            mask |= (1 << b)
        return mask

    def greedy_init():
        family = []
        used = set()
        
        # Add maximal chain
        cur = 0
        family.append(cur)
        used.add(cur)
        for i in range(n):
            cur |= (1 << i)
            family.append(cur)
            used.add(cur)
        
        # Precompute scores
        scores = [0] * total_masks
        for f in family:
            for s in comp_list[f]:
                scores[s] += 1
                
        while len(family) < m:
            best_mask = -1
            best_gain = -1
            
            for s in range(total_masks):
                if s in used:
                    continue
                if scores[s] > best_gain:
                    best_gain = scores[s]
                    best_mask = s
            
            if best_mask == -1:
                while True:
                    s = random.randint(0, total_masks - 1)
                    if s not in used:
                        best_mask = s
                        break
            
            family.append(best_mask)
            used.add(best_mask)
            
            for s in comp_list[best_mask]:
                scores[s] += 1
                
        return family

    def evaluate(fam):
        in_fam = [False] * total_masks
        for mask in fam:
            in_fam[mask] = True
            
        sum_comp = [0] * total_masks
        for s in range(total_masks):
            sc = 0
            for x in fam:
                sc += comp[s][x]
            sum_comp[s] = sc
            
        contribs = [sum_comp[fam[i]] for i in range(m)]
        current_score = sum(contribs) // 2
        return current_score, in_fam, sum_comp, contribs

    # 1. Initialization
    best_family = greedy_init()
    best_score, best_in_fam, best_sum_comp, best_contribs = evaluate(best_family)
    
    # State for SA
    fam = best_family[:]
    in_fam = best_in_fam[:]
    sum_comp = best_sum_comp[:]
    contribs = best_contribs[:]
    current_score = best_score
    
    deadline = time.time() + 75 # Leave 15s for polish
    T = 5.0
    cooling = 0.9995
    T_min = 0.01
    
    no_improve = 0
    top_k = 5 # Reduced for speed, 2-swaps are less critical if 1-swaps are fast
    
    restarts = 0
    max_restarts = 10
    
    while time.time() < deadline:
        # 1. Find best 1-swap and collect top candidates for 2-swap
        best_1_swap_delta = -1000
        best_1_swap_i = -1
        best_1_swap_s = -1
        
        top_candidates = [[] for _ in range(m)]
        
        for i in range(m):
            old_val = fam[i]
            candidates = []
            
            for s in range(total_masks):
                if in_fam[s]:
                    continue
                val = sum_comp[s] - comp[s][old_val]
                candidates.append((val, s))
            
            candidates.sort(key=lambda x: x[0], reverse=True)
            top_candidates[i] = candidates[:top_k]
            
            if candidates:
                val, s = candidates[0]
                delta = val - contribs[i]
                if delta > best_1_swap_delta:
                    best_1_swap_delta = delta
                    best_1_swap_i = best_1_swap_i if False else i # Fix logic
                    best_1_swap_s = s

        # 2. Find best 2-swap using top candidates
        best_2_swap_delta = -1000
        best_2_swap_i1 = -1
        best_2_swap_i2 = -1
        best_2_swap_s1 = -1
        best_2_swap_s2 = -1
        
        # Only do 2-swap if it might beat 1-swap
        # To save time, we can skip 2-swap if 1-swap is very good
        if best_1_swap_delta < 2: # Heuristic: if 1-swap gain is small, try 2-swap
            for i1 in range(m):
                for i2 in range(i1 + 1, m):
                    if not top_candidates[i1] or not top_candidates[i2]:
                        continue
                    
                    old1 = fam[i1]
                    old2 = fam[i2]
                    
                    for delta1, s1 in top_candidates[i1]:
                        d1 = delta1 - contribs[i1]
                        
                        for delta2, s2 in top_candidates[i2]:
                            if s1 == s2:
                                continue
                            d2 = delta2 - contribs[i2]
                            old_int = comp[old1][old2]
                            new_int = comp[s1][s2]
                            cross = comp[s1][old2] + comp[s2][old1]
                            
                            total_delta = d1 + d2 + old_int - cross + new_int
                            
                            if total_delta > best_2_swap_delta:
                                best_2_swap_delta = total_delta
                                best_2_swap_i1 = i1
                                best_2_swap_i2 = i2
                                best_2_swap_s1 = s1
                                best_2_swap_s2 = s2

        # Decide move
        accept = False
        move_type = 0
        move_i1 = -1
        move_i2 = -1
        move_s1 = -1
        move_s2 = -1
        move_delta = 0
        
        if best_2_swap_delta > best_1_swap_delta:
            if best_2_swap_delta > 0:
                accept = True
                move_type = 2
                move_i1 = best_2_swap_i1
                move_i2 = best_2_swap_i2
                move_s1 = best_2_swap_s1
                move_s2 = best_2_swap_s2
                move_delta = best_2_swap_delta
            else:
                if random.random() < math.exp(best_2_swap_delta / T):
                    accept = True
                    move_type = 2
                    move_i1 = best_2_swap_i1
                    move_i2 = best_2_swap_i2
                    move_s1 = best_2_swap_s1
                    move_s2 = best_2_swap_s2
                    move_delta = best_2_swap_delta
        else:
            if best_1_swap_delta > 0:
                accept = True
                move_type = 1
                move_i1 = best_1_swap_i
                move_s1 = best_1_swap_s
                move_delta = best_1_swap_delta
            else:
                if random.random() < math.exp(best_1_swap_delta / T):
                    accept = True
                    move_type = 1
                    move_i1 = best_1_swap_i
                    move_s1 = best_1_swap_s
                    move_delta = best_1_swap_delta

        if accept:
            if move_type == 1:
                old_val = fam[move_i1]
                new_val = move_s1
                
                fam[move_i1] = new_val
                in_fam[old_val] = False
                in_fam[new_val] = True
                
                for s in comp_list[old_val]:
                    sum_comp[s] -= 1
                for s in comp_list[new_val]:
                    sum_comp[s] += 1
                    
                contribs[move_i1] = sum_comp[new_val]
                for k in range(m):
                    if k == move_i1:
                        continue
                    contribs[k] += comp[new_val][fam[k]] - comp[old_val][fam[k]]
                        
            elif move_type == 2:
                old1 = fam[move_i1]
                old2 = fam[move_i2]
                new1 = move_s1
                new2 = move_s2
                
                fam[move_i1] = new1
                fam[move_i2] = new2
                
                in_fam[old1] = False
                in_fam[old2] = False
                in_fam[new1] = True
                in_fam[new2] = True
                
                for s in comp_list[old1]:
                    sum_comp[s] -= 1
                for s in comp_list[old2]:
                    sum_comp[s] -= 1
                for s in comp_list[new1]:
                    sum_comp[s] += 1
                for s in comp_list[new2]:
                    sum_comp[s] += 1
                    
                contribs[move_i1] = sum_comp[new1]
                contribs[move_i2] = sum_comp[new2]
                
                for k in range(m):
                    if k == move_i1 or k == move_i2:
                        continue
                    old_c = comp[old1][fam[k]] + comp[old2][fam[k]]
                    new_c = comp[new1][fam[k]] + comp[new2][fam[k]]
                    contribs[k] += new_c - old_c

            current_score = sum(contribs) // 2
            
            if current_score > best_score:
                best_score = current_score
                best_family = fam[:]
                best_in_fam = in_fam[:]
                best_sum_comp = sum_comp[:]
                best_contribs = contribs[:]
                no_improve = 0
            else:
                no_improve += 1
                
            T *= cooling
            if T < T_min:
                T = T_min
                
            if no_improve > 100:
                fam = best_family[:]
                in_fam = best_in_fam[:]
                sum_comp = best_sum_comp[:]
                contribs = best_contribs[:]
                
                num_perturb = 5
                for _ in range(num_perturb):
                    i = random.randint(0, m - 1)
                    old_val = fam[i]
                    while True:
                        s = random.randint(0, total_masks - 1)
                        if not in_fam[s]:
                            break
                    new_val = s
                    fam[i] = new_val
                    in_fam[old_val] = False
                    in_fam[new_val] = True
                    for ss in comp_list[old_val]:
                        sum_comp[ss] -= 1
                    for ss in comp_list[new_val]:
                        sum_comp[ss] += 1
                    contribs[i] = sum_comp[new_val]
                    for k in range(m):
                        if k == i: continue
                        contribs[k] += comp[new_val][fam[k]] - comp[old_val][fam[k]]
                    
                current_score, in_fam, sum_comp, contribs = evaluate(fam)
                
                T = 2.0
                no_improve = 0
                restarts += 1
        else:
            no_improve += 1
            T *= cooling
            if T < T_min:
                T = T_min

    # Phase 2: Polish with Hill Climbing (Inspiration Strategy)
    # Use the best family found by SA
    fam = best_family[:]
    fam_set = set(fam)
    
    # Recalculate scores for all external sets
    scores = [0] * total_masks
    for s in range(total_masks):
        if s in fam_set:
            scores[s] = 0
        else:
            c = 0
            for f in fam:
                if (s & ~f) == 0 or (f & ~s) == 0:
                    c += 1
            scores[s] = c
            
    current_score = best_score
    
    improved = True
    while improved and time.time() < (deadline + 14): # Use remaining time
        improved = False
        for i in range(m):
            if time.time() > (deadline + 14):
                break
            
            old_mask = fam[i]
            
            old_comp = 0
            for j in range(m):
                if j == i: continue
                if (old_mask & ~fam[j]) == 0 or (fam[j] & ~old_mask) == 0:
                    old_comp += 1
            
            best_delta = 0
            best_repl = -1
            
            for s in range(total_masks):
                if s in fam_set:
                    continue
                comp_with_old = 1 if ((s & ~old_mask) == 0 or (old_mask & ~s) == 0) else 0
                delta = scores[s] - comp_with_old - old_comp
                if delta > best_delta:
                    best_delta = delta
                    best_repl = s
                    
            if best_repl != -1 and best_delta > 0:
                fam[i] = best_repl
                fam_set.remove(old_mask)
                fam_set.add(best_repl)
                improved = True
                current_score += best_delta
                
                # Update scores
                for s in range(total_masks):
                    if s in fam_set:
                        continue
                    # scores[s] changes by:
                    # - (1 if s comp with old_mask) + (1 if s comp with best_repl)
                    if (s & ~old_mask) == 0 or (old_mask & ~s) == 0:
                        scores[s] -= 1
                    if (s & ~best_repl) == 0 or (best_repl & ~s) == 0:
                        scores[s] += 1
                
                if current_score > best_score:
                    best_score = current_score
                    best_family = fam[:]

    # Final Sanitization
    if len(set(best_family)) < m:
        unique_masks = list(set(best_family))
        used_set = set(unique_masks)
        random.shuffle(unique_masks)
        while len(unique_masks) < m:
            s = random.randint(0, total_masks - 1)
            if s not in used_set:
                unique_masks.append(s)
                used_set.add(s)
        best_family = unique_masks[:m]
        
    result = []
    for mask in best_family:
        result.append(mask_to_list(mask))
        
    return result

if __name__ == "__main__":
    res = construct()
    assert len(res) == 40
    for r in res:
        assert isinstance(r, list)
        for x in r:
            assert isinstance(x, int)
    print("Score:", end=" ")
    comp = [[0]*4096 for _ in range(4096)]
    for i in range(4096):
        for j in range(i+1, 4096):
            if (i & ~j) == 0 or (j & ~i) == 0:
                comp[i][j] = 1
                comp[j][i] = 1
    masks = [0]*40
    for i, l in enumerate(res):
        m = 0
        for b in l:
            m |= (1<<b)
        masks[i] = m
    score = 0
    for i in range(40):
        for j in range(i+1, 40):
            score += comp[masks[i]][masks[j]]
    print(score)

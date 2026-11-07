import time
import random
import math

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Precompute comparability
    # comp[i][j] = 1 if i and j are comparable (strict subset or superset)
    # Note: i and i are not comparable in the "strict" sense for pairs, 
    # but for the graph structure, we consider edges between distinct nodes.
    # The task says "one member is a strict subset of the other".
    # So comp[i][j] should be 1 if (i & ~j) == 0 and i != j, or (j & ~i) == 0 and i != j.
    # Since we only pick distinct sets, i != j is always true for pairs in the family.
    # So we just check subset relation.
    
    comp = [[0] * total_masks for _ in range(total_masks)]
    comp_list = [[] for _ in range(total_masks)]
    
    # Optimization: Use symmetry
    for i in range(total_masks):
        for j in range(i + 1, total_masks):
            # i is subset of j OR j is subset of i
            if (i & ~j) == 0 or (j & ~i) == 0:
                comp[i][j] = 1
                comp[j][i] = 1
                comp_list[i].append(j)
                comp_list[j].append(i)

    # Helper to convert mask to list
    def mask_to_list(mask):
        return [b for b in range(n) if (mask >> b) & 1]

    # Helper to convert list to mask
    def list_to_mask(lst):
        mask = 0
        for b in lst:
            mask |= (1 << b)
        return mask

    # 1. Initialization: Greedy construction
    # Start with the empty set (subset of everything) and full set (superset of everything)
    # Then greedily add the set that maximizes comparable pairs with the current family.
    
    def greedy_init():
        family = []
        used = set()
        
        # Add empty set and full set first to anchor the structure
        family.append(0)
        used.add(0)
        family.append(total_masks - 1)
        used.add(total_masks - 1)
        
        # Precompute scores for all masks relative to current family
        # score[s] = number of sets in family comparable to s
        scores = [0] * total_masks
        scores[0] = 1 # comparable with full set
        scores[total_masks - 1] = 1 # comparable with empty set
        
        for _ in range(m - 2):
            best_mask = -1
            best_gain = -1
            
            # Find mask with max score
            # We can optimize this by only checking unused masks
            for s in range(total_masks):
                if s in used:
                    continue
                if scores[s] > best_gain:
                    best_gain = scores[s]
                    best_mask = s
            
            if best_mask == -1:
                # Fallback to random
                while True:
                    s = random.randint(0, total_masks - 1)
                    if s not in used:
                        best_mask = s
                        break
            
            family.append(best_mask)
            used.add(best_mask)
            
            # Update scores for all other masks
            # For any s, score[s] increases by 1 if comp[s][best_mask] is 1
            for s in comp_list[best_mask]:
                scores[s] += 1
                
        return family

    # 2. Local Search with Simulated Annealing
    # State: fam (list of masks), in_fam (bool array), sum_comp (array of sums), contribs (array of contribs per index)
    
    def evaluate(fam):
        # Returns score and state variables
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

    best_family = greedy_init()
    best_score, best_in_fam, best_sum_comp, best_contribs = evaluate(best_family)
    
    # State for current search
    fam = best_family[:]
    in_fam = best_in_fam[:]
    sum_comp = best_sum_comp[:]
    contribs = best_contribs[:]
    current_score = best_score
    
    deadline = time.time() + 58 # 58s to leave 2s for overhead and return
    
    T = 5.0
    cooling = 0.9995
    T_min = 0.01
    
    no_improve = 0
    top_k = 5
    
    # We will run multiple restarts if time permits
    restarts = 0
    max_restarts = 5
    
    while time.time() < deadline:
        # 1. Find best 1-swap and collect top candidates for 2-swap
        best_1_swap_delta = -1000
        best_1_swap_i = -1
        best_1_swap_s = -1
        
        top_candidates = [[] for _ in range(m)]
        
        # Pre-compute list of unused masks to avoid iterating all 4096 if possible?
        # 4096 is small enough to iterate directly in Python if optimized.
        
        for i in range(m):
            old_val = fam[i]
            candidates = []
            
            # Calculate gain for swapping fam[i] with s
            # Delta = (sum_comp[s] - comp[s][old_val]) - contribs[i]
            # We want to maximize sum_comp[s] - comp[s][old_val]
            
            max_val = -1
            best_s_for_i = -1
            second_max_val = -1
            second_best_s_for_i = -1
            # ... collect top k
            
            # For speed, we can just iterate all s not in fam
            for s in range(total_masks):
                if in_fam[s]:
                    continue
                val = sum_comp[s] - comp[s][old_val]
                
                # Insert into top k list (simple sort for k=5 is fine, or manual insert)
                candidates.append((val, s))
                
            # Sort candidates to get top K
            candidates.sort(key=lambda x: x[0], reverse=True)
            top_candidates[i] = candidates[:top_k]
            
            if candidates:
                val, s = candidates[0]
                delta = val - contribs[i]
                if delta > best_1_swap_delta:
                    best_1_swap_delta = delta
                    best_1_swap_i = i
                    best_1_swap_s = s

        # 2. Find best 2-swap using top candidates
        best_2_swap_delta = -1000
        best_2_swap_i1 = -1
        best_2_swap_i2 = -1
        best_2_swap_s1 = -1
        best_2_swap_s2 = -1
        
        # Only do 2-swap if it might beat 1-swap or if 1-swap is not great
        # 2-swap is O(m^2 * k^2) = 1600 * 25 = 40000 ops. Very fast.
        for i1 in range(m):
            for i2 in range(i1 + 1, m):
                if not top_candidates[i1] or not top_candidates[i2]:
                    continue
                
                old1 = fam[i1]
                old2 = fam[i2]
                
                for delta1, s1 in top_candidates[i1]:
                    # d1 = delta1 - contribs[i1] is the delta if we only swapped i1
                    # But in 2-swap, the interaction changes.
                    # Formula:
                    # Total Delta = (NewScore - OldScore) * 2
                    # = (Sum of new contribs - Sum of old contribs)
                    # Let's use the explicit formula derived in the current program:
                    # total_delta = d1 + d2 + old_int - cross + new_int
                    # where d1 = val1 - contribs[i1], d2 = val2 - contribs[i2]
                    # val1 = sum_comp[s1] - comp[s1][old1]
                    # val2 = sum_comp[s2] - comp[s2][old2]
                    
                    for delta2, s2 in top_candidates[i2]:
                        if s1 == s2:
                            continue
                        d1 = delta1 - contribs[i1]
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
        
        # Prefer 2-swap if it's better, otherwise 1-swap
        # Use SA acceptance for negative deltas
        
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
                
                # Update sum_comp
                # sum_comp[s] changes by +1 if comp[s][new_val] - comp[s][old_val] > 0
                # i.e., if s is comparable to new_val but not old_val, +1
                # if s is comparable to old_val but not new_val, -1
                
                # Efficient update:
                # Iterate over neighbors of old_val and new_val
                set_new = set(comp_list[new_val])
                set_old = set(comp_list[old_val])
                
                # Add new_val's neighbors
                for s in set_new:
                    if s not in set_old:
                        sum_comp[s] += 1
                # Remove old_val's neighbors
                for s in set_old:
                    if s not in set_new:
                        sum_comp[s] -= 1
                        
                # Update contribs
                # contribs[move_i1] = sum_comp[new_val]
                # For other k, contribs[k] += comp[new_val][fam[k]] - comp[old_val][fam[k]]
                
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
                
                # Update sum_comp
                # Remove contributions of old1, old2
                # Add contributions of new1, new2
                
                # We can do this by iterating over the symmetric difference of neighbor sets
                # But it's simpler to just update the relevant parts.
                # Since m is small, we can re-calculate sum_comp for all s? No, that's 4096*40.
                # Let's update incrementally.
                
                # Neighbors of old1
                n_old1 = set(comp_list[old1])
                n_old2 = set(comp_list[old2])
                n_new1 = set(comp_list[new1])
                n_new2 = set(comp_list[new2])
                
                # For each s in union of these sets, calculate change
                # Change in sum_comp[s] = (comp[s][new1] + comp[s][new2]) - (comp[s][old1] + comp[s][old2])
                
                all_relevant = n_old1 | n_old2 | n_new1 | n_new2
                for s in all_relevant:
                    old_c = comp[s][old1] + comp[s][old2]
                    new_c = comp[s][new1] + comp[s][new2]
                    sum_comp[s] += new_c - old_c
                    
                # Update contribs
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
                
            # Restart logic
            if no_improve > 200:
                # Perturb the best solution
                fam = best_family[:]
                in_fam = best_in_fam[:]
                sum_comp = best_sum_comp[:]
                contribs = best_contribs[:]
                
                # Perform 5 random swaps to kick out of local optimum
                for _ in range(5):
                    i = random.randint(0, m - 1)
                    old_val = fam[i]
                    # Pick a random unused mask
                    while True:
                        s = random.randint(0, total_masks - 1)
                        if not in_fam[s]:
                            break
                    new_val = s
                    fam[i] = new_val
                    in_fam[old_val] = False
                    in_fam[new_val] = True
                    # Update sum_comp and contribs roughly (or just re-evaluate after perturb)
                    
                # Re-evaluate state after perturbation
                current_score, in_fam, sum_comp, contribs = evaluate(fam)
                
                T = 2.0 # Reheat
                no_improve = 0
                restarts += 1
                if restarts > max_restarts:
                    # Try a completely new greedy init if we've restarted too much
                    # But greedy init is deterministic, so we need randomness there too?
                    # For now, just keep perturbing best.
                    pass
        else:
            no_improve += 1
            T *= cooling
            if T < T_min:
                T = T_min

    # Final polish: Try to do a few more 1-swaps if time is left
    # The loop above should have done it.
    
    # Final Sanitization: Ensure distinct sets
    # Check for duplicates in best_family
    if len(set(best_family)) < m:
        # Remove duplicates and fill with random unused masks
        unique_masks = list(set(best_family))
        used_set = set(unique_masks)
        # Shuffle to keep some randomness or just take as is? 
        # Let's just take the unique ones and add random others
        random.shuffle(unique_masks)
        
        while len(unique_masks) < m:
            s = random.randint(0, total_masks - 1)
            if s not in used_set:
                unique_masks.append(s)
                used_set.add(s)
        
        best_family = unique_masks[:m]
        
        # Re-evaluate score to ensure consistency (though we just return the lists, 
        # the evaluator calculates the score itself, so we just need valid lists)
        
    result = []
    for mask in best_family:
        result.append(mask_to_list(mask))
        
    return result

# Quick self-test to ensure no crashes
if __name__ == "__main__":
    res = construct()
    assert len(res) == 40
    for r in res:
        assert isinstance(r, list)
        for x in r:
            assert isinstance(x, int)
    print("Score:", end=" ")
    # Calculate score for debug
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

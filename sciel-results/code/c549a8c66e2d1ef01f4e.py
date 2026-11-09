def construct():
    from fractions import Fraction
    import time
    
    deadline = time.time() + 85  # 5s safety margin
    
    # Denominators are 2..48, index i -> denom i+2
    N = 47
    weights = [Fraction(1, i + 2) for i in range(N)]
    
    # Step 1: Greedy - add in order of largest reciprocal (smallest denom)
    selected = []
    sel_set = set()
    total = Fraction(0)
    for i in range(N):
        if total + weights[i] < 1:
            selected.append(i)
            sel_set.add(i)
            total += weights[i]
    
    best_sum = total
    best_selected = selected[:]
    best_deficit = 1 - best_sum
    
    # Helper to check validity and deficit
    def get_deficit(sum_val):
        if sum_val >= 1:
            return None
        return 1 - sum_val

    # Local Search
    # We will use Best Improvement strategy
    # Moves considered:
    # 1. Add 1 (if valid)
    # 2. Swap 1 for 1
    # 3. Swap 1 for 2
    # 4. Swap 2 for 1
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            return best_selected
        
        best_new_deficit = best_deficit
        best_move = None
        # best_move structure: ('add', [j]) or ('swap11', i, j) or ('swap12', i, [j1, j2]) or ('swap21', [i1, i2], j)
        
        # 1. Try adding 1
        # If we can add any item, the sum increases, deficit decreases.
        # So if ANY single addition is valid, it's an improvement?
        # Yes, because deficit = 1 - sum. Larger sum -> smaller deficit.
        # So we just need to find the item that brings sum closest to 1.
        # Greedy already did this? Greedy adds if valid. 
        # Wait, Greedy adds in order of index. It might have skipped a larger index that fits?
        # No, Greedy checks `if total + w < 1`. If it fits, it adds.
        # So if Greedy finished, no single addition is valid?
        # Not necessarily. Greedy is sequential. 
        # Example: w=[0.9, 0.2, 0.1]. 
        # i=0: 0+0.9 < 1, add. sum=0.9.
        # i=1: 0.9+0.2 = 1.1 >= 1, skip.
        # i=2: 0.9+0.1 = 1.0 >= 1, skip.
        # Result: {0}. Sum 0.9.
        # But {1, 2} sum 0.3. Worse.
        # What if w=[0.6, 0.5, 0.4]?
        # i=0: add 0.6. sum 0.6.
        # i=1: 0.6+0.5=1.1 skip.
        # i=2: 0.6+0.4=1.0 skip.
        # Result {0}.
        # But {1, 2} sum 0.9. Better.
        # Greedy failed here.
        # So we MUST check single additions?
        # No, single addition to {0} is not possible.
        # We need to check if there's a subset that is better.
        # The "Add 1" move in local search implies adding to the CURRENT set.
        # If current set is {0}, we can't add 1 or 2.
        # So "Add 1" move yields no improvement if Greedy is stuck.
        # However, if Greedy didn't take an item because it didn't fit, it won't fit later either (sum only increases).
        # So if Greedy is done, no single addition to the current set is valid.
        
        # 2. Try Swap 1 for 1
        # Remove i, Add j.
        # New sum = total - w[i] + w[j].
        # Must be < 1.
        # Deficit = 1 - (total - w[i] + w[j]).
        # We want to minimize this, so maximize w[j] - w[i].
        # Subject to total - w[i] + w[j] < 1.
        
        for i in selected:
            sum_without_i = total - weights[i]
            # Find best j not in selected such that sum_without_i + weights[j] < 1
            # And maximizes weights[j]
            # Since we want to minimize deficit, we want max sum.
            # So for each i, we want the largest weights[j] that fits.
            best_j = -1
            best_j_weight = Fraction(0)
            for j in range(N):
                if j in sel_set:
                    continue
                candidate_sum = sum_without_i + weights[j]
                if candidate_sum < 1:
                    if weights[j] > best_j_weight:
                        best_j_weight = weights[j]
                        best_j = j
            
            if best_j != -1:
                new_sum = sum_without_i + best_j_weight
                new_deficit = 1 - new_sum
                if new_deficit < best_new_deficit:
                    best_new_deficit = new_deficit
                    best_move = ('swap11', i, best_j)
        
        # 3. Try Swap 1 for 2
        # Remove i, Add j1, j2.
        # New sum = total - w[i] + w[j1] + w[j2].
        # Must be < 1.
        # Maximize w[j1] + w[j2].
        
        for i in selected:
            sum_without_i = total - weights[i]
            # Find best pair (j1, j2) not in selected
            # Maximize w[j1] + w[j2] subject to sum_without_i + w[j1] + w[j2] < 1
            
            # This is O(N^2) per i. Total O(N^3) ~ 47^3 ~ 100k.
            # We can optimize by pre-sorting unselected weights?
            # But we need distinct indices.
            
            best_pair_sum = Fraction(0)
            best_pair = None
            unselected = [j for j in range(N) if j not in sel_set]
            
            # Sort unselected by weight descending to break early?
            # If sum_without_i + w[j1] + w[j2] >= 1, then for any j2' with w[j2'] >= w[j2], it will also be >= 1?
            # Yes, if we iterate j1, then for j2, we can stop if w[j2] is too large?
            # Actually, we want the MAX sum < 1.
            # So we want w[j1] + w[j2] to be as large as possible but < 1 - sum_without_i.
            # Let Limit = 1 - sum_without_i.
            # We want max w[j1] + w[j2] < Limit.
            
            limit = 1 - sum_without_i
            if limit <= 0:
                continue
                
            # Sort unselected indices by weight descending
            unselected_sorted = sorted(unselected, key=lambda x: weights[x], reverse=True)
            
            # Two pointers or nested loop?
            # Nested loop is fine for 47 items.
            for k1 in range(len(unselected_sorted)):
                j1 = unselected_sorted[k1]
                w1 = weights[j1]
                if w1 >= limit:
                    continue # Even with w2=0, it's too big? No, w2 > 0. So if w1 >= limit, no j2 works.
                    # Actually if w1 >= limit, then w1 + w2 > limit. So skip.
                    # Wait, if w1 is large, we need w2 to be small.
                    # So we can't just skip. We need to check.
                    # But if w1 >= limit, then w1 + w2 > limit for any w2 > 0.
                    # So yes, skip.
                for k2 in range(k1 + 1, len(unselected_sorted)):
                    j2 = unselected_sorted[k2]
                    w2 = weights[j2]
                    pair_sum = w1 + w2
                    if pair_sum < limit:
                        if pair_sum > best_pair_sum:
                            best_pair_sum = pair_sum
                            best_pair = (j1, j2)
                    else:
                        # Since sorted descending, w2' for k2' > k2 will be smaller.
                        # So we might find a valid one later?
                        # No, we want MAX pair_sum < limit.
                        # If current pair_sum >= limit, a smaller w2 might work.
                        # So we cannot break. We must continue.
                        pass
            
            if best_pair:
                new_sum = sum_without_i + best_pair_sum
                new_deficit = 1 - new_sum
                if new_deficit < best_new_deficit:
                    best_new_deficit = new_deficit
                    best_move = ('swap12', i, best_pair)
        
        # 4. Try Swap 2 for 1
        # Remove i1, i2, Add j.
        # New sum = total - w[i1] - w[i2] + w[j].
        # Must be < 1.
        # Maximize w[j] - w[i1] - w[i2].
        # Equivalent to: For each pair (i1, i2), find best j not in selected such that
        # total - w[i1] - w[i2] + w[j] < 1.
        # Maximize w[j].
        
        for idx1 in range(len(selected)):
            i1 = selected[idx1]
            for idx2 in range(idx1 + 1, len(selected)):
                i2 = selected[idx2]
                sum_without = total - weights[i1] - weights[i2]
                limit = 1 - sum_without
                if limit <= 0:
                    continue
                
                # Find max weight j not in selected such that weights[j] < limit
                best_j = -1
                best_j_weight = Fraction(0)
                for j in range(N):
                    if j in sel_set:
                        continue
                    if weights[j] < limit:
                        if weights[j] > best_j_weight:
                            best_j_weight = weights[j]
                            best_j = j
                
                if best_j != -1:
                    new_sum = sum_without + best_j_weight
                    new_deficit = 1 - new_sum
                    if new_deficit < best_new_deficit:
                        best_new_deficit = new_deficit
                        best_move = ('swap21', (i1, i2), best_j)

        # Apply best move
        if best_move:
            improved = True
            if best_move[0] == 'swap11':
                _, i, j = best_move
                selected.remove(i)
                sel_set.remove(i)
                selected.append(j)
                sel_set.add(j)
                total = total - weights[i] + weights[j]
                
            elif best_move[0] == 'swap12':
                _, i, (j1, j2) = best_move
                selected.remove(i)
                sel_set.remove(i)
                selected.append(j1)
                selected.append(j2)
                sel_set.add(j1)
                sel_set.add(j2)
                total = total - weights[i] + weights[j1] + weights[j2]
                
            elif best_move[0] == 'swap21':
                _, (i1, i2), j = best_move
                selected.remove(i1)
                selected.remove(i2)
                sel_set.remove(i1)
                sel_set.remove(i2)
                selected.append(j)
                sel_set.add(j)
                total = total - weights[i1] - weights[i2] + weights[j]
            
            best_sum = total
            best_selected = selected[:]
            best_deficit = best_new_deficit
    
    return best_selected

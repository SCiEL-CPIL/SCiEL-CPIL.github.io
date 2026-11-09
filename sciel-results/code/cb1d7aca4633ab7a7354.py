import time
from math import gcd
from fractions import Fraction
import random
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM
    L = 1
    for d in denominators:
        L = L * d // gcd(L, d)
        
    weights = [L // d for d in denominators]
    # Sort indices by weight for potential optimizations
    # But we need original indices for return
    # Let's keep weights as is, but maybe precompute sorted indices
    
    best_sum = 0
    best_indices = None
    
    rng = random.Random(42)
    
    def local_search(selected, total):
        selected_set = set(selected)
        improved = True
        max_iter = 5000  # Increase iterations if search is faster
        
        # Precompute unselected list
        unselected = [x for x in range(n) if x not in selected_set]
        # Sort unselected by weight for binary search optimizations
        # We will maintain this list and update it
        
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            
            # 1. Try 1-add
            added = False
            # To be efficient, we can just check if any unselected item fits
            # Since we want to maximize sum, we should add the largest possible item that fits?
            # Or just any item that fits? The original code adds the first one it finds.
            # Let's find the best add (largest weight < L - total)
            limit = L - total
            best_add = -1
            best_add_w = -1
            for j in unselected:
                if weights[j] < limit and weights[j] > best_add_w:
                    best_add_w = weights[j]
                    best_add = j
            
            if best_add != -1:
                selected.append(best_add)
                selected_set.add(best_add)
                total += best_add_w
                unselected.remove(best_add)
                improved = True
                continue
            
            # 2. Try 1-swap (remove i, add j)
            # Maximize w[j] - w[i] subject to w[j] < L - total + w[i]
            best_swap_gain = 0
            best_swap_i = -1
            best_swap_j = -1
            best_swap_wj = -1
            
            # Sort selected by weight? No, just iterate.
            # We can optimize by noting that we want large w[j] and small w[i].
            # For each i, find max w[j] < L - total + w[i] in unselected.
            
            # To speed this up, we can sort unselected by weight.
            # Let's keep unselected sorted.
            # Actually, updating a sorted list on removal is O(N).
            # Given N=47, O(N^2) is fine.
            
            for i in selected:
                limit_j = L - total + weights[i]
                # Find largest w[j] < limit_j in unselected
                max_wj = -1
                max_j = -1
                for j in unselected:
                    if weights[j] < limit_j and weights[j] > max_wj:
                        max_wj = weights[j]
                        max_j = j
                
                if max_j != -1:
                    gain = max_wj - weights[i]
                    if gain > best_swap_gain:
                        best_swap_gain = gain
                        best_swap_i = i
                        best_swap_j = max_j
                        best_swap_wj = max_wj
            
            if best_swap_i != -1:
                selected.remove(best_swap_i)
                selected_set.remove(best_swap_i)
                selected.append(best_swap_j)
                selected_set.add(best_swap_j)
                unselected.remove(best_swap_j)
                unselected.append(best_swap_i)
                total += best_swap_gain
                improved = True
                continue
                
            # 3. Try 1-for-2-swap (remove i, add j, k)
            # Gain: w[j] + w[k] - w[i]
            # Constraint: w[j] + w[k] < L - total + w[i]
            
            best_12_swap_gain = 0
            best_12_i = -1
            best_12_j = -1
            best_12_k = -1
            best_12_wj = -1
            best_12_wk = -1
            
            for i in selected:
                limit_jk = L - total + weights[i]
                
                # Find best pair j, k in unselected such that w[j] + w[k] < limit_jk
                # and w[j] + w[k] is maximized.
                
                # Sort unselected by weight?
                # Let's assume unselected is small enough for O(U^2) but optimized.
                # Or sort unselected once outside the loop?
                # Let's sort unselected by weight for this step.
                # This is O(U log U).
                sorted_unsel = sorted(unselected, key=lambda x: weights[x])
                
                # Two pointers or binary search
                # We want max w[j] + w[k] < limit_jk.
                # Iterate j from largest to smallest?
                # If we iterate j, we need max k such that w[k] < limit_jk - w[j].
                # Since sorted_unsel is sorted, we can binary search for k.
                
                best_pair_sum = -1
                best_j = -1
                best_k = -1
                
                # Iterate j
                for j in sorted_unsel:
                    wj = weights[j]
                    if wj >= limit_jk:
                        continue
                    # Find max k in sorted_unsel (excluding j? No, j and k must be distinct indices, but weights can be same? No, denominators are distinct, so weights are distinct?
                    # LCM/denominator. Denominators are 2..48.
                    # L is multiple of all.
                    # Weights are L/d.
                    # Are weights distinct?
                    # L/d1 = L/d2 => d1 = d2. Yes, weights are distinct.
                    # So j and k are distinct indices implies distinct weights.
                    
                    remaining = limit_jk - wj
                    # Find largest weight in sorted_unsel < remaining
                    # Use bisect on list of weights
                    # We need a list of weights corresponding to sorted_unsel
                    # Let's create a list of (weight, index) sorted by weight.
                    
                    # This is getting complex. Let's just do O(U^2) but break early if possible?
                    # Or just accept O(U^2) as U is small (~20-30).
                    # 30^2 = 900. 47 items.
                    # 900 * 47 * 5000 iterations = 200M ops. Might be too slow in Python.
                    
                    # Let's optimize the inner loop.
                    # We can pre-sort unselected by weight once per local_search call?
                    # No, unselected changes.
                    
                    # Alternative: Just iterate pairs in unselected directly.
                    # The previous code did this.
                    # Let's try to speed it up by pre-computing sorted unselected weights.
                    
                # Actually, let's just use the simple O(U^2) but with a time check?
                # Or reduce max_iter?
                
                # Let's try a simpler optimization:
                # Iterate j in unselected.
                # For each j, iterate k in unselected where k > j? No, order doesn't matter.
                # Just check all pairs.
                
                for idx_j in range(len(unselected)):
                    j = unselected[idx_j]
                    wj = weights[j]
                    if wj >= limit_jk:
                        continue
                    for idx_k in range(idx_j + 1, len(unselected)):
                        k = unselected[idx_k]
                        wk = weights[k]
                        s = wj + wk
                        if s >= limit_jk:
                            # Since we don't have sorted unselected, we can't break easily.
                            # But if we sorted unselected by weight, we could break.
                            pass
                        else:
                            gain = s - weights[i]
                            if gain > best_12_swap_gain:
                                best_12_swap_gain = gain
                                best_12_i = i
                                best_12_j = j
                                best_12_k = k
                                best_12_wj = wj
                                best_12_wk = wk

            if best_12_i != -1:
                selected.remove(best_12_i)
                selected_set.remove(best_12_i)
                selected.append(best_12_j)
                selected_set.add(best_12_j)
                selected.append(best_12_k)
                selected_set.add(best_12_k)
                unselected.remove(best_12_j)
                unselected.remove(best_12_k)
                unselected.append(best_12_i)
                total += best_12_swap_gain
                improved = True
                continue

            # 4. Try 2-for-1-swap (remove i, k; add j)
            # Gain: w[j] - w[i] - w[k]
            # Constraint: w[j] < L - total + w[i] + w[k]
            
            best_21_swap_gain = 0
            best_21_i = -1
            best_21_k = -1
            best_21_j = -1
            best_21_wj = -1
            
            sel_list = list(selected)
            for idx_i in range(len(sel_list)):
                i = sel_list[idx_i]
                for idx_k in range(idx_i + 1, len(sel_list)):
                    k = sel_list[idx_k]
                    limit_j = L - total + weights[i] + weights[k]
                    # Find best j in unselected with w[j] < limit_j
                    best_j = -1
                    best_wj = -1
                    for j in unselected:
                        if weights[j] < limit_j and weights[j] > best_wj:
                            best_wj = weights[j]
                            best_j = j
                            
                    if best_j != -1:
                        gain = best_wj - weights[i] - weights[k]
                        if gain > best_21_swap_gain:
                            best_21_swap_gain = gain
                            best_21_i = i
                            best_21_k = k
                            best_21_j = best_j
                            best_21_wj = best_wj
            
            if best_21_i != -1:
                selected.remove(best_21_i)
                selected_set.remove(best_21_i)
                selected.remove(best_21_k)
                selected_set.remove(best_21_k)
                selected.append(best_21_j)
                selected_set.add(best_21_j)
                unselected.remove(best_21_j)
                unselected.append(best_21_i)
                unselected.append(best_21_k)
                total += best_21_swap_gain
                improved = True
                continue

        return selected, total

    # Multiple restarts
    restarts = 0
    while time.time() < deadline - 1.0:
        restarts += 1
        
        # Alternate initialization strategies
        if restarts % 3 == 0:
            # Random
            perm = list(range(n))
            rng.shuffle(perm)
            order = perm
        elif restarts % 3 == 1:
            # Sorted by weight descending
            order = sorted(range(n), key=lambda i: weights[i], reverse=True)
        else:
            # Sorted by weight ascending
            order = sorted(range(n), key=lambda i: weights[i])
            
        selected = []
        selected_set = set()
        total = 0
        
        for i in order:
            if total + weights[i] < L:
                selected.append(i)
                selected_set.add(i)
                total += weights[i]
        
        selected, total = local_search(selected, total)
        
        if total > best_sum:
            best_sum = total
            best_indices = selected[:]
            
    if not best_indices:
        best_indices = [0]
        
    return best_indices

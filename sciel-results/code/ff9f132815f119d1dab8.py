def construct():
    import time
    from fractions import Fraction
    from heapq import heappush, heappop
    
    deadline = time.time() + 80  # safety margin
    
    denominators = list(range(2, 49))  # 2..48, 47 items
    n = len(denominators)
    
    # Precompute reciprocal as Fraction
    reciprocals = [Fraction(1, d) for d in denominators]
    
    # Precompute suffix sums (max possible addition from items i..n-1)
    suffix_max = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_max[i] = suffix_max[i + 1] + reciprocals[i]
    
    # Best-first search: state = (deficit, index, selected_mask, current_sum)
    # We want to minimize deficit = 1 - current_sum, with current_sum < 1
    # Start: no items selected, sum = 0, deficit = 1
    # At each step, we can add item i or skip it
    # We explore states in order of deficit (smallest first)
    
    # Use a priority queue: (deficit, index, frozenset_of_selected, current_sum)
    # But frozenset is expensive. Use a tuple of selected indices.
    
    # Better: use (deficit, next_index, selected_tuple, current_sum)
    # But this could explode. Let me use a different approach.
    
    # Alternative: branch and bound with recursion
    # Sort by reciprocal value (larger first) to prune earlier
    
    # Let's use a different strategy: iterate over subsets using a smart enumeration
    # Focus on subsets that could get close to 1
    
    # Meet in the middle: split into two halves
    # But 47 items -> 2^23 * 2^24 which is too large for meet-in-middle
    
    # Let's use branch and bound with good ordering
    # Order items by reciprocal (larger first) so we can prune when sum >= 1
    
    # Actually, let me use a simpler but effective approach:
    # Use a priority queue of (deficit, index, current_sum, selected)
    # where index is the next item to consider
    # Pop the state with smallest deficit
    # If deficit is 0 (sum == 1), that's not allowed (must be strictly below)
    # If index == n, we have a complete solution
    
    # This is essentially Dijkstra on the state space
    
    best_deficit = Fraction(1)  # initial deficit
    best_selected = [0]  # just denominator 2, sum = 1/2, deficit = 1/2
    
    # Wait, let me reconsider. The greedy gives a specific answer.
    # Let me just do a well-optimized branch and bound.
    
    # Order: process items in decreasing order of reciprocal (i.e., increasing denominator)
    # This way, we add large fractions first, which helps prune
    
    # Actually, for minimizing deficit (getting close to 1 from below),
    # we should consider adding items that get us close to 1.
    
    # Let me use a different approach: 
    # 1. Start with the greedy solution as initial best
    # 2. Use local search / hill climbing to improve
    
    # First, get the greedy solution
    selected = []
    total = Fraction(0)
    for i, d in enumerate(denominators):
        if total + reciprocals[i] < 1:
            selected.append(i)
            total += reciprocals[i]
    
    best_deficit = 1 - total
    best_selected = list(selected)
    
    # Now try to improve via local search
    # Try swapping: remove one selected item, add one unselected item
    # Or: remove one, add two, etc.
    
    selected_set = set(selected)
    unselected = [i for i in range(n) if i not in selected_set]
    
    improved = True
    iterations = 0
    
    while improved and time.time() < deadline:
        improved = False
        iterations += 1
        
        # Try single swaps: remove i from selected, add j from unselected
        for i in list(selected):
            for j in unselected:
                new_total = total - reciprocals[i] + reciprocals[j]
                if new_total < 1:
                    new_deficit = 1 - new_total
                    if new_deficit < best_deficit:
                        best_deficit = new_deficit
                        total = new_total
                        selected_set.remove(i)
                        selected_set.add(j)
                        selected = list(selected_set)
                        unselected = [k for k in range(n) if k not in selected_set]
                        improved = True
                        break
            if improved:
                break
        
        # If no single swap helped, try 2-for-1 or 1-for-2 swaps
        if not improved:
            # Try: remove two selected, add one unselected
            sel_list = list(selected)
            for a in range(len(sel_list)):
                for b in range(a + 1, len(sel_list)):
                    i, j = sel_list[a], sel_list[b]
                    for k in unselected:
                        new_total = total - reciprocals[i] - reciprocals[j] + reciprocals[k]
                        if 0 < new_total < 1:
                            new_deficit = 1 - new_total
                            if new_deficit < best_deficit:
                                best_deficit = new_deficit
                                total = new_total
                                selected_set.discard(i)
                                selected_set.discard(j)
                                selected_set.add(k)
                                selected = list(selected_set)
                                unselected = [m for m in range(n) if m not in selected_set]
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            
            # Try: remove one selected, add two unselected
            if not improved:
                for i in selected:
                    un_sel = list(unselected)
                    for a in range(len(un_sel)):
                        for b in range(a + 1, len(un_sel)):
                            j, k = un_sel[a], un_sel[b]
                            new_total = total - reciprocals[i] + reciprocals[j] + reciprocals[k]
                            if 0 < new_total < 1:
                                new_deficit = 1 - new_total
                                if new_deficit < best_deficit:
                                    best_deficit = new_deficit
                                    total = new_total
                                    selected_set.discard(i)
                                    selected_set.add(j)
                                    selected_set.add(k)
                                    selected = list(selected_set)
                                    unselected = [m for m in range(n) if m not in selected_set]
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
    
    return best_selected

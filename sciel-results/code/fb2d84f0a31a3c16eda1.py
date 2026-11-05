"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    denominators = list(range(2, 49))  # 47 items, indices 0..46
    n = len(denominators)
    reciprocals = [Fraction(1, d) for d in denominators]
    
    # Greedy initialization: take items in order if they keep sum < 1
    def greedy():
        selected = set()
        total = Fraction(0, 1)
        for i in range(n):
            if total + reciprocals[i] < 1:
                selected.add(i)
                total += reciprocals[i]
        return selected, total
    
    # Also try greedy in reverse order
    def greedy_reverse():
        selected = set()
        total = Fraction(0, 1)
        for i in range(n - 1, -1, -1):
            if total + reciprocals[i] < 1:
                selected.add(i)
                total += reciprocals[i]
        return selected, total
    
    # Local search: try adding, removing, or swapping elements
    def local_search(selected, total, max_iters=50000):
        best_selected = set(selected)
        best_total = total
        best_deficit = 1 - best_total
        
        for _ in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            
            improved = False
            # Try adding a random unselected element
            unselected = [i for i in range(n) if i not in selected]
            if unselected:
                i = random.choice(unselected)
                new_total = total + reciprocals[i]
                if new_total < 1:
                    new_deficit = 1 - new_total
                    if new_deficit < best_deficit:
                        selected = set(selected) | {i}
                        total = new_total
                        best_selected = set(selected)
                        best_total = total
                        best_deficit = new_deficit
                        improved = True
            
            # Try removing a random selected element
            if selected:
                i = random.choice(list(selected))
                new_total = total - reciprocals[i]
                if new_total > 0:  # must be nonempty
                    new_deficit = 1 - new_total
                    if new_deficit < best_deficit:
                        selected = set(selected) - {i}
                        total = new_total
                        best_selected = set(selected)
                        best_total = total
                        best_deficit = new_deficit
                        improved = True
            
            # Try swapping: remove one selected, add one unselected
            if selected and unselected:
                i_out = random.choice(list(selected))
                i_in = random.choice(unselected)
                new_total = total - reciprocals[i_out] + reciprocals[i_in]
                if 0 < new_total < 1:
                    new_deficit = 1 - new_total
                    if new_deficit < best_deficit:
                        selected = (set(selected) - {i_out}) | {i_in}
                        total = new_total
                        best_selected = set(selected)
                        best_total = total
                        best_deficit = new_deficit
                        improved = True
            
            if not improved:
                # Try a more targeted swap: find the best single swap
                best_swap_deficit = best_deficit
                best_swap = None
                for i_out in list(selected):
                    for i_in in unselected:
                        new_total = total - reciprocals[i_out] + reciprocals[i_in]
                        if 0 < new_total < 1:
                            new_deficit = 1 - new_total
                            if new_deficit < best_swap_deficit:
                                best_swap_deficit = new_deficit
                                best_swap = (i_out, i_in)
                if best_swap:
                    i_out, i_in = best_swap
                    selected = (set(selected) - {i_out}) | {i_in}
                    total = total - reciprocals[i_out] + reciprocals[i_in]
                    best_selected = set(selected)
                    best_total = total
                    best_deficit = 1 - best_total
                    improved = True
            
            if not improved:
                # Try double swap
                best_ds_deficit = best_deficit
                best_ds = None
                sel_list = list(selected)
                unsel_list = unselected
                if len(sel_list) >= 2 and len(unsel_list) >= 2:
                    for i1 in range(len(sel_list)):
                        for i2 in range(i1 + 1, len(sel_list)):
                            for j1 in range(len(unsel_list)):
                                for j2 in range(j1 + 1, len(unsel_list)):
                                    new_total = (total - reciprocals[sel_list[i1]] - reciprocals[sel_list[i2]]
                                                  + reciprocals[unsel_list[j1]] + reciprocals[unsel_list[j2]])
                                    if 0 < new_total < 1:
                                        new_deficit = 1 - new_total
                                        if new_deficit < best_ds_deficit:
                                            best_ds_deficit = new_deficit
                                            best_ds = (sel_list[i1], sel_list[i2], unsel_list[j1], unsel_list[j2])
                    if best_ds:
                        a, b, c, d = best_ds
                        selected = (set(selected) - {a, b}) | {c, d}
                        total = total - reciprocals[a] - reciprocals[b] + reciprocals[c] + reciprocals[d]
                        best_selected = set(selected)
                        best_total = total
                        best_deficit = 1 - best_total
                        improved = True
            
            if not improved:
                break
        
        return best_selected, best_total, best_deficit
    
    # Run multiple restarts
    best_deficit = Fraction(1, 1)
    best_result = []
    
    # Start with greedy
    sel, tot = greedy()
    sel, tot, deficit = local_search(sel, tot, max_iters=20000)
    if deficit < best_deficit:
        best_deficit = deficit
        best_result = list(sel)
    
    sel, tot = greedy_reverse()
    sel, tot, deficit = local_search(sel, tot, max_iters=20000)
    if deficit < best_deficit:
        best_deficit = deficit
        best_result = list(sel)
    
    # More random restarts
    random.seed(42)
    for restart in range(20):
        if time.time() > deadline - 5.0:
            break
        
        # Random initialization: pick ~15 random elements, then adjust
        sel = set()
        total = Fraction(0, 1)
        candidates = list(range(n))
        random.shuffle(candidates)
        for i in candidates[:15]:
            if total + reciprocals[i] < 1:
                sel.add(i)
                total += reciprocals[i]
        
        if not sel:
            sel = {0}
            total = reciprocals[0]
        
        sel, tot, deficit = local_search(sel, total, max_iters=15000)
        if deficit < best_deficit:
            best_deficit = deficit
            best_result = list(sel)
    
    # Final targeted improvement: try all single-element additions/removals on best
    if best_result:
        sel = set(best_result)
        total = sum(reciprocals[i] for i in sel)
        improved = True
        while improved:
            improved = False
            unselected = [i for i in range(n) if i not in sel]
            # Try all additions
            for i in unselected:
                new_total = total + reciprocals[i]
                if new_total < 1:
                    new_deficit = 1 - new_total
                    if new_deficit < best_deficit:
                        sel = sel | {i}
                        total = new_total
                        best_deficit = new_deficit
                        best_result = list(sel)
                        improved = True
            # Try all removals
            for i in list(sel):
                if len(sel) > 1:
                    new_total = total - reciprocals[i]
                    if new_total > 0:
                        new_deficit = 1 - new_total
                        if new_deficit < best_deficit:
                            sel = sel - {i}
                            total = new_total
                            best_deficit = new_deficit
                            best_result = list(sel)
                            improved = True
            # Try all swaps
            if not improved:
                unselected = [i for i in range(n) if i not in sel]
                for i_out in list(sel):
                    for i_in in unselected:
                        new_total = total - reciprocals[i_out] + reciprocals[i_in]
                        if 0 < new_total < 1:
                            new_deficit = 1 - new_total
                            if new_deficit < best_deficit:
                                sel = (sel - {i_out}) | {i_in}
                                total = new_total
                                best_deficit = new_deficit
                                best_result = list(sel)
                                improved = True
    
    if not best_result:
        best_result = [0]
    
    return best_result
# EVOLVE-BLOCK-END

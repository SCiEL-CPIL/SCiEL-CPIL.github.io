"""Rewritten program for closest strict underpacking of unit fractions."""
# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin: 50s limit to avoid 60s kill
    
    denominators = list(range(2, 49))
    n = len(denominators)
    inv = [Fraction(1, d) for d in denominators]
    inv_float = [1.0 / d for d in denominators]
    
    # 1. Greedy baseline: pick in order while sum < 1
    selected = set()
    total = Fraction(0, 1)
    for i in range(n):
        if total + inv[i] < 1:
            selected.add(i)
            total += inv[i]
    
    best_selected = set(selected)
    best_total = total
    best_deficit = 1 - total
    
    # Helper to check if a set is valid and update best
    def try_update(sel, tot):
        nonlocal best_selected, best_total, best_deficit
        if tot < 1 and tot > best_total:
            best_total = tot
            best_selected = set(sel)
            best_deficit = 1 - tot
            return True
        return False
    
    # Fast local search using only single moves (insert, delete, swap)
    # This is O(N^2) per pass, which is fast enough for many iterations.
    def fast_local_search(start_sel, start_tot, time_limit):
        nonlocal best_selected, best_total
        selected = set(start_sel)
        total = start_tot
        
        # Check time before starting
        if time.time() >= time_limit:
            return selected, total
            
        while time.time() < time_limit:
            improved = False
            
            # 1. Try inserting any unselected element
            for i in range(n):
                if i not in selected:
                    # Float check first
                    if total + inv[i] < 1:
                        new_total = total + inv[i]
                        if new_total > best_total:
                            if try_update(selected | {i}, new_total):
                                selected = best_selected
                                total = best_total
                                improved = True
            
            # 2. Try swapping any selected with any unselected
            sel_list = list(selected)
            for i in sel_list:
                if time.time() >= time_limit:
                    return selected, total
                for j in range(n):
                    if j not in selected:
                        # Float check first
                        # We want total - inv[i] + inv[j] < 1
                        # And ideally > best_total
                        if total - inv[i] + inv[j] < 1:
                            new_total = total - inv[i] + inv[j]
                            if new_total > best_total:
                                if try_update((selected - {i}) | {j}, new_total):
                                    selected = best_selected
                                    total = best_total
                                    improved = True
            
            if not improved:
                break
        
        return selected, total
    
    # Run fast local search from greedy
    fast_local_search(best_selected, best_total, deadline - 5)
    
    # Random restarts
    # Use random subsets and improve them
    # We will generate random subsets that are close to 1
    while time.time() < deadline - 2:
        # Generate a random starting set
        # Strategy: Pick a random number of items, biased towards smaller denominators?
        # Or just random and then fix.
        size = random.randint(15, 30)
        start_sel = set(random.sample(range(n), size))
        start_tot = sum(inv[i] for i in start_sel)
        
        if start_tot >= 1:
            # Remove random elements until < 1
            while start_tot >= 1 and start_sel:
                to_remove = random.choice(list(start_sel))
                start_sel.remove(to_remove)
                start_tot -= inv[to_remove]
        
        # Run fast local search with a small time slice
        # If we have 40s left, we can afford to spend ~0.5s per restart
        # But we need to be careful not to timeout.
        # Let's just run it with the global deadline.
        fast_local_search(start_sel, start_tot, deadline - 2)
        
        # If we improved, we might want to continue from the new best?
        # fast_local_search already updates global best.
        # We can restart from the global best to refine it further?
        # That's what the loop does implicitly if we just keep running.
        # But random restarts are better to escape local optima.
        
    return list(best_selected)
# EVOLVE-BLOCK-END

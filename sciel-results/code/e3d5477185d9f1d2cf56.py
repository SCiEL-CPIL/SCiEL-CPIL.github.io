import time
import random
from typing import Set

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 90s kill
    
    target = 200
    max_val = 100
    universe = set(range(1, max_val + 1))
    
    def is_valid(selected):
        """Check if no subset of selected sums to 200."""
        if not selected:
            return True
        reachable = {0}
        for x in selected:
            new_sums = set()
            for s in reachable:
                ns = s + x
                if ns <= target:
                    new_sums.add(ns)
            if target in new_sums:
                return False
            reachable |= new_sums
        return True
    
    def greedy_select(order):
        """Greedily select elements in the given order."""
        selected = set()
        reachable = {0}
        for x in order:
            if (target - x) in reachable:
                continue
            selected.add(x)
            new_sums = {s + x for s in reachable if s + x <= target}
            reachable.update(new_sums)
        return selected
    
    best_solution = None
    best_size = 0
    
    def evaluate_and_update(candidates):
        nonlocal best_solution, best_size
        if not candidates:
            return
        size = len(candidates)
        if size > best_size:
            best_solution = set(candidates)
            best_size = size
    
    # Strategy 1: Known valid structured block {66..100}, size 35
    block = set(range(66, 101))
    evaluate_and_update(block)
    
    # Strategy 2: Greedy descending (seed approach)
    sol1 = greedy_select(list(range(max_val, 0, -1)))
    evaluate_and_update(sol1)
    
    # Strategy 3: Greedy ascending
    sol2 = greedy_select(list(range(1, max_val + 1)))
    evaluate_and_update(sol2)
    
    # Strategy 4: Start from best block, try to add smaller elements
    # Compute reachable sums for the current best
    def try_extend(base_set):
        """Try to add elements from [1, min(base)-1] to base_set."""
        selected = set(base_set)
        reachable = {0}
        for x in sorted(selected):
            new_sums = {s + x for s in reachable if s + x <= target}
            reachable.update(new_sums)
        
        # Try adding elements not in selected, in increasing order
        for x in range(1, max_val + 1):
            if x in selected:
                continue
            if (target - x) in reachable:
                continue
            selected.add(x)
            new_sums = {s + x for s in reachable if s + x <= target}
            reachable.update(new_sums)
        return selected
    
    # Extend the structured block
    extended_block = try_extend(block)
    evaluate_and_update(extended_block)
    
    # Extend the greedy descending solution
    extended_greedy = try_extend(sol1)
    evaluate_and_update(extended_greedy)
    
    # Strategy 5: Randomized greedy with multiple seeds
    random.seed(42)
    
    iterations = 0
    while time.time() < deadline - 5:
        shuffled = list(universe)
        random.shuffle(shuffled)
        sol = greedy_select(shuffled)
        evaluate_and_update(sol)
        
        # Also try extending the randomized greedy result
        if len(sol) >= best_size - 2:
            ext = try_extend(sol)
            evaluate_and_update(ext)
        
        iterations += 1
        
        # Every 50 iterations, try a weighted order biased toward larger numbers
        if iterations % 50 == 0:
            weighted_order = sorted(universe, key=lambda x: -x + random.gauss(0, 10))
            sol_w = greedy_select(weighted_order)
            evaluate_and_update(sol_w)
            ext_w = try_extend(sol_w)
            evaluate_and_update(ext_w)
    
    # Strategy 6: Local search on the best solution - try swaps
    if best_solution and time.time() < deadline - 3:
        best = set(best_solution)
        best_reachable = {0}
        for x in sorted(best):
            new_sums = {s + x for s in best_reachable if s + x <= target}
            best_reachable.update(new_sums)
        
        # Try single swaps: remove one element, add another
        improved = False
        for remove_x in list(best):
            if time.time() > deadline - 2:
                break
            # Recompute reachable without remove_x
            trial_set = best - {remove_x}
            trial_reachable = {0}
            for x in sorted(trial_set):
                new_sums = {s + x for s in trial_reachable if s + x <= target}
                trial_reachable.update(new_sums)
            
            for add_x in range(1, max_val + 1):
                if add_x in trial_set:
                    continue
                if (target - add_x) in trial_reachable:
                    continue
                # This swap doesn't increase size, so skip
                # We only care about net gains
                pass
        
        # Try: remove one, add two (net +1)
        for remove_x in list(best):
            if time.time() > deadline - 1:
                break
            trial_set = best - {remove_x}
            trial_reachable = {0}
            for x in sorted(trial_set):
                new_sums = {s + x for s in trial_reachable if s + x <= target}
                trial_reachable.update(new_sums)
            
            # Try adding one element
            for add_x in range(1, max_val + 1):
                if add_x in trial_set:
                    continue
                if (target - add_x) in trial_reachable:
                    continue
                new_reachable = set(trial_reachable)
                new_sums = {s + add_x for s in new_reachable if s + add_x <= target}
                if target in new_sums:
                    continue
                new_reachable.update(new_sums)
                
                # Now try adding a second element
                for add_y in range(add_x + 1, max_val + 1):
                    if add_y in trial_set or add_y == add_x:
                        continue
                    if (target - add_y) in new_reachable:
                        continue
                    new_sums2 = {s + add_y for s in new_reachable if s + add_y <= target}
                    if target in new_sums2:
                        continue
                    # Valid! Size increases by 1
                    new_best = trial_set | {add_x, add_y}
                    if len(new_best) > len(best):
                        best = new_best
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            evaluate_and_update(best)
    
    # Final verification
    if best_solution:
        if not is_valid(best_solution):
            # Fallback
            fallback = greedy_select(list(range(max_val, 0, -1)))
            return list(fallback)
    
    if best_solution is None:
        return [1]
    
    return list(best_solution)
# EVOLVE-BLOCK-END

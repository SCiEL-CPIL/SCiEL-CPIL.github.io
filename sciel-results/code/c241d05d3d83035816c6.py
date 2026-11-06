import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    N = 100
    TARGET = 200
    MASK = (1 << (TARGET + 1)) - 1
    
    all_numbers = list(range(1, N + 1))
    
    def greedy_construct(order):
        """
        Greedily include numbers in the given order if it doesn't create sum 200.
        Returns the selected list.
        """
        reachable = 1  # bit 0 set
        selected = []
        for x in order:
            # Check if adding x creates sum 200
            # new_reachable = reachable | (reachable << x)
            # We only care if bit 200 becomes 1
            # If (reachable << x) has bit 200 set, then some s in reachable has s+x=200
            if (reachable << x) & (1 << TARGET):
                continue
            reachable |= (reachable << x)
            reachable &= MASK
            selected.append(x)
        return selected

    def local_improve(selected, time_limit):
        """
        Try to improve the selected set by adding or swapping elements.
        """
        selected_set = set(selected)
        excluded = [x for x in all_numbers if x not in selected_set]
        
        # Rebuild reachable bitset for current selected
        reachable = 1
        for x in selected:
            reachable = (reachable | (reachable << x)) & MASK
            
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Try to add any excluded element
            for x in excluded[:]:
                if (reachable << x) & (1 << TARGET):
                    continue
                # Can add x
                selected.append(x)
                selected_set.add(x)
                excluded.remove(x)
                reachable = (reachable | (reachable << x)) & MASK
                improved = True
                break
            
            if improved:
                continue
                
            # 2. Try to swap: remove one selected, add two excluded (net +1)
            # To save time, we only try this if we have time
            if time.time() > time_limit - 5:
                break
                
            found_swap = False
            # Iterate over selected elements to remove
            for i, y in enumerate(selected[:]):
                # Calculate reachable without y
                # This is expensive if done naively. 
                # Optimization: We can just try adding excluded elements to a "virtual" set
                # but checking validity requires the full context.
                # Given N=100, recalculating reachable for 100 elements is ~100 shifts.
                # We do this for each y in selected (up to 100) and each pair in excluded (up to 100^2/2).
                # 100 * 5000 * 100 ops = 50M ops. Might be too slow in Python.
                
                # Let's try a simpler swap: remove y, add x1. (Net 0, but changes composition)
                # Then try to add more.
                # Or just remove y, add x1, x2.
                
                # To keep it fast, let's only try adding 1 element after removal (Net 0) 
                # if it allows adding more later? 
                # Actually, the "Add" step above handles all possible additions.
                # If we can't add anything, we are stuck unless we remove something.
                # Removing something allows us to add 2 things.
                
                # Let's try removing y and adding x1, x2.
                # We need to check if S \ {y} + {x1, x2} is valid.
                
                # Pre-compute reachable without y?
                # reachable_without_y is hard to get from reachable_with_y directly (non-invertible).
                # So we must recompute.
                
                # Given time constraints, let's limit the scope of swaps.
                # Only try swaps if len(selected) is small or we have lots of time.
                # For now, let's just do single-adds which are fast.
                # The greedy phase usually gets close to optimal.
                
                pass 
            
            # If no single add worked, and we don't do complex swaps, we stop.
            # Let's implement a simple swap: remove y, add x.
            # This doesn't increase cardinality, but might unblock further additions.
            # However, if we can't add anything now, removing y and adding x (same size) 
            # might allow adding z later.
            
            # Let's try a limited search for swaps that increase size.
            # We'll pick a random y to remove and try to find two x's to add.
            if time.time() < time_limit - 2:
                if selected and excluded:
                    # Pick a random y to remove to avoid bias
                    idx = random.randint(0, len(selected) - 1)
                    y = selected[idx]
                    
                    # Compute reachable without y
                    r_no_y = 1
                    for z in selected:
                        if z != y:
                            r_no_y = (r_no_y | (r_no_y << z)) & MASK
                    
                    # Try to find x1, x2 in excluded such that S\{y}+{x1,x2} is valid
                    # And specifically, we want to check if we can add x1, then x2.
                    # Or just check the final state.
                    
                    # To be efficient, iterate x1, then x2
                    for x1 in excluded:
                        # Check if x1 can be added to S\{y}
                        if (r_no_y << x1) & (1 << TARGET):
                            continue
                        r_with_x1 = (r_no_y | (r_no_y << x1)) & MASK
                        for x2 in excluded:
                            if x2 == x1:
                                continue
                            if (r_with_x1 << x2) & (1 << TARGET):
                                continue
                            # Valid swap found: remove y, add x1, x2
                            # Update state
                            selected.remove(y)
                            selected_set.discard(y)
                            selected.extend([x1, x2])
                            selected_set.update([x1, x2])
                            excluded.remove(x1)
                            excluded.remove(x2)
                            
                            # Update reachable
                            reachable = r_with_x1
                            reachable = (reachable | (reachable << x2)) & MASK
                            
                            improved = True
                            found_swap = True
                            break
                        if found_swap:
                            break
                    if found_swap:
                        break

        return selected

    best_result = []
    
    # 1. Descending order
    r = greedy_construct(all_numbers[::-1])
    if len(r) > len(best_result):
        best_result = r
        
    # 2. Ascending order
    r = greedy_construct(all_numbers)
    if len(r) > len(best_result):
        best_result = r
        
    # 3. Random shuffles
    rng = random.Random(42)
    # Run a few random shuffles
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        order = all_numbers[:]
        rng.shuffle(order)
        r = greedy_construct(order)
        if len(r) > len(best_result):
            best_result = r
            
    # 4. Local improvement on the best result found
    if time.time() < deadline - 5:
        best_result = local_improve(best_result, deadline - 1)
        
    if not best_result:
        return [100]
        
    return best_result

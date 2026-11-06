"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin before 60s limit
    
    MASK = (1 << 201) - 1  # bits 0..200
    
    def can_add(reachable, x):
        """Check if adding x to current reachable set creates sum 200."""
        new_bits = (reachable << x) & MASK
        return not (new_bits & (1 << 200))
    
    def greedy_descending():
        """Greedy: include numbers in descending order if it doesn't create sum 200."""
        reachable = 1  # bit 0 set
        selected = []
        for x in range(100, 0, -1):
            if can_add(reachable, x):
                reachable |= (reachable << x) & MASK
                selected.append(x)
        return selected, reachable
    
    def greedy_ascending():
        """Greedy: include numbers in ascending order."""
        reachable = 1
        selected = []
        for x in range(1, 101):
            if can_add(reachable, x):
                reachable |= (reachable << x) & MASK
                selected.append(x)
        return selected, reachable
    
    def local_improve(selected, reachable):
        """Try to add excluded elements or do swaps to increase cardinality."""
        selected_set = set(selected)
        all_numbers = set(range(1, 101))
        excluded = [x for x in range(1, 101) if x not in selected_set]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try adding each excluded element
            for x in excluded[:]:
                if can_add(reachable, x):
                    selected.append(x)
                    selected_set.add(x)
                    excluded.remove(x)
                    reachable |= (reachable << x) & MASK
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swap: remove one from selected, add one from excluded
            # Only do this if it increases cardinality (which it won't for 1-for-1)
            # So try: remove one, add two
            if len(excluded) >= 2:
                for i in range(len(selected)):
                    y = selected[i]
                    # Compute reachable without y
                    # We need to recompute since we don't have a way to "remove" from bitset
                    # This is expensive, so only try a few candidates
                    if time.time() > deadline - 5:
                        break
                    
                    # Try removing y and adding two excluded elements
                    # Recompute reachable without y
                    selected_no_y = selected[:i] + selected[i+1:]
                    reachable_no_y = 1
                    for z in selected_no_y:
                        reachable_no_y |= (reachable_no_y << z) & MASK
                    
                    # Try adding pairs from excluded
                    found = False
                    for j in range(len(excluded)):
                        x1 = excluded[j]
                        if not can_add(reachable_no_y, x1):
                            continue
                        r1 = (reachable_no_y << x1) & MASK
                        for k in range(j+1, len(excluded)):
                            x2 = excluded[k]
                            if not can_add(r1, x2):
                                continue
                            # Swap y for x1, x2
                            selected_no_y.extend([x1, x2])
                            selected_set.discard(y)
                            selected_set.update([x1, x2])
                            excluded.remove(x1)
                            excluded.remove(x2)
                            selected = selected_no_y
                            reachable = 1
                            for z in selected:
                                reachable |= (reachable << z) & MASK
                            improved = True
                            found = True
                            break
                        if found:
                            break
                    if found:
                        break
        return selected, reachable
    
    # Try different greedy strategies
    best_result = []
    best_reachable = 0
    
    # Descending greedy (likely best)
    r1, reachable1 = greedy_descending()
    if len(r1) > len(best_result):
        best_result = r1
        best_reachable = reachable1
    
    # Ascending greedy
    r2, reachable2 = greedy_ascending()
    if len(r2) > len(best_result):
        best_result = r2
        best_reachable = reachable2
    
    # Local improvement on best result
    if time.time() < deadline - 5:
        best_result, best_reachable = local_improve(best_result, best_reachable)
    
    if not best_result:
        return [100]
    
    return best_result
# EVOLVE-BLOCK-END

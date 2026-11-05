import time
import random

def _can_add(x, selected):
    """Check if x can be added to selected without violating the constraint."""
    n = len(selected)
    if n >= 3:
        # x must not divide sum of any 3 distinct elements in selected
        # Iterate pairs i, j and check k
        for i in range(n):
            si = selected[i]
            for j in range(i+1, n):
                sij = si + selected[j]
                for k in range(j+1, n):
                    if (sij + selected[k]) % x == 0:
                        return False
    
    # For each a in selected, a must not divide x + u + v for any u, v in selected \ {a}
    for a_idx, a in enumerate(selected):
        # We need to check pairs (u, v) from selected excluding a
        # u, v distinct, u != v, u != a, v != a
        # (x + u + v) % a == 0
        for i in range(n):
            if i == a_idx:
                continue
            si = selected[i]
            for j in range(i+1, n):
                if j == a_idx:
                    continue
                sj = selected[j]
                if (x + si + sj) % a == 0:
                    return False
    return True

def _greedy(order, deadline):
    """Run greedy in the given order."""
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _local_search(selected, deadline):
    """Improve the set by adding elements or performing swaps that enable growth."""
    selected = list(selected)
    in_set = set(selected)
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        
        # 1. Try to add any element
        for x in range(192, 0, -1):
            if x not in in_set and _can_add(x, selected):
                selected.append(x)
                in_set.add(x)
                improved = True
                break
        
        # 2. If no addition, try swaps
        if not improved:
            # Try to swap one out and one in
            # We want a swap that allows us to add more elements afterwards
            candidates_out = list(selected)
            random.shuffle(candidates_out)
            
            for x_out in candidates_out:
                if time.time() > deadline - 2.0:
                    return selected
                
                selected_without = [v for v in selected if v != x_out]
                
                # Try to find an x_in that makes the set valid and potentially growable
                for x_in in range(192, 0, -1):
                    if x_in in in_set:
                        continue
                    
                    if _can_add(x_in, selected_without):
                        new_set = selected_without + [x_in]
                        new_in_set = set(new_set)
                        
                        # Heuristic: Check if we can add at least one more element
                        can_grow = False
                        for x_add in range(192, 0, -1):
                            if x_add not in new_in_set:
                                if _can_add(x_add, new_set):
                                    can_grow = True
                                    break
                        
                        if can_grow:
                            selected = new_set
                            in_set = new_in_set
                            improved = True
                            break
                
                if improved:
                    break
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best = []
    
    # Strategy 1: Greedy descending
    t1 = time.time()
    result = _greedy(range(192, 0, -1), deadline)
    if len(result) > len(best):
        best = result
    
    # Strategy 2: Greedy ascending
    if time.time() < deadline - 5:
        result = _greedy(range(1, 193), deadline)
        if len(result) > len(best):
            best = result

    # Strategy 3: Multiple random shuffles
    random.seed(42)
    for _ in range(50):
        if time.time() > deadline - 20:
            break
        order = list(range(1, 193))
        random.shuffle(order)
        result = _greedy(order, deadline)
        if len(result) > len(best):
            best = result
            # Update best if we found something better, maybe restart local search on this?
            # For now, just keep the best raw greedy result
    
    # Strategy 4: Local search on the best greedy result
    if time.time() < deadline - 10:
        best = _local_search(best, deadline)
        
    # Strategy 5: Local search on other good results?
    # We can keep a list of top 3 greedy results and run local search on each
    
    return best

import time
import random
import sys

N = 256
MAX_SUM = 2 * N  # Maximum possible sum is 2*N

def _is_valid_sidon(sel):
    """Check if sel is a valid Sidon set."""
    sums = set()
    for i in range(len(sel)):
        a = sel[i]
        for j in range(i, len(sel)):
            s = a + sel[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(sel):
    """Check if sel is inclusion-maximal."""
    if not _is_valid_sidon(sel):
        return False
    sel_set = set(sel)
    sums = set()
    for i in range(len(sel)):
        a = sel[i]
        for j in range(i, len(sel)):
            sums.add(a + sel[j])
    
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if x can be added
        can_add = True
        if 2 * x in sums:
            can_add = False
        else:
            for a in sel:
                if x + a in sums:
                    can_add = False
                    break
        if can_add:
            return False
    return True

def _greedy_backtrack(order, deadline):
    """
    Greedy backtracking: build set by choosing elements that block the most candidates.
    Returns a maximal Sidon set.
    """
    selected = []
    selected_set = set()
    sums = set()
    available = set(order)
    
    while available and time.time() < deadline:
        # For each available element, compute how many other elements it would block
        best_elem = None
        best_block_count = -1
        best_new_sums = None
        
        for x in available:
            # Check if x can be added
            if 2 * x in sums:
                continue
            can_add = True
            for a in selected:
                if x + a in sums:
                    can_add = False
                    break
            if not can_add:
                continue
            
            # Compute new sums if we add x
            new_sums = set(sums)
            new_sums.add(2 * x)
            for a in selected:
                new_sums.add(x + a)
            
            # Count how many available elements are now blocked
            block_count = 0
            for y in available:
                if y == x:
                    continue
                # Check if y is blocked by new_sums
                if 2 * y in new_sums:
                    block_count += 1
                    continue
                for a in selected + [x]:
                    if y + a in new_sums:
                        block_count += 1
                        break
            
            if block_count > best_block_count:
                best_block_count = block_count
                best_elem = x
                best_new_sums = new_sums
        
        if best_elem is None:
            break
        
        # Add the best element
        selected.append(best_elem)
        selected_set.add(best_elem)
        available.remove(best_elem)
        sums = best_new_sums
    
    # Now we have a Sidon set. Make it maximal by adding necessary elements
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, N + 1):
            if x in selected_set:
                continue
            # Check if x can be added
            if 2 * x in sums:
                continue
            can_add = True
            for a in selected:
                if x + a in sums:
                    can_add = False
                    break
            if can_add:
                selected.append(x)
                selected_set.add(x)
                sums.add(2 * x)
                for a in selected:
                    if a != x:
                        sums.add(x + a)
                changed = True
                break
    
    return selected

def _local_search_reduce(sel, deadline):
    """Try to remove elements while maintaining maximality."""
    current_sel = list(sel)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        n = len(current_sel)
        
        for i in range(n):
            if time.time() > deadline:
                return current_sel
            
            x_to_remove = current_sel[i]
            new_sel = current_sel[:i] + current_sel[i+1:]
            
            if _is_maximal(new_sel):
                current_sel = new_sel
                improved = True
                break
    
    return current_sel

def _try_remove_pairs(sel, deadline):
    """Try removing pairs of elements."""
    current_sel = list(sel)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        n = len(current_sel)
        
        for i in range(n):
            if time.time() > deadline:
                return current_sel
            for j in range(i + 1, n):
                new_sel = current_sel[:i] + current_sel[i+1:j] + current_sel[j+1:]
                if _is_maximal(new_sel):
                    current_sel = new_sel
                    improved = True
                    break
            if improved:
                break
    
    return current_sel

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    best_set = None
    best_size = float('inf')
    
    # 1. Try backtracking with different orderings
    rng = random.Random(42)
    
    # Try ascending, descending, and random orders
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
    ]
    
    # Add random orders
    for _ in range(20):
        order = list(range(1, N + 1))
        rng.shuffle(order)
        orders.append(order)
    
    for order in orders:
        if time.time() > deadline - 5:
            break
        
        sel = _greedy_backtrack(order, deadline)
        
        if not sel:
            continue
        
        # Verify and update best
        if _is_valid_sidon(sel) and _is_maximal(sel):
            if len(sel) < best_size:
                best_size = len(sel)
                best_set = list(sel)
    
    # 2. Local search on best found
    if best_set is not None:
        if time.time() < deadline - 3:
            optimized = _local_search_reduce(best_set, deadline)
            if _is_valid_sidon(optimized) and _is_maximal(optimized):
                if len(optimized) < best_size:
                    best_size = len(optimized)
                    best_set = optimized
        
        if time.time() < deadline - 2:
            further = _try_remove_pairs(best_set, deadline)
            if _is_valid_sidon(further) and _is_maximal(further):
                if len(further) < best_size:
                    best_size = len(further)
                    best_set = further
    
    # 3. Fallback: simple greedy if nothing found
    if best_set is None:
        sel = _greedy_backtrack(list(range(1, N + 1)), deadline)
        if sel:
            best_set = sel
        else:
            # Last resort: ascending greedy
            selected = []
            sums = set()
            for x in range(1, N + 1):
                if 2 * x in sums:
                    continue
                can_add = True
                for a in selected:
                    if x + a in sums:
                        can_add = False
                        break
                if can_add:
                    selected.append(x)
                    sums.add(2 * x)
                    for a in selected:
                        if a != x:
                            sums.add(x + a)
            best_set = selected
    
    return best_set

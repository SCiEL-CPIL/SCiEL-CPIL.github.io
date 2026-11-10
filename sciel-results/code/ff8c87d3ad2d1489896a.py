"""Improved program for small inclusion-maximal Sidon sets in [1,256]."""
import time
import random
from itertools import combinations

N = 256
TIME_LIMIT = 55.0  # safety margin before 60s kill

def _is_sidon_fast(selected):
    """Check if selected is a Sidon set (all a+b for a<=b distinct)."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        for j in range(i, n):
            s = a + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _get_sums(selected):
    """Get all pair sums a+b for a<=b in selected."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        for j in range(i, n):
            sums.add(a + selected[j])
    return sums

def _can_add(x, selected, sums):
    """Check if x can be added to selected without breaking Sidon property."""
    for a in selected:
        s = a + x
        if s in sums:
            return False
    if 2 * x in sums:
        return False
    # Also need to check that new sums don't collide with each other
    # Since all a are distinct and x is new, a1+x = a2+x implies a1=a2, so no collision
    # But we need to check 2x vs a+x: 2x = a+x => a=x, but x not in selected
    return True

def _is_maximal_fast(selected, N):
    """Check if selected is inclusion-maximal in [1,N]."""
    sel_set = set(selected)
    sums = _get_sums(selected)
    
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        if _can_add(x, selected, sums):
            return False
    return True

def _count_blocked(x, selected, sums):
    """Count how many existing elements would conflict if x were added."""
    count = 0
    for a in selected:
        if (a + x) in sums or (2 * x) in sums:
            count += 1
    return count

def _greedy_construct(order):
    """Greedy construction using given order of elements."""
    selected = []
    sums = set()
    for x in order:
        # Check if x can be added
        ok = True
        for a in selected:
            s = x + a
            if s in sums:
                ok = False
                break
        if ok and 2 * x in sums:
            ok = False
        if ok:
            # Also check internal consistency of new sums
            new_sums = set()
            for a in selected:
                new_sums.add(x + a)
            new_sums.add(2 * x)
            if len(new_sums) != len(selected) + 1:
                ok = False
            if ok:
                selected.append(x)
                sums.update(new_sums)
    return selected

def _local_search_improved(selected, N, time_limit):
    """Improved local search with smarter removal ordering."""
    start = time.time()
    selected = list(selected)
    
    # Phase 1: Iteratively remove elements, prioritizing those that block fewest candidates
    improved = True
    while improved:
        if time.time() - start > time_limit:
            break
        improved = False
        
        # Score each element by how many candidates it blocks
        sums = _get_sums(selected)
        sel_set = set(selected)
        
        # For each element, count how many missing elements it conflicts with
        block_counts = {}
        for i, elem in enumerate(selected):
            block_count = 0
            for x in range(1, N + 1):
                if x in sel_set:
                    continue
                # Check if removing elem would allow x to be added
                # Simplified: count conflicts of elem with x
                if (elem + x) in sums or (2 * x) in sums:
                    block_count += 1
            block_counts[i] = block_count
        
        # Try removing elements in order of lowest block count
        indices = sorted(block_counts.keys(), key=lambda i: block_counts[i])
        
        for i in indices:
            if time.time() - start > time_limit:
                break
            new_sel = selected[:i] + selected[i+1:]
            if _is_sidon_fast(new_sel) and _is_maximal_fast(new_sel, N):
                selected = new_sel
                improved = True
                break
    
    # Phase 2: Try single-element swaps (replace one with another, same size)
    # This can help find configurations where different elements are more "efficient"
    for iteration in range(30):
        if time.time() - start > time_limit:
            break
        
        found_swap = False
        sel_set = set(selected)
        
        for i in range(len(selected)):
            if time.time() - start > time_limit:
                break
            elem_to_remove = selected[i]
            
            # Try replacing with each missing element
            for x in range(1, N + 1):
                if x in sel_set:
                    continue
                new_sel = selected[:i] + [x] + selected[i+1:]
                if _is_sidon_fast(new_sel) and _is_maximal_fast(new_sel, N):
                    # Accept the swap if it might help (e.g., different structure)
                    # For now, just accept any valid swap to explore the space
                    selected = new_sel
                    sel_set = set(selected)
                    found_swap = True
                    break
            if found_swap:
                break
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best = None
    best_size = float('inf')
    
    rng = random.Random(42)
    
    # Try multiple construction strategies
    orders = []
    
    # Ascending
    orders.append(list(range(1, N + 1)))
    # Descending
    orders.append(list(range(N, 0, -1)))
    
    # Random orders - generate many
    num_random = 300
    for _ in range(num_random):
        order = list(range(1, N + 1))
        rng.shuffle(order)
        orders.append(order)
    
    # Also try some structured orders
    # Even then odd
    orders.append(list(range(2, N + 1, 2)) + list(range(1, N + 1, 2)))
    # Odd then even
    orders.append(list(range(1, N + 1, 2)) + list(range(2, N + 1, 2)))
    
    for order in orders:
        if time.time() > deadline - 3.0:
            break
        
        sel = _greedy_construct(order)
        if len(sel) < best_size:
            best = sel
            best_size = len(sel)
            
            # Try local search if we have time
            remaining = deadline - time.time() - 2.0
            if remaining > 5.0:
                improved = _local_search_improved(sel, N, remaining)
                if len(improved) < best_size:
                    best = improved
                    best_size = len(best)
    
    # Final polish
    remaining = deadline - time.time() - 1.0
    if remaining > 2.0 and best is not None:
        polished = _local_search_improved(best, N, remaining)
        if len(polished) < best_size:
            best = polished
            best_size = len(best)
    
    # Fallback
    if best is None:
        best = _greedy_construct(list(range(1, N + 1)))
    
    # Ensure validity
    if not _is_sidon_fast(best):
        best = _greedy_construct(list(range(1, N + 1)))
    
    if not _is_maximal_fast(best, N):
        # Extend to maximal
        sel_set = set(best)
        sums = _get_sums(best)
        for x in range(1, N + 1):
            if x in sel_set:
                continue
            if _can_add(x, best, sums):
                best.append(x)
                sel_set.add(x)
                for a in best[:-1]:
                    sums.add(a + x)
                sums.add(2 * x)
        best.sort()
    
    return best

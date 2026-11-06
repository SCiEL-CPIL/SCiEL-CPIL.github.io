"""Optimized program: fast greedy with efficient maximality check and local search."""
import random
import time

N = 256

def _is_sidon(sel):
    """Check if sel is a Sidon set."""
    sums = set()
    for i in range(len(sel)):
        a = sel[i]
        for j in range(i, len(sel)):
            b = sel[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def _compute_sums(sel):
    sums = set()
    for i in range(len(sel)):
        a = sel[i]
        for j in range(i, len(sel)):
            sums.add(a + sel[j])
    return sums

def _is_maximal_fast(sel, sums):
    """
    Check if sel is inclusion-maximal.
    Returns True if no missing element can be added.
    Uses precomputed sums for efficiency.
    """
    sel_set = set(sel)
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        
        # Check if x can be added.
        # 1. 2*x must not be in sums
        if 2 * x in sums:
            continue
            
        # 2. x + a must not be in sums for any a in sel
        #    Also x+a must be distinct from other x+b (true since a!=b)
        #    and distinct from 2*x (true since a!=x)
        blocked = False
        for a in sel:
            if (x + a) in sums:
                blocked = True
                break
                
        if not blocked:
            # If we found an x that is not blocked, the set is NOT maximal
            return False
            
    return True

def _greedy_construct(order):
    """
    Greedily select elements from order to form a Sidon set.
    Returns the selected list and the set of sums.
    """
    selected = []
    sums = set()
    for x in order:
        # Check if x can be added to current selected set
        # 1. 2*x not in sums
        if 2 * x in sums:
            continue
        # 2. x+a not in sums for any a in selected
        can_add = True
        for a in selected:
            if (x + a) in sums:
                can_add = False
                break
        if can_add:
            selected.append(x)
            sums.add(2 * x)
            for a in selected:
                # Note: selected now includes x, so x+x is handled.
                # We need to add x+a for all a in selected BEFORE x was added?
                # Actually, the sums added are x+a for a in selected.
                # Since x is now in selected, the loop `for a in selected` includes x.
                # So x+x is added. x+a for a!=x are added.
                # This is correct.
                sums.add(x + a)
    return selected, sums

def construct():
    deadline = time.time() + 25  # Safety margin for 30s limit
    
    best_set = None
    best_size = float('inf')
    
    # 1. Try ascending greedy
    order_asc = list(range(1, N + 1))
    sel, sums = _greedy_construct(order_asc)
    if _is_maximal_fast(sel, sums):
        best_set = sel
        best_size = len(sel)
        
    # 2. Try descending greedy
    if time.time() < deadline:
        order_desc = list(range(N, 0, -1))
        sel, sums = _greedy_construct(order_desc)
        if _is_maximal_fast(sel, sums):
            if len(sel) < best_size:
                best_size = len(sel)
                best_set = sel

    # 3. Randomized Greedy Trials
    rng = random.Random(42)
    trials = 0
    while time.time() < deadline - 5:
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        sel, sums = _greedy_construct(order)
        
        # Quick check: is it maximal?
        if _is_maximal_fast(sel, sums):
            if len(sel) < best_size:
                best_size = len(sel)
                best_set = sel
                
                # If we found a very small set, try to minimize it further with local search
                # Local Search: Try removing elements
                # A subset of a Sidon set is Sidon.
                # We need to check if the subset is still Maximal.
                # If removing an element makes it non-maximal (i.e., some other element becomes addable),
                # then we shouldn't remove it?
                # Wait. "Maximal" means you CAN'T add anything.
                # If we remove an element, the set becomes smaller.
                # It is possible that the smaller set is NOT maximal (i.e., we could add something back,
                # or add something else).
                # We want the smallest *Maximal* set.
                # So if we remove x, and the new set S' is Maximal, we keep it.
                # If S' is not Maximal, we discard x (keep original S).
                
                # Optimization: Only do local search if we have time.
                if time.time() < deadline - 2:
                    improved = True
                    while improved:
                        improved = False
                        new_sums = sums
                        for i in range(len(sel)):
                            # Try removing sel[i]
                            x_to_remove = sel[i]
                            # Construct new set
                            # Instead of rebuilding sums from scratch, we can try to undo?
                            # It's safer to rebuild for correctness in this simple implementation.
                            new_sel = sel[:i] + sel[i+1:]
                            new_sums = _compute_sums(new_sel)
                            
                            if _is_maximal_fast(new_sel, new_sums):
                                # We found a smaller maximal set
                                sel = new_sel
                                sums = new_sums
                                improved = True
                                best_size = len(sel)
                                best_set = sel
                                break # Restart loop with new set size
                                # Note: breaking the for loop, not the while.
                                # The while loop will re-evaluate 'improved' which is True, so it continues.
                                # But we modified sel. The range(len(sel)) in the for loop was captured at start?
                                # No, `range(len(sel))` is evaluated once.
                                # If we modify sel, the indices might be off if we continue the for loop.
                                # So `break` is good.
                
                # If we found a very small set, we might stop early
                if best_size <= 8:
                    break
                    
        trials += 1
        
    if best_set is None:
        # Fallback
        return [1]
        
    return best_set

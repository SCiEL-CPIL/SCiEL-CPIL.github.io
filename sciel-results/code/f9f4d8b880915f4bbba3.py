"""Small inclusion-maximal Sidon sets in [1,256]"""
import time
import random

N = 256

# EVOLVE-BLOCK-START

def _is_sidon(selected):
    """Check if selected is a Sidon set (all a+b with a<=b distinct)."""
    s = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            total = a + b
            if total in s:
                return False
            s.add(total)
    return True

def _greedy_build(order, deadline):
    """Build a Sidon set by processing elements in the given order.
    Returns (selected, sums) where sums is the set of all a+b sums."""
    selected = []
    sums = set()
    for x in order:
        if time.time() > deadline:
            break
        # Check if x can be added
        ok = True
        added_sums = set()
        for a in selected:
            s = x + a
            if s in sums or s in added_sums:
                ok = False
                break
            added_sums.add(s)
        if ok:
            s = 2 * x
            if s in sums or s in added_sums:
                ok = False
            else:
                added_sums.add(s)
        if ok:
            selected.append(x)
            sums.update(added_sums)
    return selected, sums

def _maximalize(selected, sums, deadline):
    """Try to make the set maximal by adding elements that don't break Sidon.
    Returns (new_selected, new_sums)."""
    # Find all addable elements
    addable = []
    for x in range(1, N + 1):
        if x in selected:
            continue
        ok = True
        for a in selected:
            s = x + a
            if s in sums:
                ok = False
                break
        if ok:
            s = 2 * x
            if s in sums:
                ok = False
        if ok:
            addable.append(x)
    
    # Greedily add addable elements in some order
    # Try to add them to make the set maximal
    new_selected = list(selected)
    new_sums = set(sums)
    
    # Keep trying to add elements until none can be added
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Recompute addable
        addable = []
        for x in range(1, N + 1):
            if x in new_selected:
                continue
            ok = True
            for a in new_selected:
                s = x + a
                if s in new_sums:
                    ok = False
                    break
            if ok:
                s = 2 * x
                if s in new_sums:
                    ok = False
            if ok:
                addable.append(x)
        
        if not addable:
            break
        
        # Add the first addable element (or random one)
        # To maximize saturation, add the one that blocks the most others
        # For simplicity, add the smallest
        x = addable[0]
        new_selected.append(x)
        for a in new_selected:
            if a != x:
                new_sums.add(x + a)
        new_sums.add(2 * x)
        changed = True
    
    return new_selected, new_sums

def _is_maximal(selected, sums):
    """Check if the Sidon set is inclusion-maximal."""
    for x in range(1, N + 1):
        if x in selected:
            continue
        # Check if x can be added
        ok = True
        for a in selected:
            s = x + a
            if s in sums:
                ok = False
                break
        if ok:
            s = 2 * x
            if s in sums:
                ok = False
        if ok:
            return False  # x can be added, so not maximal
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    
    # Strategy: try many random orderings, keep the best (smallest) maximal set
    best_selected = None
    best_size = N + 1
    
    # First, try the standard ascending greedy as baseline
    selected, sums = _greedy_build(range(1, N + 1), deadline)
    if selected:
        # Make it maximal
        selected, sums = _maximalize(selected, sums, deadline)
        if _is_sidon(selected) and _is_maximal(selected, sums):
            best_selected = selected
            best_size = len(selected)
    
    # Now try random orderings
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline and attempts < 5000:
        attempts += 1
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        selected, sums = _greedy_build(order, deadline)
        if not selected:
            continue
        
        # Make it maximal
        selected, sums = _maximalize(selected, sums, deadline)
        
        if len(selected) >= best_size:
            continue
        
        # Verify
        if _is_sidon(selected) and _is_maximal(selected, sums):
            best_selected = selected
            best_size = len(selected)
    
    if best_selected is None:
        # Fallback: return the ascending greedy result
        selected, sums = _greedy_build(range(1, N + 1), deadline)
        selected, sums = _maximalize(selected, sums, deadline)
        best_selected = selected
    
    return sorted(best_selected)

# EVOLVE-BLOCK-END

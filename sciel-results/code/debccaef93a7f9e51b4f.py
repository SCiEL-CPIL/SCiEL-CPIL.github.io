"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
import time

_TARGET = 200
_UNIVERSE = list(range(1, 101))
_MASK = (1 << (_TARGET + 1)) - 1  # Mask to keep bits 0..200

def _greedy_descending_bitset():
    """Greedy: process from largest to smallest, keep if it does not create a sum of 200.
    Uses bitset for speed."""
    selected = []
    reachable = 1  # bit 0 is set (sum 0)
    for x in range(100, 0, -1):
        # Check if adding x creates sum 200
        # (reachable << x) has bit 200 set iff reachable has bit 200-x set
        if ((reachable << x) & (1 << _TARGET)) == 0:
            selected.append(x)
            reachable |= (reachable << x)
            reachable &= _MASK
    return selected

def _local_improve_bitset(selected):
    """Try to add back excluded elements one at a time if safe. Repeat until no change.
    Uses bitset for speed."""
    selected_set = set(selected)
    excluded = [x for x in _UNIVERSE if x not in selected_set]
    
    # Build initial bitset for current selected
    reachable = 1
    for x in selected:
        reachable |= (reachable << x)
        reachable &= _MASK

    improved = True
    while improved:
        improved = False
        for x in excluded:
            if x in selected_set:
                continue
            # Check if adding x is safe
            new_reachable = reachable | (reachable << x)
            if (new_reachable & (1 << _TARGET)) == 0:
                # Safe to add
                selected_set.add(x)
                reachable = new_reachable & _MASK
                improved = True
    return list(selected_set)

def _try_orders_bitset():
    """Try multiple construction orders and keep the best (largest) valid set."""
    best = None
    orders = [
        list(range(100, 0, -1)),   # descending
        list(range(1, 101)),       # ascending
    ]
    
    # Add heuristics
    evens = [x for x in range(100, 0, -2)]
    odds = [x for x in range(99, 0, -2)]
    orders.append(evens + odds)
    orders.append(odds + evens)
    
    for order in orders:
        selected = []
        reachable = 1
        for x in order:
            if ((reachable << x) & (1 << _TARGET)) == 0:
                selected.append(x)
                reachable |= (reachable << x)
                reachable &= _MASK
        # local improvement
        sel = _local_improve_bitset(selected)
        if best is None or len(sel) > len(best):
            best = sel
    return best

def _has_sum_bitset(selected, target=_TARGET):
    """Check validity using bitset."""
    reachable = 1
    for x in selected:
        reachable |= (reachable << x)
        reachable &= (1 << (target + 1)) - 1
    return bool(reachable & (1 << target))

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    # 1. Greedy descending
    best = _greedy_descending_bitset()
    best = _local_improve_bitset(best)
    
    # 2. Try other orders
    if time.time() < deadline:
        cand = _try_orders_bitset()
        if cand is not None and len(cand) > len(best):
            best = cand

    # Final safety check
    if _has_sum_bitset(best):
        # Fallback
        best = _greedy_descending_bitset()
        best = _local_improve_bitset(best)

    if not best:
        best = [100]

    return best
# EVOLVE-BLOCK-END

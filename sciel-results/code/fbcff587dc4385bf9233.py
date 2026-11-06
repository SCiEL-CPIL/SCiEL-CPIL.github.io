import random
import time
# EVOLVE-BLOCK-START

_LCM = 232792560
_INV = {x: _LCM // x for x in range(1, 23)}
_ALL = list(range(1, 23))

def _greedy_order(order, deadline):
    """
    Given an ordering of candidates, greedily add elements maintaining dissociation.
    Returns (set_of_elements, set_of_subset_sums).
    Uses fast incremental collision check.
    """
    sums = {0}
    chosen = set()
    for x in order:
        if x in chosen:
            continue
        inv = _INV[x]
        # collision check: does shifting sums by inv hit sums?
        # Use set comprehension and isdisjoint for speed (C implementation)
        shifted = {s + inv for s in sums}
        if not sums.isdisjoint(shifted):
            continue
        # accept
        # update sums: add s+inv for all s
        # sums = sums | shifted is faster than loop
        sums = sums | shifted
        chosen.add(x)
        if time.time() > deadline:
            break
    return chosen, sums

def _complete(chosen, sums, deadline):
    """Try to add more elements to chosen in random order until no more fit."""
    remaining = [x for x in _ALL if x not in chosen]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(remaining)
        still_remaining = []
        for x in remaining:
            if x in chosen:
                continue
            inv = _INV[x]
            shifted = {s + inv for s in sums}
            if sums.isdisjoint(shifted):
                sums = sums | shifted
                chosen.add(x)
                improved = True
            else:
                still_remaining.append(x)
        remaining = still_remaining
    return chosen, sums

def _local_search(chosen, sums, deadline):
    """Try swap-based local search: remove one element, try to add two others."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        if len(chosen) < 2:
            break
        # Try removing each element and adding a different one
        candidates = list(chosen)
        random.shuffle(candidates)
        for rem in candidates:
            if time.time() > deadline:
                break
            if rem not in chosen:
                continue
            # Remove rem
            rem_inv = _INV[rem]
            # Rebuild sums without rem: we need to subtract
            # Actually, let's just rebuild from scratch for simplicity when set is small
            # But that's expensive. Let's use a different approach:
            # Try all pairs of (remove one, add one)
            new_chosen = set(chosen)
            new_chosen.discard(rem)
            # Rebuild sums for new_chosen
            new_sums = {0}
            for x in new_chosen:
                inv = _INV[x]
                shifted = {s + inv for s in new_sums}
                new_sums = new_sums | shifted
            # Try to add any element not in new_chosen
            for add in _ALL:
                if add in new_chosen:
                    continue
                inv = _INV[add]
                shifted = {s + inv for s in new_sums}
                if new_sums.isdisjoint(shifted):
                    new_sums = new_sums | shifted
                    new_chosen.add(add)
                    # Try to add more
                    for add2 in _ALL:
                        if add2 in new_chosen:
                            continue
                        inv2 = _INV[add2]
                        shifted2 = {s + inv2 for s in new_sums}
                        if new_sums.isdisjoint(shifted2):
                            new_sums = new_sums | shifted2
                            new_chosen.add(add2)
                    if len(new_chosen) > len(chosen):
                        chosen = new_chosen
                        sums = new_sums
                        improved = True
                        break
            if improved:
                break
    return chosen, sums

def _polish(chosen, sums, deadline):
    """Final high-precision polish: try all single-element swaps systematically."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        if len(chosen) < 1:
            break
        # Try each element in chosen for removal
        for rem in list(chosen):
            if time.time() > deadline:
                break
            # Rebuild sums without rem
            new_chosen = set(chosen)
            new_chosen.discard(rem)
            new_sums = {0}
            for x in new_chosen:
                inv = _INV[x]
                shifted = {s + inv for s in new_sums}
                new_sums = new_sums | shifted
            
            # Try to add each element not in new_chosen
            for add in _ALL:
                if add in new_chosen:
                    continue
                inv = _INV[add]
                shifted = {s + inv for s in new_sums}
                if new_sums.isdisjoint(shifted):
                    candidate_sums = new_sums | shifted
                    candidate_chosen = new_chosen | {add}
                    # Check if this improves cardinality or is equal but we want to explore
                    if len(candidate_chosen) > len(chosen):
                        chosen = candidate_chosen
                        sums = candidate_sums
                        improved = True
                        break
            if improved:
                break
    return chosen, sums

def construct():
    start = time.time()
    deadline = start + 50  # safety margin under 60s (evaluator kills at 90s)

    best = set()
    best_size = 0

    # Warm start: try different orderings including sorted by size
    # Larger elements have smaller reciprocals, less likely to collide
    warm_orders = [
        _ALL[:],              # natural order
        sorted(_ALL, reverse=True),  # descending
        sorted(_ALL, key=lambda x: _INV[x], reverse=True),  # by inverse desc
        sorted(_ALL, key=lambda x: _INV[x]),  # by inverse asc
    ]
    for order in warm_orders:
        if time.time() > deadline:
            break
        shuffled = order[:]
        random.shuffle(shuffled)
        chosen, sums = _greedy_order(shuffled, deadline)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best = set(chosen)
    # Also try pure greedy on sorted orders
    for order in warm_orders:
        if time.time() > deadline:
            break
        chosen, sums = _greedy_order(order, deadline)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best = set(chosen)

    # Main loop: random greedy + completion + local search
    while time.time() < deadline - 2:
        order = _ALL[:]
        random.shuffle(order)
        chosen, sums = _greedy_order(order, deadline - 2)
        # completion
        chosen, sums = _complete(chosen, sums, deadline - 1)
        # local search to escape local optima
        if time.time() < deadline - 3:
            chosen, sums = _local_search(chosen, sums, deadline - 3)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best = set(chosen)

    # Final polish pass on the best solution found
    if best and time.time() < deadline - 1:
        # Rebuild sums for best
        best_sums = {0}
        for x in best:
            inv = _INV[x]
            shifted = {s + inv for s in best_sums}
            best_sums = best_sums | shifted
        best, best_sums = _polish(best, best_sums, deadline)

    if not best:
        best = {1}

    return sorted(best)

# EVOLVE-BLOCK-END

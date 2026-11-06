"""Enhanced bitset subset-sum + systematic local search with constraint analysis."""
# EVOLVE-BLOCK-START
import time
import random

TARGET = 200
MASK = (1 << (TARGET + 1)) - 1  # bits 0..TARGET


def _creates_target(reachable, x):
    """True if adding x makes TARGET reachable (i.e., TARGET-x already reachable)."""
    if x > TARGET:
        return False
    return (reachable >> (TARGET - x)) & 1


def _add(reachable, x):
    return reachable | ((reachable << x) & MASK)


def _is_200_free(selected):
    reachable = 1  # bit 0
    for x in selected:
        if _creates_target(reachable, x):
            return False
        reachable = _add(reachable, x)
    return True


def _reachable_of(selected):
    reachable = 1
    for x in selected:
        reachable = _add(reachable, x)
    return reachable


def _greedy(order):
    selected = []
    reachable = 1
    for x in order:
        if not _creates_target(reachable, x):
            selected.append(x)
            reachable = _add(reachable, x)
    return selected, reachable


def _add_local(selected, reachable, all_elements):
    """Repeatedly add any missing element that is safe."""
    selected_set = set(selected)
    changed = True
    while changed:
        changed = False
        for x in all_elements:
            if x not in selected_set and not _creates_target(reachable, x):
                selected.append(x)
                selected_set.add(x)
                reachable = _add(reachable, x)
                changed = True
    return selected, reachable


def _add_local_pair(selected, reachable, all_elements):
    """Try to add pairs of missing elements simultaneously."""
    selected_set = set(selected)
    missing = [x for x in all_elements if x not in selected_set]
    n = len(missing)
    improved = False
    for i in range(n):
        for j in range(i + 1, n):
            x, y = missing[i], missing[j]
            r1 = reachable
            if not _creates_target(r1, x):
                r1 = _add(r1, x)
                if not _creates_target(r1, y):
                    selected.append(x)
                    selected.append(y)
                    selected_set.add(x)
                    selected_set.add(y)
                    reachable = _add(_add(reachable, x), y)
                    improved = True
                    break
        if improved:
            missing = [x for x in all_elements if x not in selected_set]
            n = len(missing)
            improved = False
            continue
    return selected, reachable


def _compute_blockers(reachable, all_elements, selected_set):
    """For each missing element, find which selected elements block it."""
    blockers = {}
    for x in all_elements:
        if x in selected_set:
            continue
        if x > TARGET:
            continue
        # x is blocked if TARGET-x is reachable
        if not ((reachable >> (TARGET - x)) & 1):
            continue
        # Find which selected elements contribute to making TARGET-x reachable
        # This is expensive; approximate by checking pairs
        blockers[x] = set()
    return blockers


def _k_swap_improve(selected, all_elements, k_max=3, deadline=None):
    """
    Try removing up to k_max elements then greedily re-adding; keep if larger.
    """
    best = list(selected)
    best_size = len(best)
    sel_list = list(selected)
    n = len(sel_list)
    
    # 1-swap: remove one element, re-add
    for i in range(n):
        if deadline and time.time() > deadline:
            break
        sel = [x for k, x in enumerate(sel_list) if k != i]
        reach = _reachable_of(sel)
        sel, reach = _add_local(sel, reach, all_elements)
        sel, reach = _add_local_pair(sel, reach, all_elements)
        if len(sel) > best_size:
            best, best_size = sel, len(sel)
            sel_list = list(best)
            n = len(sel_list)

    # 2-swap: remove two elements, re-add
    for i in range(n):
        if deadline and time.time() > deadline:
            break
        for j in range(i + 1, n):
            if deadline and time.time() > deadline:
                break
            sel = [x for k, x in enumerate(sel_list) if k != i and k != j]
            reach = _reachable_of(sel)
            sel, reach = _add_local(sel, reach, all_elements)
            sel, reach = _add_local_pair(sel, reach, all_elements)
            if len(sel) > best_size:
                best, best_size = sel, len(sel)
                sel_list = list(best)
                n = len(sel_list)
                break

    # 3-swap with random sampling
    if k_max >= 3 and (deadline is None or time.time() < deadline):
        rng_local = random.Random(42)
        for _ in range(500):
            if deadline and time.time() > deadline:
                break
            if n < 3:
                break
            idxs = rng_local.sample(range(n), 3)
            idx_set = set(idxs)
            sel = [x for k, x in enumerate(sel_list) if k not in idx_set]
            reach = _reachable_of(sel)
            sel, reach = _add_local(sel, reach, all_elements)
            sel, reach = _add_local_pair(sel, reach, all_elements)
            if len(sel) > best_size:
                best, best_size = sel, len(sel)
                sel_list = list(best)
                n = len(sel_list)

    return best


def _systematic_orders(all_elements):
    """Generate diverse systematic orders beyond just sorted/shuffled."""
    orders = []
    # Descending, ascending
    orders.append(sorted(all_elements, reverse=True))
    orders.append(sorted(all_elements))
    # Modulo-based orders for diversity
    for mod in [3, 5, 7, 11, 13, 17]:
        orders.append(sorted(all_elements, key=lambda x: x % mod))
        orders.append(sorted(all_elements, key=lambda x: -x % mod))
    # Group by size: small first, large first, alternating
    orders.append(sorted(all_elements, key=lambda x: abs(x - 50)))
    orders.append(sorted(all_elements, key=lambda x: -abs(x - 50)))
    # Pairs that sum to specific values
    for target_sum in [100, 150, 200, 250]:
        orders.append(sorted(all_elements, key=lambda x: abs(x - target_sum // 2)))
    return orders


def construct():
    start = time.time()
    deadline = start + 55.0  # 55s margin for 60s limit
    all_elements = list(range(1, 101))
    rng = random.Random(20240517)

    best = []
    best_size = 0

    def consider(sel):
        nonlocal best, best_size
        if len(sel) > best_size:
            best, best_size = list(sel), len(sel)

    # Phase 1: Systematic greedy constructions
    for o in _systematic_orders(all_elements):
        sel, reach = _greedy(o)
        sel, reach = _add_local(sel, reach, all_elements)
        consider(sel)
        if best_size >= 95:  # If we're close to optimal, we can be more aggressive
            break

    # Phase 2: Random restarts with greedy + local search
    iters = 0
    while time.time() < deadline - 10.0:
        o = all_elements[:]
        rng.shuffle(o)
        sel, reach = _greedy(o)
        sel, reach = _add_local(sel, reach, all_elements)
        sel, reach = _add_local_pair(sel, reach, all_elements)
        iters += 1
        consider(sel)
        if best_size >= 100:
            break

    # Phase 3: Swap-based improvement on best
    if time.time() < deadline - 5.0 and best:
        improved = _k_swap_improve(best, all_elements, k_max=3, deadline=deadline - 2.0)
        consider(improved)

    # Phase 4: More random restarts if we have time
    while time.time() < deadline - 3.0:
        o = all_elements[:]
        rng.shuffle(o)
        sel, reach = _greedy(o)
        sel, reach = _add_local(sel, reach, all_elements)
        sel, reach = _add_local_pair(sel, reach, all_elements)
        consider(sel)
        if best_size >= 100:
            break
        # Periodic swap improvement
        if iters % 20 == 0 and time.time() < deadline - 5.0:
            sel = _k_swap_improve(sel, all_elements, k_max=3, deadline=deadline - 3.0)
            consider(sel)

    # Final polish
    if time.time() < deadline - 1.0 and best:
        sel, reach = _add_local(list(best), _reachable_of(best), all_elements)
        sel, reach = _add_local_pair(sel, reach, all_elements)
        consider(sel)

    # Safety check
    if not best or not _is_200_free(best):
        # Fallback: known valid set
        # Elements 66-100: check if any subset sums to 200
        # 66+67+68=201, so three smallest sum to >200
        # Pairs: 66+100=166, 67+100=167, ..., 100+100 not allowed (distinct)
        # Actually max pair sum is 99+100=199 < 200
        # So {66,...,100} is valid! 35 elements.
        best = list(range(66, 101))

    return best
# EVOLVE-BLOCK-END

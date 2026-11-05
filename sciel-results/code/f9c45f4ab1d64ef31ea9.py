"""Bitmask-accelerated 4-AP-free set construction maximizing 3-APs."""
import time
import random

N = 256

def _precompute():
    """Precompute 4-AP and 3-AP data structures with bitmask tracking."""
    # All 4-APs: (a, a+d, a+2d, a+3d)
    four_aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    num_4aps = len(four_aps)

    # All 3-APs: (a, a+d, a+2d)
    three_aps = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))
    num_3aps = len(three_aps)

    # For each element (1-indexed), list of 4-AP indices it belongs to
    elem_to_4ap_idx = [[] for _ in range(N + 1)]
    for idx, ap in enumerate(four_aps):
        for x in ap:
            elem_to_4ap_idx[x].append(idx)

    # For each element, list of 3-AP indices it belongs to
    elem_to_3ap_idx = [[] for _ in range(N + 1)]
    for idx, ap in enumerate(three_aps):
        for x in ap:
            elem_to_3ap_idx[x].append(idx)

    # For each 4-AP, the set of elements (as a frozenset for quick check)
    four_ap_elems = [frozenset(ap) for ap in four_aps]

    # For each 3-AP, the set of elements
    three_ap_elems = [frozenset(ap) for ap in three_aps]

    return (four_aps, three_aps, elem_to_4ap_idx, elem_to_3ap_idx,
            four_ap_elems, three_ap_elems, num_4aps, num_3aps)

(_FOUR_APS, _THREE_APS, _ELEM_TO_4AP, _ELEM_TO_3AP,
 _FOUR_AP_ELEMS, _THREE_AP_ELEMS, _NUM_4APS, _NUM_3APS) = _precompute()


def _count_3aps(selected):
    """Count 3-APs fully contained in selected set."""
    count = 0
    for ap in _THREE_AP_ELEMS:
        if ap.issubset(selected):
            count += 1
    return count


def _has_4ap(selected):
    """Check if selected contains any 4-AP."""
    for ap in _FOUR_AP_ELEMS:
        if ap.issubset(selected):
            return True
    return False


def _greedy_construct():
    """Greedy: add elements in order if no 4-AP is created."""
    selected = set()
    # Track 4-AP counts for O(1) feasibility
    ap4_count = [0] * _NUM_4APS

    for x in range(1, N + 1):
        # Check if adding x creates a 4-AP
        creates = False
        for ap_idx in _ELEM_TO_4AP[x]:
            if ap4_count[ap_idx] == 3:  # other 3 elements already present
                creates = True
                break
        if not creates:
            selected.add(x)
            for ap_idx in _ELEM_TO_4AP[x]:
                ap4_count[ap_idx] += 1
    return selected


def _greedy_randomized(seed_offset=0):
    """Greedy with randomized element order."""
    rng = random.Random(seed_offset)
    order = list(range(1, N + 1))
    rng.shuffle(order)

    selected = set()
    ap4_count = [0] * _NUM_4APS

    for x in order:
        creates = False
        for ap_idx in _ELEM_TO_4AP[x]:
            if ap4_count[ap_idx] == 3:
                creates = True
                break
        if not creates:
            selected.add(x)
            for ap_idx in _ELEM_TO_4AP[x]:
                ap4_count[ap_idx] += 1
    return selected


def _delta_3aps_add(x, selected):
    """Compute how many new 3-APs are created when adding x to selected."""
    delta = 0
    for ap_idx in _ELEM_TO_3AP[x]:
        ap = _THREE_AP_ELEMS[ap_idx]
        # Check if the other 2 elements are in selected
        other = ap - {x}
        if other.issubset(selected):
            delta += 1
    return delta


def _delta_3aps_remove(x, selected):
    """Compute how many 3-APs are lost when removing x from selected."""
    delta = 0
    for ap_idx in _ELEM_TO_3AP[x]:
        ap = _THREE_AP_ELEMS[ap_idx]
        other = ap - {x}
        if other.issubset(selected):
            delta += 1
    return delta


def _creates_4ap(x, selected, ap4_count):
    """Check if adding x to selected would create a 4-AP."""
    for ap_idx in _ELEM_TO_4AP[x]:
        if ap4_count[ap_idx] == 3:
            return True
    return False


def _simulated_annealing(initial, time_limit=40.0, seed=42):
    """Simulated annealing with fast delta evaluation."""
    rng = random.Random(seed)
    selected = set(initial)
    ap4_count = [0] * _NUM_4APS
    for x in selected:
        for ap_idx in _ELEM_TO_4AP[x]:
            ap4_count[ap_idx] += 1

    current_score = _count_3aps(selected)
    best_selected = set(selected)
    best_score = current_score

    start_time = time.time()
    T = 5.0
    cooling = 0.9995
    min_T = 0.05

    # Precompute candidate additions (elements not in selected)
    all_elements = set(range(1, N + 1))

    while time.time() - start_time < time_limit:
        T *= cooling
        if T < min_T:
            T = min_T

        # Random move: add, remove, or swap
        move_type = rng.random()
        if move_type < 0.4:
            # Try to add a random element
            candidates = [x for x in all_elements if x not in selected]
            if not candidates:
                continue
            x = rng.choice(candidates)
            if _creates_4ap(x, selected, ap4_count):
                continue
            delta = _delta_3aps_add(x, selected)
            # Always accept improvement
            selected.add(x)
            for ap_idx in _ELEM_TO_4AP[x]:
                ap4_count[ap_idx] += 1
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best_selected = set(selected)

        elif move_type < 0.7:
            # Try to remove a random element
            if not selected:
                continue
            x = rng.choice(list(selected))
            delta = -_delta_3aps_remove(x, selected)
            # Accept with probability based on delta
            if delta >= 0 or rng.random() < _boltzmann(-delta, T):
                selected.discard(x)
                for ap_idx in _ELEM_TO_4AP[x]:
                    ap4_count[ap_idx] -= 1
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_selected = set(selected)
            else:
                # Revert (do nothing since we didn't modify)
                pass

        else:
            # Swap: remove one, add one
            if not selected:
                continue
            x = rng.choice(list(selected))
            candidates = [y for y in all_elements if y not in selected and y != x]
            if not candidates:
                continue
            y = rng.choice(candidates)
            # Check if removing x and adding y is valid
            # First check y doesn't create 4-AP with (selected - {x})
            temp_ap4 = list(ap4_count)
            for ap_idx in _ELEM_TO_4AP[x]:
                temp_ap4[ap_idx] -= 1
            valid = True
            for ap_idx in _ELEM_TO_4AP[y]:
                if temp_ap4[ap_idx] == 3:
                    valid = False
                    break
            if not valid:
                continue
            delta = _delta_3aps_add(y, selected - {x}) - _delta_3aps_remove(x, selected)
            # Note: _delta_3aps_add(y, selected-{x}) counts 3-APs where y completes
            # with elements in selected-{x}. But some 3-APs involving x and y might be
            # double-counted. Let's compute more carefully.
            # Actually, the net change in 3-AP count:
            # Lost: 3-APs that had x and 2 other elements in selected
            # Gained: 3-APs that have y and 2 other elements in (selected - {x})
            # But 3-APs with both x and y: if x,y,z form a 3-AP, removing x loses it,
            # adding y gains it back only if z is in selected-{x} (which it is if z != x).
            # So the simple delta formula should work.
            if delta >= 0 or rng.random() < _boltzmann(-delta, T):
                selected.discard(x)
                for ap_idx in _ELEM_TO_4AP[x]:
                    ap4_count[ap_idx] -= 1
                selected.add(y)
                for ap_idx in _ELEM_TO_4AP[y]:
                    ap4_count[ap_idx] += 1
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_selected = set(selected)

    return best_selected


def _boltzmann(delta, T):
    """Boltzmann acceptance probability for negative delta."""
    if delta >= 0:
        return 1.0
    if T < 1e-10:
        return 0.0
    import math
    return math.exp(delta / T)


def _local_search(selected, time_limit=15.0):
    """Local search: hill-climbing with add/swap moves."""
    selected = set(selected)
    ap4_count = [0] * _NUM_4APS
    for x in selected:
        for ap_idx in _ELEM_TO_4AP[x]:
            ap4_count[ap_idx] += 1

    current_score = _count_3aps(selected)
    best_selected = set(selected)
    best_score = current_score

    start_time = time.time()
    all_elements = set(range(1, N + 1))

    while time.time() - start_time < time_limit:
        improved = False

        # Try all single additions
        for x in sorted(all_elements - selected):
            if _creates_4ap(x, selected, ap4_count):
                continue
            delta = _delta_3aps_add(x, selected)
            if delta > 0:
                selected.add(x)
                for ap_idx in _ELEM_TO_4AP[x]:
                    ap4_count[ap_idx] += 1
                current_score += delta
                improved = True
                if current_score > best_score:
                    best_score = current_score
                    best_selected = set(selected)

        # Try all swaps
        for x in list(selected):
            for y in sorted(all_elements - selected):
                if y == x:
                    continue
                # Check validity
                temp_ap4 = list(ap4_count)
                for ap_idx in _ELEM_TO_4AP[x]:
                    temp_ap4[ap_idx] -= 1
                valid = True
                for ap_idx in _ELEM_TO_4AP[y]:
                    if temp_ap4[ap_idx] == 3:
                        valid = False
                        break
                if not valid:
                    continue
                delta = _delta_3aps_add(y, selected - {x}) - _delta_3aps_remove(x, selected)
                if delta > 0:
                    selected.discard(x)
                    for ap_idx in _ELEM_TO_4AP[x]:
                        ap4_count[ap_idx] -= 1
                    selected.add(y)
                    for ap_idx in _ELEM_TO_4AP[y]:
                        ap4_count[ap_idx] += 1
                    current_score += delta
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = set(selected)
                    break  # Restart swap search after one swap

        if not improved:
            break

    return best_selected


def construct():
    """Construct a subset of [1,256] with no 4-AP, maximizing 3-APs."""
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin

    # Phase 1: Multiple greedy constructions
    best = set()
    best_score = 0

    # Standard greedy
    g1 = _greedy_construct()
    s1 = _count_3aps(g1)
    if s1 > best_score:
        best, best_score = g1, s1

    # Randomized greedy (a few variants)
    for seed in [1, 7, 13, 42, 99]:
        if time.time() > deadline - 30.0:
            break
        g = _greedy_randomized(seed)
        s = _count_3aps(g)
        if s > best_score:
            best, best_score = g, s

    # Phase 2: Local search on best
    if time.time() < deadline - 25.0:
        ls_result = _local_search(best, time_limit=15.0)
        ls_score = _count_3aps(ls_result)
        if ls_score > best_score:
            best, best_score = ls_result, ls_score

    # Phase 3: Simulated annealing
    if time.time() < deadline - 15.0:
        sa_time = min(30.0, deadline - time.time() - 10.0)
        if sa_time > 5.0:
            sa_result = _simulated_annealing(best, time_limit=sa_time, seed=123)
            sa_score = _count_3aps(sa_result)
            if sa_score > best_score:
                best, best_score = sa_result, sa_score

    # Phase 4: Final local search
    if time.time() < deadline - 5.0:
        final_time = min(10.0, deadline - time.time() - 3.0)
        if final_time > 2.0:
            final_result = _local_search(best, time_limit=final_time)
            final_score = _count_3aps(final_result)
            if final_score > best_score:
                best, best_score = final_result, sa_score

    return sorted(best)

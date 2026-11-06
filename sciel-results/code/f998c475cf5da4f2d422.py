"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

MAX_SUM = 2000  # max possible sum of two elements from 1..1000


def _count_sums_array(chosen):
    """Count all pairwise sums a+b where a<=b from chosen. Returns list of length MAX_SUM+1."""
    counts = [0] * (MAX_SUM + 1)
    n = len(chosen)
    for i in range(n):
        xi = chosen[i]
        for j in range(i, n):
            s = xi + chosen[j]
            counts[s] += 1
    return counts


def _can_add_array(chosen, counts, x):
    """Check if x can be added to chosen without violating B2[2] using array counts."""
    for y in chosen:
        s = x + y
        if counts[s] >= 2:
            return False
    if counts[2 * x] >= 2:
        return False
    return True


def _add_to_counts(counts, chosen, x):
    """Update counts in-place after adding x to chosen (x not yet in chosen)."""
    for y in chosen:
        s = x + y
        counts[s] += 1
    counts[2 * x] += 1


def _remove_from_counts(counts, chosen, x):
    """Update counts in-place after removing x from chosen (x is in chosen)."""
    for y in chosen:
        if y != x:
            s = x + y
            counts[s] -= 1
    counts[2 * x] -= 1


def _greedy_construct(order):
    """Greedy construction following a specific order of candidates."""
    chosen = []
    counts = [0] * (MAX_SUM + 1)
    for x in order:
        if _can_add_array(chosen, counts, x):
            chosen.append(x)
            _add_to_counts(counts, chosen, x)
    return chosen


def _greedy_density_aware():
    """Greedy that at each step picks the candidate whose addition creates the fewest
    new sum collisions (soft cost = number of sums that would go from 1 to 2).
    Only candidates with no hard violations (sums already at 2) are considered.
    Ties broken by smaller x to leave room for larger elements."""
    chosen = []
    counts = [0] * (MAX_SUM + 1)
    remaining = set(range(1, 1001))

    while remaining:
        best_x = None
        best_soft = None
        # First pass: find candidates with no hard violations and fewest soft collisions
        for x in remaining:
            hard = False
            soft = 0
            for y in chosen:
                s = x + y
                c = counts[s]
                if c >= 2:
                    hard = True
                    break
                elif c == 1:
                    soft += 1
            if not hard:
                c2 = counts[2 * x]
                if c2 >= 2:
                    hard = True
                elif c2 == 1:
                    soft += 1
            if not hard:
                if best_x is None or soft < best_soft or (soft == best_soft and x < best_x):
                    best_x = x
                    best_soft = soft

        if best_x is None:
            break

        remaining.discard(best_x)
        chosen.append(best_x)
        _add_to_counts(counts, chosen, best_x)

    return sorted(chosen)


def _greedy_weighted_random(seed_val):
    """Greedy with a randomized priority: at each step, among all valid candidates,
    pick one with probability weighted by how 'spread out' it is (fewer collisions)."""
    random.seed(seed_val)
    chosen = []
    counts = [0] * (MAX_SUM + 1)
    remaining = set(range(1, 1001))

    while remaining:
        valid = []
        for x in remaining:
            ok = True
            for y in chosen:
                if counts[x + y] >= 2:
                    ok = False
                    break
            if ok and counts[2 * x] >= 2:
                ok = False
            if ok:
                valid.append(x)
        if not valid:
            break
        # Pick randomly from valid, but bias toward middle of range
        # Use a simple weighted selection: weight = 1.0 for all (uniform) for speed
        best_x = random.choice(valid)
        remaining.discard(best_x)
        chosen.append(best_x)
        _add_to_counts(counts, chosen, best_x)

    return chosen


def _local_improve_fast(chosen, deadline):
    """Fast local improvement using incremental count updates.
    Try single swaps: remove one element, add another that fits.
    Also try double swaps (remove two, add two) for escaping local optima."""
    chosen = sorted(chosen)
    counts = _count_sums_array(chosen)
    chosen_set = set(chosen)
    n = len(chosen)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Single-swap pass
        for i in range(n):
            if time.time() > deadline:
                break
            x_remove = chosen[i]
            # Remove x_remove
            _remove_from_counts(counts, chosen, x_remove)
            chosen_set.discard(x_remove)
            temp_chosen = chosen[:i] + chosen[i + 1:]

            best_add = None
            for x in range(1, 1001):
                if x in chosen_set:
                    continue
                if _can_add_array(temp_chosen, counts, x):
                    best_add = x
                    break

            if best_add is not None:
                # Accept the swap
                temp_chosen.append(best_add)
                _add_to_counts(counts, temp_chosen, best_add)
                chosen_set.add(best_add)
                chosen = sorted(temp_chosen)
                n = len(chosen)
                improved = True
            else:
                # Put x_remove back
                chosen_set.add(x_remove)
                chosen = sorted(temp_chosen + [x_remove])
                _add_to_counts(counts, chosen, x_remove)
                n = len(chosen)

        # Double-swap pass (remove two, add two) - try a limited number
        if time.time() < deadline:
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    x1 = chosen[i]
                    x2 = chosen[j]
                    # Remove both
                    _remove_from_counts(counts, chosen, x1)
                    _remove_from_counts(counts, chosen, x2)
                    chosen_set.discard(x1)
                    chosen_set.discard(x2)
                    temp_chosen = [c for c in chosen if c != x1 and c != x2]

                    # Find two candidates to add
                    added = []
                    for x in range(1, 1001):
                        if x in chosen_set:
                            continue
                        if _can_add_array(temp_chosen + added, counts, x):
                            added.append(x)
                            _add_to_counts(counts, temp_chosen + added[:-1], x)
                            chosen_set.add(x)
                            if len(added) == 2:
                                break

                    if len(added) == 2:
                        chosen = sorted(temp_chosen + added)
                        counts = _count_sums_array(chosen)
                        chosen_set = set(chosen)
                        n = len(chosen)
                        improved = True
                    else:
                        # Rollback
                        for x in added:
                            if x in chosen_set:
                                _remove_from_counts(counts, temp_chosen, x)
                                chosen_set.discard(x)
                        # Re-add x1, x2
                        chosen_set.add(x1)
                        chosen_set.add(x2)
                        chosen = sorted(temp_chosen + [x1, x2])
                        counts = _count_sums_array(chosen)
                        n = len(chosen)
                    if improved:
                        break
                if improved:
                    break

    return chosen


def _verify(chosen):
    """Verify that the set satisfies B2[2]: every sum has at most 2 representations."""
    counts = _count_sums_array(chosen)
    for s in range(2, MAX_SUM + 1):
        if counts[s] > 2:
            return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin

    best_set = []
    best_size = 0

    # Strategy 1: Ascending greedy (fast baseline)
    order1 = list(range(1, 1001))
    set1 = _greedy_construct(order1)
    if len(set1) > best_size:
        best_size = len(set1)
        best_set = set1

    if time.time() > deadline:
        return best_set

    # Strategy 2: Descending greedy
    order2 = list(range(1000, 0, -1))
    set2 = _greedy_construct(order2)
    if len(set2) > best_size:
        best_size = len(set2)
        best_set = set2

    if time.time() > deadline:
        return best_set

    # Strategy 3: Density-aware greedy (picks candidates with fewest collisions)
    set3 = _greedy_density_aware()
    if len(set3) > best_size:
        best_size = len(set3)
        best_set = set3

    if time.time() > deadline:
        return best_set

    # Strategy 4: Random shuffles - run many since array-based is fast
    random.seed(42)
    for trial in range(50):
        if time.time() > deadline - 60:
            break
        order = list(range(1, 1001))
        random.shuffle(order)
        s = _greedy_construct(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Strategy 5: Weighted random greedy
    for trial in range(20):
        if time.time() > deadline - 45:
            break
        s = _greedy_weighted_random(seed_val=1000 + trial)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Local improvement on the best set found
    if time.time() < deadline - 10:
        improved = _local_improve_fast(best_set, deadline - 5)
        if len(improved) > len(best_set):
            best_set = improved
            best_size = len(improved)

    # Final verification
    if best_set and not _verify(best_set):
        # Fallback to a known-valid construction
        best_set = _greedy_construct(list(range(1, 1001)))

    return best_set
# EVOLVE-BLOCK-END

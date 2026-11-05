"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    deadline = time.time() + 55.0  # Safety margin before 90s limit
    best_sets = ([], [])
    best_size = 0

    def _can_add(x, S, set_sums, set_diffs, other_diffs):
        """Check if x can be added to set S with given sums/diffs."""
        for a in S:
            if (a + x) in set_sums:
                return False
        if (x + x) in set_sums:
            return False
        for a in S:
            if (x - a) in other_diffs or (a - x) in other_diffs:
                return False
        return True

    def _do_add(x, side, sets, set_sums, set_diffs):
        """Actually add x to sets[side], updating sums/diffs incrementally."""
        S = sets[side]
        for a in S:
            set_sums[side].add(a + x)
            set_diffs[side].add(x - a)
            set_diffs[side].add(a - x)
        set_sums[side].add(x + x)
        sets[side].append(x)

    def _remove_and_rebuild(idx, side, sets, set_sums, set_diffs):
        """Remove element at idx from sets[side] and rebuild its sums/diffs."""
        sets[side].pop(idx)
        S = sets[side]
        new_sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                new_sums.add(S[i] + S[j])
        new_diffs = set()
        for i in range(len(S)):
            for j in range(len(S)):
                if i != j:
                    new_diffs.add(S[j] - S[i])
        set_sums[side] = new_sums
        set_diffs[side] = new_diffs

    def _rebuild_state(sets):
        """Rebuild set_sums and set_diffs from scratch."""
        set_sums = [set(), set()]
        set_diffs = [set(), set()]
        for side in (0, 1):
            S = sets[side]
            for i in range(len(S)):
                for j in range(i, len(S)):
                    set_sums[side].add(S[i] + S[j])
            for i in range(len(S)):
                for j in range(len(S)):
                    if i != j:
                        set_diffs[side].add(S[j] - S[i])
        return set_sums, set_diffs

    def _greedy_one_ordering(order, deadline, strategy="first_fit"):
        """Run greedy construction with a given ordering of candidates."""
        sets = [[], []]
        set_sums = [set(), set()]
        set_diffs = [set(), set()]

        for x in order:
            if time.time() > deadline:
                break

            if strategy == "first_fit":
                sides = [0, 1]
            elif strategy == "best_fit":
                conflicts = []
                for side in (0, 1):
                    S = sets[side]
                    other_diffs = set_diffs[1 - side]
                    conf = 0
                    for a in S:
                        if (a + x) in set_sums[side] or (x - a) in other_diffs or (a - x) in other_diffs:
                            conf += 1
                    conflicts.append(conf)
                sides = [0, 1] if conflicts[0] <= conflicts[1] else [1, 0]
            else:
                sides = [0, 1]
                random.shuffle(sides)

            for side in sides:
                S = sets[side]
                other_diffs = set_diffs[1 - side]
                if not _can_add(x, S, set_sums[side], set_diffs[side], other_diffs):
                    continue
                _do_add(x, side, sets, set_sums, set_diffs)
                break

        return sets

    def _local_search(sets, deadline, max_iter=200):
        """Local search: add elements, move elements between sets, or remove-and-reinsert."""
        sets = [list(sets[0]), list(sets[1])]
        set_sums, set_diffs = _rebuild_state(sets)

        current_size = len(sets[0]) + len(sets[1])
        best_size_ls = current_size
        best_sets_ls = [list(sets[0]), list(sets[1])]

        all_used = set(sets[0]) | set(sets[1])
        candidates = [x for x in range(1, 257) if x not in all_used]
        random.shuffle(candidates)

        iter_count = 0
        while iter_count < max_iter and time.time() < deadline - 1.0:
            iter_count += 1
            improved = False

            # Phase 1: try to add a candidate element
            for x in candidates:
                if time.time() > deadline - 1.0:
                    break
                for side in (0, 1):
                    if _can_add(x, sets[side], set_sums[side], set_diffs[side], set_diffs[1 - side]):
                        _do_add(x, side, sets, set_sums, set_diffs)
                        candidates.remove(x)
                        new_size = len(sets[0]) + len(sets[1])
                        if new_size > best_size_ls:
                            best_size_ls = new_size
                            best_sets_ls = [list(sets[0]), list(sets[1])]
                        improved = True
                        break
                if improved:
                    break

            if improved:
                continue

            # Phase 2: try to move an element from one set to the other
            moved = False
            for side in (0, 1):
                if len(sets[side]) <= 1:
                    continue
                for idx in range(len(sets[side])):
                    if time.time() > deadline - 1.0:
                        break
                    x = sets[side][idx]
                    _remove_and_rebuild(idx, side, sets, set_sums, set_diffs)
                    other = 1 - side
                    if _can_add(x, sets[other], set_sums[other], set_diffs[other], set_diffs[side]):
                        _do_add(x, other, sets, set_sums, set_diffs)
                        moved = True
                        break
                    else:
                        # Put it back
                        _do_add(x, side, sets, set_sums, set_diffs)
                if moved:
                    break

            if moved:
                continue

            # Phase 3: remove an element and try to reinsert it (possibly into the other set)
            # This helps escape local optima where the element blocks others
            removed = False
            for side in (0, 1):
                if len(sets[side]) <= 1:
                    continue
                for idx in range(len(sets[side])):
                    if time.time() > deadline - 1.0:
                        break
                    x = sets[side][idx]
                    _remove_and_rebuild(idx, side, sets, set_sums, set_diffs)

                    # Try to reinsert x into the other set
                    other = 1 - side
                    reinserted = False
                    if _can_add(x, sets[other], set_sums[other], set_diffs[other], set_diffs[side]):
                        _do_add(x, other, sets, set_sums, set_diffs)
                        reinserted = True
                    else:
                        # Try to reinsert x into the same set (may now fit after removal freed constraints? unlikely but check)
                        if _can_add(x, sets[side], set_sums[side], set_diffs[side], set_diffs[other]):
                            _do_add(x, side, sets, set_sums, set_diffs)
                            reinserted = True

                    if reinserted:
                        # Try to add some candidates now
                        for cx in candidates:
                            if time.time() > deadline - 1.0:
                                break
                            for s2 in (0, 1):
                                if _can_add(cx, sets[s2], set_sums[s2], set_diffs[s2], set_diffs[1 - s2]):
                                    _do_add(cx, s2, sets, set_sums, set_diffs)
                                    candidates.remove(cx)
                                    new_size = len(sets[0]) + len(sets[1])
                                    if new_size > best_size_ls:
                                        best_size_ls = new_size
                                        best_sets_ls = [list(sets[0]), list(sets[1])]
                                    removed = True
                                    break
                            if removed:
                                break
                        if not removed:
                            # No improvement, but the move itself may have changed structure
                            # Try adding more candidates
                            for cx in candidates:
                                if time.time() > deadline - 1.0:
                                    break
                                for s2 in (0, 1):
                                    if _can_add(cx, sets[s2], set_sums[s2], set_diffs[s2], set_diffs[1 - s2]):
                                        _do_add(cx, s2, sets, set_sums, set_diffs)
                                        candidates.remove(cx)
                                        new_size = len(sets[0]) + len(sets[1])
                                        if new_size > best_size_ls:
                                            best_size_ls = new_size
                                            best_sets_ls = [list(sets[0]), list(sets[1])]
                                        removed = True
                                        break
                                if removed:
                                    break
                        break

            if not removed:
                # Try a random restart of local search with a different candidate order
                random.shuffle(candidates)
                # If still stuck, try one more round
                if iter_count > max_iter // 2:
                    break

        return best_sets_ls

    # Try multiple strategies and orderings
    random.seed(42)
    base_order = list(range(1, 257))

    # Strategy 1: Natural order with first-fit
    sets = _greedy_one_ordering(base_order, deadline, "first_fit")
    size = len(sets[0]) + len(sets[1])
    if size > best_size:
        best_size = size
        best_sets = sets

    # Strategy 2: Natural order with best-fit
    sets = _greedy_one_ordering(base_order, deadline, "best_fit")
    size = len(sets[0]) + len(sets[1])
    if size > best_size:
        best_size = size
        best_sets = sets

    # Strategy 3: Descending order
    desc_order = list(range(256, 0, -1))
    sets = _greedy_one_ordering(desc_order, deadline, "first_fit")
    size = len(sets[0]) + len(sets[1])
    if size > best_size:
        best_size = size
        best_sets = sets

    # Strategy 4: Random permutations with different strategies
    strategies = ["first_fit", "best_fit", "random"]
    num_trials = 0
    while time.time() < deadline - 8.0:
        random_order = base_order[:]
        random.shuffle(random_order)
        strat = strategies[num_trials % len(strategies)]

        sets = _greedy_one_ordering(random_order, deadline, strat)
        size = len(sets[0]) + len(sets[1])
        if size > best_size:
            best_size = size
            best_sets = sets

        num_trials += 1

    # Local search polish on the best solution
    if best_size > 0 and time.time() < deadline - 3.0:
        polished = _local_search(best_sets, deadline, max_iter=1000)
        polished_size = len(polished[0]) + len(polished[1])
        if polished_size > best_size:
            best_size = polished_size
            best_sets = polished

    # Ensure nonempty
    if not best_sets[0] and not best_sets[1]:
        best_sets = ([1], [2])
    elif not best_sets[0]:
        best_sets[0] = [1]
    elif not best_sets[1]:
        best_sets[1] = [1]

    return best_sets
# EVOLVE-BLOCK-END

import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit (evaluator kills at 90s)
    n = 256
    max_per_set = 64

    def get_diffs(S):
        """Get set of nonzero differences for set S."""
        diffs = set()
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                diffs.add(S[i] - S[j])
                diffs.add(S[j] - S[i])
        return diffs

    def get_sums(S):
        """Get set of pair sums (including a+a) for set S."""
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        return sums

    def is_valid_addition(x, S, S_sums, S_diffs, other_diffs):
        """Check if x can be added to S maintaining Sidon property and disjoint diffs."""
        # Check Sidon: all pair sums including x+x must be unique
        for a in S:
            s = x + a
            if s in S_sums:
                return False
        if 2 * x in S_sums:
            return False

        # Check differences don't conflict with other set
        for a in S:
            d1 = x - a
            d2 = a - x
            if d1 in other_diffs or d2 in other_diffs:
                return False

        return True

    def greedy_construct(order, max_time=None):
        """Greedy construction given a specific element order."""
        sets = [[], []]
        sets_sums = [set(), set()]
        sets_diffs = [set(), set()]

        for x in order:
            if max_time is not None and time.time() > max_time:
                break
            best_side = -1
            for side in (0, 1):
                if is_valid_addition(x, sets[side], sets_sums[side], sets_diffs[side], sets_diffs[1 - side]):
                    best_side = side
                    break
            if best_side >= 0:
                x_side = best_side
                # Update sums
                for a in sets[x_side]:
                    sets_sums[x_side].add(x + a)
                sets_sums[x_side].add(2 * x)
                # Update diffs
                for a in sets[x_side]:
                    sets_diffs[x_side].add(x - a)
                    sets_diffs[x_side].add(a - x)
                sets[x_side].append(x)

        return sets

    def validate(sets):
        """Validate that both sets are Sidon and their nonzero differences are disjoint."""
        if not sets[0] or not sets[1]:
            return False
        if len(sets[0]) > max_per_set or len(sets[1]) > max_per_set:
            return False

        # Check Sidon property for each set
        for S in sets:
            sums = set()
            for i in range(len(S)):
                for j in range(i, len(S)):
                    s = S[i] + S[j]
                    if s in sums:
                        return False
                    sums.add(s)

        # Check disjoint nonzero differences
        diffs0 = get_diffs(sets[0])
        diffs1 = get_diffs(sets[1])
        if diffs0 & diffs1:
            return False

        return True

    def local_search_add(sets, max_time=None):
        """Try to add elements from [1,256] that aren't in either set."""
        if max_time is None:
            max_time = deadline - 3
        improved = True
        while improved and time.time() < max_time:
            improved = False
            all_elements = set(range(1, n + 1))
            in_sets = set(sets[0]) | set(sets[1])
            candidates = list(all_elements - in_sets)
            random.shuffle(candidates)

            for x in candidates:
                if time.time() > max_time:
                    break
                for side in (0, 1):
                    S = sets[side]
                    if len(S) >= max_per_set:
                        continue
                    S_sums = get_sums(S)
                    S_diffs = get_diffs(S)
                    other_diffs = get_diffs(sets[1 - side])

                    if is_valid_addition(x, S, S_sums, S_diffs, other_diffs):
                        sets[side].append(x)
                        improved = True
                        break

        return sets

    def local_search_swap(sets, max_time=None):
        """Try to swap elements between sets or remove/add to improve."""
        if max_time is None:
            max_time = deadline - 3
        improved = True
        iterations = 0
        while improved and time.time() < max_time and iterations < 50:
            improved = False
            iterations += 1

            # Try removing an element from one set and adding a different one
            for side in (0, 1):
                for i in range(len(sets[side])):
                    if time.time() > max_time:
                        break
                    removed = sets[side][i]
                    new_set = sets[side][:]
                    del new_set[i]

                    # Try adding candidates to the modified set
                    S_sums = get_sums(new_set)
                    S_diffs = get_diffs(new_set)
                    other_diffs = get_diffs(sets[1 - side])

                    # Try to add removed element back (no-op) or find better placement
                    # Actually, try to add other elements that were previously blocked
                    all_elements = set(range(1, n + 1))
                    in_both = set(sets[0]) | set(sets[1])
                    candidates = [x for x in in_both if x != removed]
                    random.shuffle(candidates)

                    best_add = None
                    for x in candidates:
                        if x in new_set:
                            continue
                        if is_valid_addition(x, new_set, S_sums, S_diffs, other_diffs):
                            # Check if this improves total size
                            # We're replacing removed with x, so size stays same
                            # Only do this if it frees up something useful
                            best_add = x
                            break

                    if best_add is not None and best_add != removed:
                        # Actually, swapping doesn't help size. Let's skip this.
                        pass

        return sets

    def hill_climb(sets, max_time=None):
        """More sophisticated local search: try all possible additions and some swaps."""
        if max_time is None:
            max_time = deadline - 3

        # First, try pure additions
        sets = local_search_add(sets, max_time)

        # Try to remove one element from a set and add two others (net +1)
        for side in (0, 1):
            if time.time() > max_time:
                break
            for i in range(len(sets[side])):
                if time.time() > max_time:
                    break
                removed = sets[side][i]
                new_set = sets[side][:]
                del new_set[i]
                new_set.sort()

                S_sums = get_sums(new_set)
                S_diffs = get_diffs(new_set)
                other_diffs = get_diffs(sets[1 - side])

                # Try to add multiple elements to new_set
                added = []
                all_elements = set(range(1, n + 1))
                in_other = set(sets[1 - side])
                candidates = list(all_elements - in_other - set(new_set))
                random.shuffle(candidates)

                for x in candidates:
                    if len(new_set) + len(added) >= max_per_set:
                        break
                    if is_valid_addition(x, new_set, S_sums, S_diffs, other_diffs):
                        new_set.append(x)
                        for a in new_set[:-1]:
                            S_sums.add(x + a)
                        S_sums.add(2 * x)
                        for a in new_set[:-1]:
                            S_diffs.add(x - a)
                            S_diffs.add(a - x)
                        added.append(x)

                if len(new_set) > len(sets[side]):
                    sets[side] = new_set

        return sets

    # Strategy: Try multiple orderings and keep the best valid result
    best_sets = [[], []]
    best_size = 0

    random.seed(42)

    # Try several different orderings
    orderings = []
    # Natural order
    orderings.append(list(range(1, n + 1)))
    # Reverse order
    orderings.append(list(range(n, 0, -1)))
    # Random orderings
    for _ in range(5):
        order = list(range(1, n + 1))
        random.shuffle(order)
        orderings.append(order)
    # Weighted orderings (favor certain ranges)
    for _ in range(3):
        order = list(range(1, n + 1))
        # Bias towards smaller numbers
        order.sort(key=lambda x: x * random.random())
        orderings.append(order)
    for _ in range(3):
        order = list(range(1, n + 1))
        # Bias towards larger numbers
        order.sort(key=lambda x: (n + 1 - x) * random.random())
        orderings.append(order)

    for order in orderings:
        if time.time() > deadline - 10:
            break
        sets = greedy_construct(order, deadline - 8)
        if validate(sets):
            sets = hill_climb(sets, deadline - 5)
            if validate(sets):
                size = len(sets[0]) + len(sets[1])
                if size > best_size:
                    best_size = size
                    best_sets = [list(sets[0]), list(sets[1])]

    # If we didn't find a good solution, use a simple fallback
    if not best_sets[0] or not best_sets[1]:
        sets = greedy_construct(list(range(1, n + 1)))
        if validate(sets):
            best_sets = [list(sets[0]), list(sets[1])]
        else:
            best_sets = [[1], [2]]

    return best_sets

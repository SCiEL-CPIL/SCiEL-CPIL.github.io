"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
from itertools import combinations


def construct():
    start_time = time.time()
    timeout = 118  # Safety margin for 120s limit

    def instance(size, seed, limit=10000):
        from random import Random
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def can_add_element(x, s):
        """Check if x can be added to sum-free set s without creating conflicts."""
        if 2 * x in s:
            return False
        for y in s:
            if y != x:
                if x + y in s:
                    return False
                if x - y in s:
                    return False
        return True

    def get_conflicts(x, s):
        """Return set of elements in s that conflict with x."""
        conflicts = set()
        if 2 * x in s:
            conflicts.add(2 * x)
        for y in s:
            if y != x:
                if x + y in s:
                    conflicts.add(x + y)
                if x - y in s:
                    conflicts.add(x - y)
        return conflicts

    def greedy_sum_free(elements, order_key=None):
        """
        Greedy construction with multiple strategies.
        """
        if not elements:
            return []

        pos = [x for x in elements if x > 0]
        neg = [x for x in elements if x < 0]

        max_pos = max(pos) if pos else 0
        min_neg = min(neg) if neg else 0

        # Strategy 1: High magnitude candidates
        candidates = []
        for x in pos:
            if 2 * x > max_pos:
                candidates.append(x)
        for x in neg:
            if 2 * x < min_neg:
                candidates.append(x)

        selected_set = set(candidates)

        # Resolve conflicts
        def has_conflict(s):
            for a in s:
                if 2 * a in s:
                    return True
                for b in s:
                    if b != a and a + b in s:
                        return True
            return False

        while has_conflict(selected_set):
            removed = False
            for a in list(selected_set):
                if 2 * a in selected_set:
                    selected_set.remove(2 * a)
                    removed = True
                    break
                for b in selected_set:
                    if b != a:
                        s_val = a + b
                        if s_val in selected_set:
                            selected_set.remove(s_val)
                            removed = True
                            break
                if removed:
                    break
            if not removed:
                break

        remaining = [x for x in elements if x not in selected_set]
        if order_key:
            remaining.sort(key=order_key)
        else:
            remaining.sort(key=lambda x: -abs(x))

        for x in remaining:
            if can_add_element(x, selected_set):
                selected_set.add(x)

        return list(selected_set)

    def remove_and_retry_improve(elements, initial_subset, max_rounds=100):
        """
        Remove-and-retry: for each element in the subset, try removing it and
        then greedily adding back as many elements as possible.
        """
        s = set(initial_subset)
        remaining = [x for x in elements if x not in s]

        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1

            best_gain = 0
            best_new_set = None

            for x in list(s):
                s_no_x = set(s)
                s_no_x.discard(x)

                candidates = [y for y in remaining if y not in s_no_x]
                candidates.append(x)
                candidates.sort(key=lambda z: -abs(z))

                s_candidate = set(s_no_x)
                for y in candidates:
                    if can_add_element(y, s_candidate):
                        s_candidate.add(y)

                gain = len(s_candidate) - len(s)
                if gain > best_gain:
                    best_gain = gain
                    best_new_set = s_candidate

            if best_gain > 0 and best_new_set is not None:
                s = best_new_set
                remaining = [x for x in elements if x not in s]
                improved = True

        return list(s)

    def simulated_annealing_improve(elements, initial_subset, time_budget=10):
        """
        Simulated annealing with better move generation.
        """
        rng = random.Random(42)
        s = set(initial_subset)
        remaining = [x for x in elements if x not in s]

        best_s = set(s)
        best_size = len(s)

        T = 1.0
        T_min = 0.001
        cooling = 0.99

        end_time = time.time() + time_budget

        while time.time() < end_time:
            move_type = rng.random()
            new_s = set(s)

            if move_type < 0.25 and remaining:
                # Add a random element from remaining
                idx = rng.randrange(len(remaining))
                x = remaining[idx]
                if can_add_element(x, new_s):
                    new_s.add(x)
                else:
                    conflicts = get_conflicts(x, new_s)
                    if conflicts:
                        # Remove a random conflict
                        c = list(conflicts)[rng.randrange(len(conflicts))]
                        new_s.discard(c)
                        if can_add_element(x, new_s):
                            new_s.add(x)
            elif move_type < 0.45 and s:
                # Remove a random element
                x = list(s)[rng.randrange(len(s))]
                new_s.discard(x)
            elif move_type < 0.7 and s and remaining:
                # Swap: remove one, try to add another
                x = list(s)[rng.randrange(len(s))]
                new_s.discard(x)
                for _ in range(min(5, len(remaining))):
                    idx = rng.randrange(len(remaining))
                    y = remaining[idx]
                    if can_add_element(y, new_s):
                        new_s.add(y)
                        break
            else:
                # 2-opt style: remove two, add up to two
                if len(s) >= 2 and len(remaining) >= 1:
                    x1 = list(s)[rng.randrange(len(s))]
                    new_s.discard(x1)
                    x2 = list(s)[rng.randrange(len(s))] if len(s) > 1 else None
                    if x2 is not None:
                        new_s.discard(x2)
                    for _ in range(min(3, len(remaining))):
                        idx = rng.randrange(len(remaining))
                        y = remaining[idx]
                        if can_add_element(y, new_s):
                            new_s.add(y)
                            break

            old_size = len(s)
            new_size = len(new_s)
            delta = new_size - old_size

            if delta > 0 or (delta == 0 and rng.random() < 0.5):
                s = new_s
                remaining = [x for x in elements if x not in s]
                if new_size > best_size:
                    best_size = new_size
                    best_s = set(s)
            elif delta < 0:
                if rng.random() < math.exp(delta / T):
                    s = new_s
                    remaining = [x for x in elements if x not in s]

            T = max(T_min, T * cooling)

        return list(best_s)

    def multi_start_local_search(elements, time_budget=20):
        """
        Run multiple greedy starts with different orderings, then improve.
        """
        orderings = [
            lambda x: -abs(x),
            lambda x: x,
            lambda x: -x,
            lambda x: abs(x),
            lambda x: (0 if x > 0 else 1, -abs(x)),
            lambda x: (1 if x > 0 else 0, abs(x)),
            lambda x: (0 if x > 0 else 1, abs(x)),
            lambda x: (1 if x > 0 else 0, -abs(x)),
        ]

        best_subset = None
        best_size = -1

        per_order_budget = time_budget / (len(orderings) + 3)

        for order in orderings:
            if time.time() - start_time > timeout:
                break
            t0 = time.time()
            subset = greedy_sum_free(elements, order_key=order)

            # Improve with remove-and-retry
            if time.time() - t0 < per_order_budget * 0.4:
                improved = remove_and_retry_improve(elements, subset, max_rounds=30)
                if len(improved) > len(subset):
                    subset = improved

            if len(subset) > best_size:
                best_size = len(subset)
                best_subset = subset

        # Final polish with simulated annealing on the best
        if best_subset is not None:
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time > 5:
                sa_budget = min(remaining_time - 2, 20)
                if sa_budget > 2:
                    polished = simulated_annealing_improve(elements, best_subset, time_budget=sa_budget)
                    if len(polished) > best_size:
                        best_subset = polished
                        best_size = len(polished)

        return best_subset if best_subset is not None else []

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() - start_time > timeout:
            results.append([])
            continue

        elems = instance(size, seed)

        # Allocate time proportionally to problem size
        if size <= 60:
            time_budget = 12
        elif size <= 120:
            time_budget = 22
        else:
            time_budget = 38

        result = multi_start_local_search(elems, time_budget=time_budget)
        results.append(result if result is not None else [])

    return results
# EVOLVE-BLOCK-END

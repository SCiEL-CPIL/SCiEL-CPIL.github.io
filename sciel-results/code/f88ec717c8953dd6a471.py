import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def build_conflicts(S):
        """
        For each element, compute the set of elements it conflicts with.
        x conflicts with y if there exists z in S such that x+y=z, x+z=y, or y+z=x.
        Equivalently, x and y can't both be in a sum-free set if x+y in S, or x-y in S, or y-x in S.
        """
        S_set = set(S)
        conflicts = {}
        for x in S:
            c = set()
            for y in S:
                if x == y:
                    # x+x = 2x: if 2x in S, x conflicts with itself (can't include x)
                    # but for pairwise, we track distinct
                    if 2 * x in S_set:
                        c.add(x)  # self-conflict marker
                else:
                    if x + y in S_set:
                        c.add(y)
                    if x - y in S_set:
                        c.add(y)
                    if y - x in S_set:
                        c.add(y)
            conflicts[x] = c
        return conflicts

    def try_add(x, S_set):
        """Check if x can be added to S_set while keeping it sum-free."""
        if 2 * x in S_set:
            return False
        for a in S_set:
            if x + a in S_set:
                return False
            if x - a in S_set:
                return False
        return True

    def greedy_build(S, order, S_set_all):
        """Build a sum-free set by processing elements in given order."""
        current = []
        current_set = set()
        for x in order:
            if try_add(x, current_set):
                current.append(x)
                current_set.add(x)
        return current

    def local_search(S, initial, time_limit, S_set_all):
        """
        Local search with add and 1-swap moves, using conflict precomputation for speed.
        """
        current = list(initial)
        current_set = set(current)
        best = list(current)
        best_size = len(current)
        no_improve = 0
        max_no_improve = 50

        while time.time() < time_limit:
            improved = False

            # 1. Try to add any element (greedy pass, then random)
            candidates = [x for x in S if x not in current_set]
            # Sort by number of conflicts with current (fewer conflicts = easier to add)
            # Actually, just try all, prefer those with fewer "blocked" reasons
            random.shuffle(candidates)
            for add in candidates:
                if try_add(add, current_set):
                    current.append(add)
                    current_set.add(add)
                    if len(current) > best_size:
                        best = list(current)
                        best_size = len(current)
                    improved = True
                    no_improve = 0
                    break

            if improved:
                continue

            # 2. Try 1-swap: remove one, add one (net 0, but may unlock adds)
            # Try 2-swap: remove two, add two (net 0)
            # For efficiency, try 1-swap first
            rem_candidates = list(current)
            random.shuffle(rem_candidates)

            for rem in rem_candidates:
                if time.time() > time_limit:
                    break
                trial_set = current_set.copy()
                trial_set.discard(rem)

                add_candidates = [x for x in S if x not in trial_set]
                random.shuffle(add_candidates)

                for add in add_candidates:
                    if try_add(add, trial_set):
                        # Accept swap if it allows an immediate add
                        current = [x for x in current if x != rem]
                        current.append(add)
                        current_set = set(current)
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break

            if not improved:
                # Try 2-swap: remove two, add two
                if len(current) >= 2:
                    rem_pairs = []
                    for i in range(len(current)):
                        for j in range(i + 1, len(current)):
                            rem_pairs.append((current[i], current[j]))
                    random.shuffle(rem_pairs)
                    # Limit pairs to try
                    for rem1, rem2 in rem_pairs[:20]:
                        if time.time() > time_limit:
                            break
                        trial_set = current_set.copy()
                        trial_set.discard(rem1)
                        trial_set.discard(rem2)

                        add_candidates = [x for x in S if x not in trial_set]
                        random.shuffle(add_candidates)

                        added = []
                        for add in add_candidates:
                            if len(added) == 2:
                                break
                            if try_add(add, trial_set | set(added)):
                                added.append(add)

                        if len(added) == 2:
                            current = [x for x in current if x != rem1 and x != rem2]
                            current.extend(added)
                            current_set = set(current)
                            improved = True
                            no_improve = 0
                            break

            if not improved:
                no_improve += 1
                if no_improve > max_no_improve:
                    break

        return best

    def solve_instance(S, time_limit):
        n = len(S)
        S_set_all = set(S)
        best = []
        best_size = 0

        # Structural priors
        # 1. Odds
        odds = [x for x in S if x % 2 != 0]
        if len(odds) > best_size:
            best = odds
            best_size = len(odds)

        # 2. Positives upper half
        pos = sorted([x for x in S if x > 0])
        if pos:
            half = len(pos) // 2
            pos_large = pos[half:]
            if len(pos_large) > best_size:
                best = pos_large
                best_size = len(pos_large)

        # 3. Negatives lower half
        neg = sorted([x for x in S if x < 0])
        if neg:
            half = len(neg) // 2
            neg_small = neg[:half]
            if len(neg_small) > best_size:
                best = neg_small
                best_size = len(neg_small)

        # 4. Greedy from multiple diverse orderings
        greedy_time = min(3.0, (time_limit - time.time()) * 0.2)
        greedy_deadline = time.time() + greedy_time

        while time.time() < greedy_deadline:
            # Different orderings
            strategies = [
                list(S),  # sorted
                list(reversed(S)),
                sorted(S, key=lambda x: abs(x)),
                sorted(S, key=lambda x: -abs(x)),
            ]
            for strat in strategies:
                if time.time() > greedy_deadline:
                    break
                order = list(strat)
                random.shuffle(order)
                current = greedy_build(S, order, S_set_all)
                if len(current) > best_size:
                    best = current
                    best_size = len(current)

        # 5. Local search from best and diverse starts
        remaining = time_limit - time.time()
        if remaining > 1.0:
            # Local search from best
            improved = local_search(S, best, time.time() + remaining * 0.5, S_set_all)
            if len(improved) > best_size:
                best = improved
                best_size = len(improved)

            # Local search from odds
            time_left = time_limit - time.time()
            if time_left > 2.0 and len(odds) > 0:
                improved = local_search(S, odds, time.time() + time_left * 0.5, S_set_all)
                if len(improved) > best_size:
                    best = improved
                    best_size = len(improved)

            # Local search from pos_large
            time_left = time_limit - time.time()
            if time_left > 2.0 and pos:
                half = len(pos) // 2
                pos_large = pos[half:]
                improved = local_search(S, pos_large, time.time() + time_left * 0.5, S_set_all)
                if len(improved) > best_size:
                    best = improved
                    best_size = len(improved)

        return best

    # Solve all three instances
    results = []
    instances = [(60, 801), (120, 802), (240, 803)]

    # Time allocation: proportional to size, with minimum
    # 60: 20%, 120: 30%, 240: 50%
    fractions = [0.2, 0.3, 0.5]

    for i, (size, seed) in enumerate(instances):
        S = instance(size, seed)
        time_budget = DEADLINE - (time.time() - start_time)
        instance_time = start_time + time_budget * fractions[i]
        # Ensure minimum
        if instance_time < time.time() + 3.0:
            instance_time = time.time() + 3.0
        # Ensure not past deadline
        if instance_time > start_time + DEADLINE - 1.0:
            instance_time = start_time + DEADLINE - 1.0

        result = solve_instance(S, instance_time)
        results.append(result)

        # Check time
        if time.time() > start_time + DEADLINE - 2.0:
            break

    # Fill missing with quick greedy (odds)
    while len(results) < len(instances):
        size, seed = instances[len(results)]
        S = instance(size, seed)
        result = [x for x in S if x % 2 != 0]
        results.append(result)

    # Final verification
    for i, result in enumerate(results):
        if not is_sum_free(result):
            S = instance(instances[i][0], instances[i][1])
            results[i] = [x for x in S if x % 2 != 0]

    return results

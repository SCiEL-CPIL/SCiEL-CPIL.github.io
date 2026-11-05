def construct():
    import time
    import random

    start_time = time.time()
    DEADLINE = 55.0  # Safety margin

    N = 180

    # Precompute all forbidden triples
    conflict_pairs = {i: [] for i in range(1, N + 1)}

    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            diff = b - a
            num = a * b
            if num % diff != 0:
                continue
            c = num // diff
            if c > b and c <= N:
                conflict_pairs[a].append((b, c))
                conflict_pairs[b].append((a, c))
                conflict_pairs[c].append((a, b))

    def solve_trial(seed_val):
        random.seed(seed_val)
        in_set = [False] * (N + 1)
        # Initialize with high range [91, 180]
        result = set(range(91, N + 1))
        for i in result:
            in_set[i] = True

        def can_add(x):
            """Check if x can be added to current set defined by in_set."""
            for (u, v) in conflict_pairs[x]:
                if in_set[u] and in_set[v]:
                    return False
            return True

        # Greedy addition: try adding x in random order
        candidates = list(range(90, 0, -1))
        random.shuffle(candidates)

        for x in candidates:
            if time.time() - start_time > DEADLINE:
                return result
            if can_add(x):
                in_set[x] = True
                result.add(x)

        # Local Search
        improved = True
        while improved and time.time() - start_time < DEADLINE:
            improved = False

            # 1. Try to add any y not in set
            add_cands = [y for y in range(1, N + 1) if not in_set[y]]
            random.shuffle(add_cands)
            for y in add_cands:
                if time.time() - start_time > DEADLINE:
                    return result
                if can_add(y):
                    in_set[y] = True
                    result.add(y)
                    improved = True

            if not improved:
                # 2. Try swaps: Remove x, Add y's
                remove_cands = list(result)
                random.shuffle(remove_cands)

                for x in remove_cands:
                    if time.time() - start_time > DEADLINE:
                        return result

                    # Remove x
                    in_set[x] = False
                    result.discard(x)

                    added_count = 0
                    # Try to add elements from 1 to N that are not in set
                    add_cands_swap = [y for y in range(1, N + 1) if not in_set[y]]
                    random.shuffle(add_cands_swap)

                    for y in add_cands_swap:
                        if in_set[y]:
                            continue
                        if can_add(y):
                            in_set[y] = True
                            result.add(y)
                            added_count += 1
                            # Stop early if we have a net gain of at least 1 (added >= 2)
                            # to avoid spending too much time on a single swap
                            if added_count >= 2:
                                break

                    if added_count == 0:
                        # Restore x
                        in_set[x] = True
                        result.add(x)
                    else:
                        # added_count >= 1
                        # If added_count == 1, size is unchanged (swap).
                        # If added_count >= 2, size increased.
                        # Mark improved to trigger another pass or continue swapping.
                        improved = True

        return result

    best_result = set()
    best_size = 0

    # Run multiple trials
    while time.time() - start_time < DEADLINE:
        seed_val = int(time.time() * 1000000) % (2**31)
        current_result = solve_trial(seed_val)
        current_size = len(current_result)

        if current_size > best_size:
            best_size = current_size
            best_result = current_result

    return list(best_result)

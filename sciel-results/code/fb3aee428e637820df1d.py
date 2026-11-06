"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    import time
    import random
    from math import gcd

    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill

    # Denominators are 2,3,...,48 -> indices 0..46
    n = 47
    denominators = list(range(2, 49))

    # Compute LCM of 2..48 for exact integer arithmetic
    def lcm(a, b):
        return a // gcd(a, b) * b

    L = 1
    for d in denominators:
        L = lcm(L, d)

    # Integer weights: weight[i] = L // denominators[i]
    weights = [L // d for d in denominators]
    target = L  # we need sum of weights < target

    def sum_of(selected_set):
        return sum(weights[i] for i in selected_set)

    def best_improvement(seed_set, time_budget):
        """Run best-improvement local search from seed_set.
        Returns (best_set, best_sum)."""
        current = set(seed_set)
        current_sum = sum_of(current)

        while (time.time() - start_time) < time_budget:
            best_move_sum = current_sum
            best_move = None  # ('add', j) or ('swap', i, j)

            # Neighborhood: add and swap
            # Add
            for j in range(n):
                if j in current:
                    continue
                ns = current_sum + weights[j]
                if ns < target and ns > best_move_sum:
                    best_move_sum = ns
                    best_move = ('add', j)

            # Swap
            for i in current:
                for j in range(n):
                    if j in current:
                        continue
                    if weights[j] <= weights[i]:
                        continue
                    ns = current_sum - weights[i] + weights[j]
                    if ns < target and ns > best_move_sum:
                        best_move_sum = ns
                        best_move = ('swap', i, j)

            if best_move is None:
                break  # local optimum

            if best_move[0] == 'add':
                current.add(best_move[1])
            else:
                current.discard(best_move[2])
                current.add(best_move[2])
                # best_move = ('swap', i, j): remove i, add j
                # fix: best_move[1]=i, best_move[2]=j
                current.discard(best_move[1])
                current.add(best_move[2])

            current_sum = best_move_sum

        return current, current_sum

    # Generate diverse seeds
    def greedy_skip(k_skip):
        """Greedy: add smallest denominators (largest weights) first,
        but skip the k_skip-th smallest denominator to create a different basin."""
        # Sort by weight descending (denominator ascending)
        order = sorted(range(n), key=lambda i: weights[i], reverse=True)
        # order[0] is largest weight (denom 2), order[1] denom 3, etc.
        selected = set()
        s = 0
        skip_index = k_skip  # skip this position in the order
        for pos, i in enumerate(order):
            if pos == skip_index:
                continue
            if s + weights[i] < target:
                selected.add(i)
                s += weights[i]
        return selected

    # Collect candidate seeds
    seeds = []
    # Plain greedy (skip nothing)
    seeds.append(greedy_skip(-1))  # -1 means skip none (pos never == -1)
    # Greedy skipping the k-th smallest denominator, for various k
    for k in range(1, 12):
        seeds.append(greedy_skip(k))

    # Also a few random-ish seeds: take the greedy set and randomly swap a few
    base = seeds[0]
    rng = random.Random(12345)
    for r in range(8):
        s = set(base)
        cur_sum = sum_of(s)
        # Do a few random improving moves
        for _ in range(5):
            # pick a random unselected
            unselected = [x for x in range(n) if x not in s]
            if not unselected:
                break
            j = rng.choice(unselected)
            # try to add
            if cur_sum + weights[j] < target:
                s.add(j)
                cur_sum += weights[j]
            elif s:
                # try swap
                i = rng.choice(list(s))
                if weights[j] > weights[i] and cur_sum - weights[i] + weights[j] < target:
                    s.discard(i)
                    s.add(j)
                    cur_sum = cur_sum - weights[i] + weights[j]
        seeds.append(s)

    # Deduplicate seeds
    seen = set()
    unique_seeds = []
    for s in seeds:
        key = frozenset(s)
        if key not in seen:
            seen.add(key)
            unique_seeds.append(s)

    # Run best-improvement on each seed, keep global best
    global_best_set = set()
    global_best_sum = -1

    # Allocate time budget across restarts
    # First pass: quick budget per restart
    n_restarts = len(unique_seeds)
    per_restart_budget = (deadline - 2.0) / max(n_restarts, 1)

    for idx, seed in enumerate(unique_seeds):
        if (time.time() - start_time) > deadline - 2.0:
            break
        # Give each restart a portion of the remaining time
        remaining = deadline - 2.0 - (time.time() - start_time)
        budget = min(per_restart_budget, remaining)
        s, ssum = best_improvement(seed, start_time + budget)
        if ssum > global_best_sum:
            global_best_sum = ssum
            global_best_set = s

    # Second pass: re-run best-improvement on the global best with more time
    # (polish)
    if (time.time() - start_time) < deadline - 3.0:
        s, ssum = best_improvement(global_best_set, deadline - 1.0)
        if ssum > global_best_sum:
            global_best_sum = ssum
            global_best_set = s

    # Verify
    assert len(global_best_set) > 0, "Must select at least one item"
    total = sum(weights[i] for i in global_best_set)
    assert total < target, f"Sum {total} not strictly below target {target}"

    return list(global_best_set)
# EVOLVE-BLOCK-END

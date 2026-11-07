import time

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # Safety margin before 90s kill

    # Primes up to 40
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    n_primes = len(primes)
    p_index = {p: i for i, p in enumerate(primes)}

    # Precompute prime exponent vectors for each number 2..40
    vectors = {}
    for x in range(2, 41):
        v = [0] * n_primes
        temp = x
        for i, p in enumerate(primes):
            while temp % p == 0:
                v[i] += 1
                temp //= p
        # For numbers <= 40, all prime factors are in our list
        vectors[x] = tuple(v)

    def is_dissociated(selected):
        """Check if all subset sums of vectors are distinct (dissociated set)."""
        if not selected:
            return True
        sums = {tuple([0] * n_primes)}
        for x in selected:
            v = vectors[x]
            new_sums = set()
            for s in sums:
                ns = tuple(s[i] + v[i] for i in range(n_primes))
                if ns in sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        return True

    def can_add(current_list, candidate):
        """Check if candidate can be added to current_list without collision."""
        if candidate in current_list:
            return False
        # Build current sums
        sums = {tuple([0] * n_primes)}
        for x in current_list:
            v = vectors[x]
            new_sums = set()
            for s in sums:
                ns = tuple(s[i] + v[i] for i in range(n_primes))
                new_sums.add(ns)
            sums.update(new_sums)
        # Check if adding candidate causes collision
        v = vectors[candidate]
        for s in sums:
            ns = tuple(s[i] + v[i] for i in range(n_primes))
            if ns in sums:
                return False
        return True

    # Greedy construction: start with primes, try adding composites
    primes_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    current = list(primes_list)

    # Sort composites by number of distinct prime factors (simpler first)
    def num_distinct_prime_factors(n):
        count = 0
        for p in primes:
            if n % p == 0:
                count += 1
        return count

    composites = [x for x in range(2, 41) if x not in primes_list]
    composites.sort(key=lambda x: (num_distinct_prime_factors(x), x))

    for c in composites:
        if time.time() - start_time > DEADLINE:
            break
        if can_add(current, c):
            current.append(c)

    best_size = len(current)
    best_set = list(current)

    # Backtracking search to try to find larger sets
    candidates = list(range(2, 41))

    def dfs(idx, current_set, current_sums, depth):
        nonlocal best_size, best_set, start_time
        if time.time() - start_time > DEADLINE:
            return

        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            if best_size >= 18:
                return

        if idx == len(candidates):
            return

        # Try including candidates[idx]
        x = candidates[idx]
        v = vectors[x]

        # Check if x can be added
        valid = True
        new_sums_check = []
        for s in current_sums:
            ns = tuple(s[i] + v[i] for i in range(n_primes))
            if ns in current_sums:
                valid = False
                break
            new_sums_check.append(ns)

        if valid:
            new_sums = current_sums | set(new_sums_check)
            dfs(idx + 1, current_set + [x], new_sums, depth + 1)

        # Try excluding candidates[idx]
        dfs(idx + 1, current_set, current_sums, depth)

    # Run DFS with a time-limited approach
    # Use the greedy result as initial best
    # Sort candidates to help pruning: primes first (more likely to be included)
    def sort_key(x):
        # Primes first, then by value
        if x in primes_list:
            return (0, x)
        return (1, x)

    sorted_candidates = sorted(candidates, key=sort_key)

    initial_sums = {tuple([0] * n_primes)}
    dfs(0, [], initial_sums, 0)

    # If we found a better set, use it
    if len(best_set) > len(current):
        current = best_set

    # Final verification
    if not is_dissociated(current):
        # Fallback to primes only
        current = list(primes_list)

    return [int(x) for x in current]

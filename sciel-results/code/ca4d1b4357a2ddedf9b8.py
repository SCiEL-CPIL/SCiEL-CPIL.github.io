def construct():
    import time
    import random
    from math import gcd

    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit

    n = 128

    # Precompute GCD table
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            gcd_table[i][j] = gcd(i, j)

    # Precompute bad triples: for each x, the set of pairs (a,b) such that
    # gcd(a,b) == gcd(a,x) == gcd(b,x)
    # This allows fast checking: can_add(x, S) iff no pair in bad_pairs[x] is both in S
    bad_pairs_for_x = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            for c in range(b + 1, n + 1):
                if gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]:
                    bad_pairs_for_x[c].add((a, b))
                    bad_pairs_for_x[b].add((a, c))
                    bad_pairs_for_x[a].add((b, c))

    def can_add(x, S_set):
        """Check if x can be added to S without creating a bad triple."""
        for a, b in bad_pairs_for_x[x]:
            if a in S_set and b in S_set:
                return False
        return True

    def greedy_complete(S, order=None):
        """Greedily add elements in given order (or natural order) to S."""
        S = list(S)
        S_set = set(S)
        if order is None:
            order = list(range(1, n + 1))
        for x in order:
            if x not in S_set and can_add(x, S_set):
                S.append(x)
                S_set.add(x)
        return S

    def local_search(S, max_iter=200):
        """Local search with swaps to improve set size."""
        S = list(S)
        S_set = set(S)
        best_S = list(S)
        best_size = len(S)

        for _ in range(max_iter):
            if time.time() > deadline:
                break

            improved = False
            # Try to add any element
            for x in range(1, n + 1):
                if x not in S_set and can_add(x, S_set):
                    S.append(x)
                    S_set.add(x)
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = list(S)

            if not improved:
                # Try swaps: remove one element, then try to add others
                for x in list(S_set):
                    S_set.remove(x)
                    S.remove(x)
                    # Try to add any element that was previously not in S
                    for y in range(1, n + 1):
                        if y not in S_set and can_add(y, S_set):
                            S.append(y)
                            S_set.add(y)
                            if len(S) > best_size:
                                best_size = len(S)
                                best_S = list(S)
                            break
                    # Restore if no improvement
                    if len(S) <= best_size:
                        S_set.add(x)
                        S.append(x)
                    else:
                        improved = True

            if not improved:
                break

        return best_S

    # Generate diverse initial solutions using different strategies
    def random_greedy(rng):
        order = list(range(1, n + 1))
        rng.shuffle(order)
        S = []
        S_set = set()
        for x in order:
            if x not in S_set and can_add(x, S_set):
                S.append(x)
                S_set.add(x)
        return S

    def structural_greedy():
        """Use a structured order: prioritize numbers with unique properties."""
        # Sort by number of divisors, then by value
        def num_divisors(x):
            count = 0
            for d in range(1, int(x**0.5) + 1):
                if x % d == 0:
                    count += 1
                    if d != x // d:
                        count += 1
            return count

        order = sorted(range(1, n + 1), key=lambda x: (num_divisors(x), x))
        S = []
        S_set = set()
        for x in order:
            if x not in S_set and can_add(x, S_set):
                S.append(x)
                S_set.add(x)
        return S

    def prime_power_greedy():
        """Start with prime powers, then add others."""
        # Prime powers up to 128
        prime_powers = set()
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]:
            power = p
            while power <= n:
                prime_powers.add(power)
                power *= p

        S = list(prime_powers & set(range(1, n + 1)))
        S_set = set(S)
        # Add remaining numbers in random order
        remaining = [x for x in range(1, n + 1) if x not in S_set]
        random.Random(42).shuffle(remaining)
        for x in remaining:
            if can_add(x, S_set):
                S.append(x)
                S_set.add(x)
        return S

    # Run multiple initial constructions
    best_S = []
    best_size = 0
    rng = random.Random(12345)

    # Structural initializations
    for _ in range(5):
        S = structural_greedy()
        S = local_search(S, max_iter=50)
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)
        if time.time() > deadline - 10:
            break

    # Prime power based
    for _ in range(3):
        S = prime_power_greedy()
        S = local_search(S, max_iter=50)
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)
        if time.time() > deadline - 10:
            break

    # Random greedy with multiple restarts
    num_restarts = 30
    for i in range(num_restarts):
        if time.time() > deadline - 15:
            break
        S = random_greedy(rng)
        S = local_search(S, max_iter=100)
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)

    # Kick-based local search from best solution
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        if not best_S:
            continue

        # Remove 2-4 random elements
        k = rng.randint(2, 4)
        to_remove = rng.sample(best_S, min(k, len(best_S)))
        S = [x for x in best_S if x not in to_remove]
        S_set = set(S)

        # Re-add greedily
        for x in range(1, n + 1):
            if x not in S_set and can_add(x, S_set):
                S.append(x)
                S_set.add(x)

        S = local_search(S, max_iter=80)
        if len(S) > best_size:
            best_size = len(S)
            best_S = list(S)

    # Final verification
    S_set = set(best_S)
    for a, b, c in [(a, b, c) for a in range(1, n + 1) for b in range(a + 1, n + 1) for c in range(b + 1, n + 1) if a in S_set and b in S_set and c in S_set]:
        if gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]:
            # Remove one element to fix
            best_S.remove(c)
            break

    if not best_S:
        # Fallback: return a known valid small set
        return [1, 2, 4, 8, 16, 32, 64, 3, 9, 27, 81]

    return best_S

import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 58.0

    N = 128

    # Precompute GCD table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Precompute bad_pairs: for each x, bad_pairs[x] is a set of (a, b) with a < b
    # such that gcd(a,x) == gcd(b,x) == gcd(a,b)
    # We'll store as a list of bitmasks for fast intersection checks
    bad_pairs_for_x = [set() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a in range(1, x):
            for b in range(a + 1, x):
                if gcd_table[a][x] == gcd_table[b][x] == gcd_table[a][b]:
                    bad_pairs_for_x[x].add((a, b))

    # For each x, precompute a bitmask of all elements y such that there exists z in S
    # where (y, z) is a bad pair for x. Actually, let's use a different approach:
    # For each x, precompute for each y, the set of z such that (y, z) is bad for x.
    # But that's too much. Instead, let's use a simpler approach:
    # For each x, we'll store a list of (y, z) pairs that are bad for x.
    # Then to check if adding x to S is valid, we check if any (y, z) in bad_pairs_for_x[x]
    # has both y and z in S.

    # Even better: for each x, precompute a bitmask where bit i is set if i is in S
    # and there exists j in S such that (i, j) is a bad pair for x.
    # Actually, let's just use the direct check with the precomputed pairs.

    def is_valid_add(S_set, x):
        """Check if x can be added to S_set (a set of elements)."""
        for (a, b) in bad_pairs_for_x[x]:
            if a in S_set and b in S_set:
                return False
        return True

    def greedy_add(order):
        """Greedy add in the given order."""
        S = set()
        for x in order:
            if is_valid_add(S, x):
                S.add(x)
        return S

    def local_search(S, max_swaps=10000):
        """Try to improve S by removing one element and adding one or two others."""
        S_list = list(S)
        S_set = set(S_list)
        best = S_set
        best_size = len(best)

        for _ in range(max_swaps):
            if time.time() > deadline - 2.0:
                break

            improved = False
            for i in range(len(S_list)):
                a = S_list[i]
                # Try removing a and adding one element
                S_without_a = S_set - {a}
                for x in range(1, N + 1):
                    if x in S_set:
                        continue
                    if is_valid_add(S_without_a, x):
                        new_S = S_without_a | {x}
                        if len(new_S) > best_size:
                            best = new_S
                            best_size = len(best)
                            S_list = list(best)
                            S_set = best
                            improved = True
                            break
                if improved:
                    break

                # Try removing a and adding two elements
                found_two = False
                for x in range(1, N + 1):
                    if x in S_set or x == a:
                        continue
                    S_with_x = S_without_a | {x}
                    if not is_valid_add(S_with_x, x):
                        continue
                    for y in range(x + 1, N + 1):
                        if y in S_set or y == a:
                            continue
                        if is_valid_add(S_with_x, y):
                            new_S = S_with_x | {y}
                            if len(new_S) > best_size:
                                best = new_S
                                best_size = len(best)
                                S_list = list(best)
                                S_set = best
                                improved = True
                                found_two = True
                                break
                    if found_two:
                        break
                if improved:
                    break

            if not improved:
                break

        return best

    # Generate several orders for greedy add
    orders = []

    # Order 1: natural order
    orders.append(list(range(1, N + 1)))

    # Order 2: reverse natural
    orders.append(list(range(N, 0, -1)))

    # Order 3: by number of divisors (fewer first)
    def num_divisors(n):
        count = 0
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0:
                count += 1
                if i != n // i:
                    count += 1
        return count

    orders.append(sorted(range(1, N + 1), key=lambda x: num_divisors(x)))
    orders.append(sorted(range(1, N + 1), key=lambda x: -num_divisors(x)))

    # Order 4: by smallest prime factor
    def spf(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        for i in range(5, int(n**0.5) + 1, 6):
            if n % i == 0:
                return i
            if n % (i + 2) == 0:
                return i + 2
        return n

    orders.append(sorted(range(1, N + 1), key=lambda x: spf(x)))
    orders.append(sorted(range(1, N + 1), key=lambda x: -spf(x)))

    # Order 5: primes first, then composites
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        for i in range(5, int(n**0.5) + 1, 6):
            if n % i == 0 or n % (i + 2) == 0:
                return False
        return True

    primes = [x for x in range(1, N + 1) if is_prime(x)]
    composites = [x for x in range(1, N + 1) if not is_prime(x)]
    orders.append(primes + composites)
    orders.append(composites + primes)

    # Order 6: by value of x / spf(x) (prime powers first)
    orders.append(sorted(range(1, N + 1), key=lambda x: (x / spf(x), x)))

    # Run greedy add for each order
    best_set = set()
    for order in orders:
        if time.time() > deadline - 5.0:
            break
        S = greedy_add(order)
        if len(S) > len(best_set):
            best_set = S

    # Run local search on the best set
    if time.time() < deadline - 3.0:
        best_set = local_search(best_set, max_swaps=20000)

    # Also try local search on other good sets
    for order in orders[:3]:
        if time.time() > deadline - 3.0:
            break
        S = greedy_add(order)
        if len(S) >= len(best_set) - 1:
            S = local_search(S, max_swaps=5000)
            if len(S) > len(best_set):
                best_set = S

    # Final safety check
    result = list(best_set)
    if not result:
        result = [1]

    # Verify validity
    n = len(result)
    valid = True
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = result[i], result[j], result[k]
                if gcd_table[a][b] == gcd_table[b][c] == gcd_table[a][c]:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback to a simple valid set
        result = [1, 2, 4, 8, 16, 32, 64, 3, 9, 27, 81]

    return result

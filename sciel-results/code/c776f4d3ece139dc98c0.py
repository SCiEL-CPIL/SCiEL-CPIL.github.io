import math
import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill

    def check_pair(r1, m1, r2, m2):
        g = math.gcd(m1, m2)
        return (r1 - r2) % g != 0

    def check_all(classes):
        n = len(classes)
        for i in range(n):
            for j in range(i + 1, n):
                if not check_pair(classes[i][0], classes[i][1], classes[j][0], classes[j][1]):
                    return False
        return True

    # Seed solution: powers of 2
    seed = [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]
    best = seed
    best_count = len(seed)

    def solve_for_prime(p, time_limit=20.0):
        """Find maximum set of pairwise disjoint residue classes using moduli all divisible by p."""
        moduli = [m for m in range(2 * p, 129, p)]
        n = len(moduli)
        if n == 0:
            return []

        # Precompute gcd matrix
        gcd_mat = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                gcd_mat[i][j] = math.gcd(moduli[i], moduli[j])
                gcd_mat[j][i] = gcd_mat[i][j]

        # Variable ordering: most constrained first (highest max gcd)
        def max_gcd_score(idx):
            return max(gcd_mat[idx][j] for j in range(n) if j != idx) if n > 1 else 0

        var_order = sorted(range(n), key=max_gcd_score, reverse=True)

        t0 = time.time()
        assignment = [None] * n

        def bt(pos):
            if time.time() - t0 > time_limit:
                return False
            if pos == n:
                return True
            idx = var_order[pos]
            m = moduli[idx]

            # Try residues: prefer those that differ from already-assigned mod gcd
            for r in range(m):
                ok = True
                for prev_pos in range(pos):
                    pidx = var_order[prev_pos]
                    if assignment[pidx] is not None:
                        g = gcd_mat[idx][pidx]
                        if (r - assignment[pidx]) % g == 0:
                            ok = False
                            break
                if ok:
                    assignment[idx] = r
                    if bt(pos + 1):
                        return True
                    assignment[idx] = None

            # Skip this modulus
            assignment[idx] = None
            return bt(pos + 1)

        bt(0)

        result = []
        for i in range(n):
            if assignment[i] is not None:
                result.append([assignment[i], moduli[i]])
        return result

    # Try primes 2, 3, 5, 7, 11, 13
    for p in [2, 3, 5, 7, 11, 13]:
        if time.time() > deadline - 5:
            break
        remaining = deadline - time.time() - 3
        if remaining < 2:
            break
        result = solve_for_prime(p, time_limit=min(remaining, 20.0))
        if len(result) > best_count:
            best = result
            best_count = len(result)

    # Also try a greedy + local search approach on all moduli
    if time.time() < deadline - 10:
        all_moduli = list(range(2, 129))

        def greedy_build(moduli_order, time_limit=3.0):
            t0 = time.time()
            selected = []
            for m in moduli_order:
                if time.time() - t0 > time_limit:
                    break
                for r in range(m):
                    if all(check_pair(r, m, r2, m2) for r2, m2 in selected):
                        selected.append([r, m])
                        break
            return selected

        # Order by descending number of divisors (more shared factors)
        def num_divisors(n):
            count = 0
            for i in range(1, int(n**0.5) + 1):
                if n % i == 0:
                    count += 1
                    if i != n // i:
                        count += 1
            return count

        orderings = [
            sorted(all_moduli, key=lambda x: -num_divisors(x)),
            sorted(all_moduli, reverse=True),
            sorted(all_moduli),
        ]

        for order in orderings:
            if time.time() > deadline - 5:
                break
            result = greedy_build(order, time_limit=min(2.0, deadline - time.time() - 3))
            if len(result) > best_count:
                best = result
                best_count = len(result)

        # Random restarts
        random.seed(42)
        for attempt in range(300):
            if time.time() > deadline - 5:
                break
            shuffled = list(all_moduli)
            random.shuffle(shuffled)
            result = greedy_build(shuffled, time_limit=1.0)
            if len(result) > best_count:
                best = result
                best_count = len(result)

    # Final verification
    if not check_all(best):
        best = seed
        best_count = len(seed)

    return best

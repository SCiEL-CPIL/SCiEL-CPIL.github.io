import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # 5s margin before 90s kill
    MAX_MOD = 128

    def _time_left():
        return DEADLINE - (time.time() - start_time)

    def _is_disjoint(r1, m1, r2, m2):
        g = math.gcd(m1, m2)
        return (r1 - r2) % g != 0

    def _try_add(r, m, existing):
        for er, em in existing:
            if not _is_disjoint(r, m, er, em):
                return False
        return True

    def _greedy(ordering, residue_order_fn):
        selected = []
        for m in ordering:
            if _time_left() < 2.0:
                break
            residues = residue_order_fn(m)
            for r in residues:
                if _try_add(r, m, selected):
                    selected.append((r, m))
                    break
        return selected

    def _local_search(selected, time_limit):
        """1-for-2 and 1-for-1 local search."""
        start = time.time()
        current = list(selected)
        improved = True
        while improved and (time.time() - start) < time_limit:
            improved = False
            current_size = len(current)
            used_moduli = set(m for _, m in current)
            avail_moduli = [m for m in range(2, MAX_MOD + 1) if m not in used_moduli]

            # Try 1-for-2: remove one, add two
            for i in range(len(current)):
                if (time.time() - start) >= time_limit:
                    break
                removed = current[i]
                remaining = current[:i] + current[i+1:]
                rem_moduli = set(m for _, m in remaining)
                avail = [m for m in range(2, MAX_MOD + 1) if m not in rem_moduli]

                # Find all (r,m) that can be added to remaining
                candidates = []
                for m in avail:
                    for r in range(m):
                        if _try_add(r, m, remaining):
                            candidates.append((r, m))

                # Try pairs
                found = False
                for j in range(len(candidates)):
                    if (time.time() - start) >= time_limit:
                        break
                    c1 = candidates[j]
                    for k in range(j + 1, len(candidates)):
                        c2 = candidates[k]
                        if not _is_disjoint(c1[0], c1[1], c2[0], c2[1]):
                            continue
                        new_current = remaining + [c1, c2]
                        if len(new_current) > current_size:
                            current = new_current
                            current_size = len(current)
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if improved:
                    break

            # Also try 1-for-1: remove one, add one (if it helps with future 1-for-2)
            if not improved:
                for i in range(len(current)):
                    if (time.time() - start) >= time_limit:
                        break
                    removed = current[i]
                    remaining = current[:i] + current[i+1:]
                    rem_moduli = set(m for _, m in remaining)
                    avail = [m for m in range(2, MAX_MOD + 1) if m not in rem_moduli]
                    best_add = None
                    for m in avail:
                        for r in range(m):
                            if _try_add(r, m, remaining):
                                best_add = (r, m)
                                break
                        if best_add:
                            break
                    if best_add:
                        new_current = remaining + [best_add]
                        # Only accept if it enables future improvements (heuristic: always accept same size)
                        # Actually for 1-for-1, size stays same, so only useful as intermediate
                        # Skip for now to avoid cycles
                        pass

        return current

    # Generate orderings
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True

    def num_prime_factors(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    all_moduli = list(range(2, MAX_MOD + 1))
    orderings = []

    # 1. Natural order
    orderings.append(all_moduli[:])

    # 2. Reverse order
    orderings.append(all_moduli[::-1])

    # 3. Primes first
    primes = [m for m in all_moduli if is_prime(m)]
    composites = [m for m in all_moduli if not is_prime(m)]
    orderings.append(primes + composites)

    # 4. By number of prime factors ascending
    by_factors = sorted(all_moduli, key=lambda x: (num_prime_factors(x), x))
    orderings.append(by_factors)

    # 5. By number of prime factors descending
    by_factors_desc = sorted(all_moduli, key=lambda x: (-num_prime_factors(x), x))
    orderings.append(by_factors_desc)

    # 6. Even moduli first
    evens = [m for m in all_moduli if m % 2 == 0]
    odds = [m for m in all_moduli if m % 2 != 0]
    orderings.append(evens + odds)
    orderings.append(odds + evens)

    # 7. Random shuffles
    random.seed(42)
    for _ in range(30):
        shuffled = all_moduli[:]
        random.shuffle(shuffled)
        orderings.append(shuffled)

    # Residue ordering functions
    def residues_asc(m):
        return list(range(m))

    def residues_desc(m):
        return list(range(m - 1, -1, -1))

    def residues_rand(m):
        rs = list(range(m))
        random.shuffle(rs)
        return rs

    # Run greedy with multiple orderings and residue orders
    best = []
    best_size = 0

    for ordering in orderings:
        if _time_left() < 3.0:
            break
        for res_fn in [residues_asc, residues_desc]:
            if _time_left() < 2.0:
                break
            result = _greedy(ordering, res_fn)
            if len(result) > best_size:
                best = result
                best_size = len(result)

    # Randomized greedy with random residue orders
    for _ in range(50):
        if _time_left() < 3.0:
            break
        shuffled = all_moduli[:]
        random.shuffle(shuffled)
        result = _greedy(shuffled, residues_rand)
        if len(result) > best_size:
            best = result
            best_size = len(result)

    # Local search on the best solution
    if _time_left() > 5.0 and best_size > 0:
        time_for_ls = min(_time_left() - 3.0, 40.0)
        improved = _local_search(best, time_for_ls)
        if len(improved) > best_size:
            best = improved
            best_size = len(best)

    # Additional local search passes
    if _time_left() > 5.0 and best_size > 0:
        time_for_ls2 = min(_time_left() - 3.0, 20.0)
        improved2 = _local_search(best, time_for_ls2)
        if len(improved2) > best_size:
            best = improved2
            best_size = len(best)

    # Convert to result format
    result = [[r, m] for r, m in best]

    # Ensure nonempty
    if not result:
        return [[(1 << k) - 1, 1 << (k + 1)] for k in range(7)]

    return result

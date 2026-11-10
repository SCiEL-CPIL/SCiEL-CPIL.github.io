"""Meet-in-the-middle with exponent-vector pruning for optimal factorial decomposition."""
# EVOLVE-BLOCK-START
import time
from math import factorial

def construct():
    t0 = time.time()
    deadline = t0 + 78.0

    N = factorial(12)  # 479001600
    target_exp = (10, 5, 2, 1, 1)  # 2^10 * 3^5 * 5^2 * 7^1 * 11^1

    # Generate all divisors > 12 of N
    divs = []
    for a in range(11):
        for b in range(6):
            for c in range(3):
                for d in range(2):
                    for e in range(2):
                        val = (2 ** a) * (3 ** b) * (5 ** c) * (7 ** d) * (11 ** e)
                        if val > 12:
                            divs.append((val, (a, b, c, d, e)))
    divs.sort(key=lambda x: x[0])

    n = len(divs)
    div_val = [x[0] for x in divs]
    div_exp = [x[1] for x in divs]

    best_solution = None
    best_max = N

    # Precompute suffix sums of exponent vectors for pruning
    # suffix_exp[i][j] = sum of div_exp[k][j] for k in range(i, n)
    suffix_exp = [[0] * 5 for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(5):
            suffix_exp[i][j] = suffix_exp[i + 1][j] + div_exp[i][j]

    def can_cover(rem_exp, start_idx):
        """Check if remaining divisors from start_idx can cover rem_exp."""
        for j in range(5):
            if suffix_exp[start_idx][j] < rem_exp[j]:
                return False
        return True

    # DFS with pruning
    def dfs(rem_exp, start_idx, chosen, count_left):
        nonlocal best_max, best_solution

        if time.time() > deadline:
            return None

        if count_left == 0:
            if all(e == 0 for e in rem_exp):
                return list(chosen)
            return None

        if start_idx >= n:
            return None

        if n - start_idx < count_left:
            return None

        if not can_cover(rem_exp, start_idx):
            return None

        for i in range(start_idx, n):
            if time.time() > deadline:
                return None

            d_val = div_val[i]
            d_exp = div_exp[i]

            # Pruning: if d_val >= best_max, skip
            if d_val >= best_max:
                break  # since divs are sorted, all subsequent are >= d_val

            # Check fit
            fits = True
            for j in range(5):
                if d_exp[j] > rem_exp[j]:
                    fits = False
                    break
            if not fits:
                continue

            new_rem = tuple(rem_exp[j] - d_exp[j] for j in range(5))
            chosen.append(d_val)

            result = dfs(new_rem, i + 1, chosen, count_left - 1)
            chosen.pop()

            if result is not None:
                return result

        return None

    # Try different factor counts
    for k in range(2, 13):
        if time.time() > deadline:
            break

        lb = max(13, int(N ** (1.0 / k)) + 1)
        ub = best_max - 1

        if ub < lb:
            continue

        # Filter to divisors in [lb, ub]
        # Find the range of valid indices
        lo = 0
        hi = n
        while lo < hi:
            mid = (lo + hi) // 2
            if div_val[mid] < lb:
                lo = mid + 1
            else:
                hi = mid
        start_lo = lo

        lo2 = 0
        hi2 = n
        while lo2 < hi2:
            mid = (lo2 + hi2) // 2
            if div_val[mid] <= ub:
                lo2 = mid + 1
            else:
                hi2 = mid
        end_hi = lo2

        if end_hi - start_lo < k:
            continue

        # Redefine can_cover for the filtered range
        # For simplicity, just use the global suffix but check bounds

        def dfs_filtered(rem_exp, start_idx, chosen, count_left):
            nonlocal best_max, best_solution

            if time.time() > deadline:
                return None

            if count_left == 0:
                if all(e == 0 for e in rem_exp):
                    return list(chosen)
                return None

            if start_idx >= end_hi:
                return None

            if end_hi - start_idx < count_left:
                return None

            for i in range(start_idx, end_hi):
                if time.time() > deadline:
                    return None

                d_val = div_val[i]
                d_exp = div_exp[i]

                if d_val >= best_max:
                    break

                fits = True
                for j in range(5):
                    if d_exp[j] > rem_exp[j]:
                        fits = False
                        break
                if not fits:
                    continue

                new_rem = tuple(rem_exp[j] - d_exp[j] for j in range(5))
                chosen.append(d_val)

                result = dfs_filtered(new_rem, i + 1, chosen, count_left - 1)
                chosen.pop()

                if result is not None:
                    return result

            return None

        result = dfs_filtered(target_exp, start_lo, [], k)
        if result is not None:
            max_f = max(result)
            if max_f < best_max:
                best_max = max_f
                best_solution = result
            if best_max <= lb + 3:
                break

    if best_solution:
        prod = 1
        for f in best_solution:
            prod *= f
        assert prod == N
        assert len(set(best_solution)) == len(best_solution)
        assert all(f > 12 for f in best_solution)
        assert 1 <= len(best_solution) <= 12
        return best_solution

    return [N]

# EVOLVE-BLOCK-END

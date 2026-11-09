import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit

    def dfs_exact(rem_sum, min_d, remaining_terms, current_list):
        if time.time() > deadline:
            return None

        if remaining_terms == 0:
            if rem_sum == 0:
                return current_list
            return None

        if remaining_terms == 1:
            if rem_sum > 0 and rem_sum.numerator == 1:
                d = rem_sum.denominator
                if d >= min_d and d >= 5 and d <= 10000000:
                    return current_list + [d]
            return None

        # Pruning: Max possible sum with remaining_terms terms starting from min_d
        max_possible_sum = Fraction(0, 1)
        for i in range(remaining_terms):
            max_possible_sum += Fraction(1, min_d + i)
        if max_possible_sum < rem_sum:
            return None

        # Lower bound for d: d >= ceil(1/rem_sum) and d >= min_d
        if rem_sum > 0:
            inv_rem = 1 / rem_sum
            d_min = max(min_d, int(inv_rem) + (1 if inv_rem != int(inv_rem) else 0))
        else:
            return None

        # Upper bound for d: rem_sum <= remaining_terms / d => d <= remaining_terms / rem_sum
        d_max = int(remaining_terms / rem_sum)
        if d_max < d_min:
            return None
        if d_max > 10000000:
            d_max = 10000000

        # Iterate d from d_min to d_max
        for d in range(d_min, d_max + 1):
            if time.time() > deadline:
                return None

            if Fraction(1, d) > rem_sum:
                break

            rem_new = rem_sum - Fraction(1, d)

            # Prune: Check if it's possible to fill rem_new with remaining_terms-1 terms >= d+1
            if remaining_terms > 1:
                max_next_sum = Fraction(0, 1)
                for i in range(remaining_terms - 1):
                    max_next_sum += Fraction(1, d + 1 + i)
                if rem_new > max_next_sum:
                    break

            res = dfs_exact(rem_new, d + 1, remaining_terms - 1, current_list + [d])
            if res is not None:
                return res

        return None

    # Try increasing k starting from 10 (proven lower bound)
    for k in range(10, 16):
        res = dfs_exact(Fraction(1, 1), 5, k, [])
        if res is not None:
            s = Fraction(0, 1)
            for d in res:
                s += Fraction(1, d)
            if s == 1 and len(set(res)) == len(res) and all(5 <= d <= 10000000 for d in res):
                return res

    # Fallback: Use the subset DP approach which is guaranteed to find a solution
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]

    raise RuntimeError("no solution found")

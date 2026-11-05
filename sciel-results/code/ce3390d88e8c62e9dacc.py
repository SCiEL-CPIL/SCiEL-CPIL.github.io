import time
from math import factorial, log
from functools import lru_cache

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    target = factorial(12)  # 479001600

    def time_check():
        return time.time() - start_time > DEADLINE

    # Generate all divisors of 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    def get_divisors():
        divs = [1]
        for p, e in [(2, 10), (3, 5), (5, 2), (7, 1), (11, 1)]:
            new_divs = []
            for d in divs:
                for k in range(e + 1):
                    new_divs.append(d * (p ** k))
            divs = new_divs
        return sorted(divs)

    all_divs = get_divisors()
    # Valid divisors: > 12 and <= target
    valid_divs = [d for d in all_divs if d > 12]
    n = len(valid_divs)

    # Precompute prime factorization for each divisor
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # For each divisor d, compute its prime exponents (e2, e3, e5, e7, e11)
    prime_exponents = []
    for d in valid_divs:
        tmp = d
        e2 = 0
        while tmp % 2 == 0:
            e2 += 1
            tmp //= 2
        e3 = 0
        while tmp % 3 == 0:
            e3 += 1
            tmp //= 3
        e5 = 0
        while tmp % 5 == 0:
            e5 += 1
            tmp //= 5
        e7 = 0
        while tmp % 7 == 0:
            e7 += 1
            tmp //= 7
        e11 = 0
        while tmp % 11 == 0:
            e11 += 1
            tmp //= 11
        prime_exponents.append((e2, e3, e5, e7, e11))

    # Total exponents needed
    total_exp = (10, 5, 2, 1, 1)

    # Sort candidates by value (ascending) for the DFS
    # We'll do a direct DFS that minimizes the maximum factor
    # Key insight: instead of binary search, do a single DFS with pruning
    # that finds the solution with minimum maximum factor

    # Precompute log for pruning
    log_valid = [log(d) for d in valid_divs]
    log_target = log(target)

    # For efficient pruning, precompute suffix information
    # suffix_max_prod[i][k] = product of the largest k elements from valid_divs[i:]
    # Since valid_divs is sorted ascending, the largest k from valid_divs[i:] 
    # are valid_divs[n-k:n] if k <= n-i
    # This is independent of i as long as k <= n-i!
    global_max_prod = [1] * 13
    for k in range(1, 13):
        if k <= n:
            prod = 1
            for j in range(k):
                prod *= valid_divs[n - 1 - j]
            global_max_prod[k] = prod
        else:
            global_max_prod[k] = float('inf')

    best_solution = None
    best_max_factor = target + 1

    def dfs(start_idx, current_prod, remaining_exp, chosen, current_max):
        nonlocal best_solution, best_max_factor

        if time_check():
            return

        # Check if we've reached the target
        if current_prod == target:
            if 1 <= len(chosen) <= 12:
                if current_max < best_max_factor:
                    best_max_factor = current_max
                    best_solution = list(chosen)
            return

        if current_prod > target:
            return

        if len(chosen) >= 12:
            return

        remaining_slots = 12 - len(chosen)
        avail = n - start_idx
        if avail == 0:
            return

        # Prune: if even using the largest remaining_slots elements can't reach target
        pick_k = min(remaining_slots, avail)
        if current_prod * global_max_prod[pick_k] < target:
            return

        # Prune: if the smallest available factor makes current_prod exceed target
        if current_prod * valid_divs[start_idx] > target:
            return

        # Prune based on prime exponents: check if remaining exponents are achievable
        # with the available candidates from start_idx onward
        # This is a complex check, so we do a simpler version:
        # For each prime, the sum of exponents in remaining candidates must be >= needed

        # Try each candidate from start_idx
        for i in range(start_idx, n):
            d = valid_divs[i]

            if current_prod * d > target:
                break

            # Check divisibility
            if target % (current_prod * d) != 0:
                continue

            # Check prime exponent feasibility
            exp = prime_exponents[i]
            feasible = True
            for j in range(5):
                if remaining_exp[j] - exp[j] < 0:
                    feasible = False
                    break
            if not feasible:
                continue

            new_chosen = chosen + [d]
            new_remaining_exp = tuple(remaining_exp[j] - exp[j] for j in range(5))
            new_current_max = max(current_max, d)

            # Prune: if new_current_max >= best_max_factor, we can skip this branch
            # because we're looking for a solution with smaller max
            if new_current_max >= best_max_factor:
                continue

            dfs(i + 1, current_prod * d, new_remaining_exp, new_chosen, new_current_max)

            if time_check():
                return

    dfs(0, 1, total_exp, [], 0)

    if best_solution is None:
        # Fallback: return [target] which is valid (1 factor, > 12, product = target)
        return [target]
    return best_solution

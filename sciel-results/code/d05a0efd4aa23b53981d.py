import math
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    N = 479001600  # 12!

    # Prime factorization of 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    target_pf = [10, 5, 2, 1, 1]
    n_primes = 5

    # Generate all divisors of 12!
    divisors = [1]
    for i, p in enumerate(primes):
        new_divs = []
        for d in divisors:
            for e in range(target_pf[i] + 1):
                new_divs.append(d * (p ** e))
        divisors = new_divs

    # Filter to divisors > 12 and sort
    valid_divisors = sorted([d for d in divisors if d > 12])

    # Precompute prime-factorization for each valid divisor
    def get_pf(n):
        pf = [0] * n_primes
        for i, p in enumerate(primes):
            while n % p == 0:
                pf[i] += 1
                n //= p
        return pf

    divisor_pf = {}
    for d in valid_divisors:
        divisor_pf[d] = get_pf(d)

    n_div = len(valid_divisors)

    # Precompute suffix products of valid_divisors (for max-product pruning)
    # suffix_prod[i] = product of valid_divisors[i:]
    # This can overflow, so we cap at a large value
    CAP = N * 10  # cap to avoid huge numbers
    suffix_prod = [1] * (n_div + 1)
    for i in range(n_div - 1, -1, -1):
        suffix_prod[i] = suffix_prod[i + 1] * valid_divisors[i]
        if suffix_prod[i] > CAP:
            suffix_prod[i] = CAP

    best_max_val = N  # initial: single factor N
    best_factors = [N]

    def time_exceeded():
        return time.time() - start_time > DEADLINE

    def dfs(start_idx, rem_pf, current_path, current_max, count_used):
        nonlocal best_max_val, best_factors

        if time_exceeded():
            return

        # If we've used all prime factors, check if valid
        if all(x == 0 for x in rem_pf):
            if current_path:
                if current_max < best_max_val:
                    best_max_val = current_max
                    best_factors = list(current_path)
            return

        # Max number of factors is 12
        if count_used >= 12:
            return

        # Pruning: if current_max is already >= best, no improvement possible
        # (we could still find same max with fewer factors, but we want strictly better)
        # Actually, we want to minimize max, so if current_max >= best_max_val,
        # any extension will have max >= current_max >= best_max_val, so skip
        if current_max >= best_max_val:
            return

        # Pruning: minimum possible product of remaining factors
        # The smallest available factors are valid_divisors[start_idx], valid_divisors[start_idx+1], ...
        # But we need their product to equal the remaining target
        # Compute remaining target
        rem_target = 1
        for i in range(n_primes):
            for _ in range(rem_pf[i]):
                rem_target *= primes[i]

        # If rem_target is 1, we need no more factors (handled above)
        # Check if rem_target can be formed by distinct factors from valid_divisors[start_idx:]

        # Lower bound: the smallest factor we can pick is valid_divisors[start_idx]
        # If rem_target < valid_divisors[start_idx] and rem_target != 1, impossible
        if rem_target > 1:
            if start_idx >= n_div:
                return
            if rem_target < valid_divisors[start_idx]:
                return

        # Upper bound: product of all remaining valid divisors
        if suffix_prod[start_idx] < rem_target:
            return

        # Try to use fewer factors first to minimize max
        # Iterate over possible next factors
        for i in range(start_idx, n_div):
            if time_exceeded():
                return

            d = valid_divisors[i]

            # Pruning: if d > best_max_val, no point (since we want max < best_max_val)
            # But we might still find a solution with max == best_max_val? No, we want strictly better.
            # Actually, we want to minimize max. If d >= best_max_val, then any solution using d
            # will have max >= d >= best_max_val, so skip.
            if d >= best_max_val:
                break

            # Check if d divides rem_target
            if rem_target % d != 0:
                continue

            # Check prime factorization compatibility
            pf_d = divisor_pf[d]
            valid = True
            for j in range(n_primes):
                if pf_d[j] > rem_pf[j]:
                    valid = False
                    break
            if not valid:
                continue

            new_rem_pf = [rem_pf[j] - pf_d[j] for j in range(n_primes)]
            new_rem_target = rem_target // d

            # Pruning: if new_rem_target > 1, need more factors
            # The smallest next factor is valid_divisors[i+1]
            if new_rem_target > 1:
                if i + 1 >= n_div:
                    continue
                if new_rem_target < valid_divisors[i + 1]:
                    continue
                # Upper bound check
                if suffix_prod[i + 1] < new_rem_target:
                    continue

            # Recurse
            current_path.append(d)
            new_max = d if d > current_max else current_max
            dfs(i + 1, new_rem_pf, current_path, new_max, count_used + 1)
            current_path.pop()

            # If we found a solution with this d, we might want to stop early
            # But we want to explore to find even smaller max
            # However, since we iterate d in increasing order, and we break when d >= best_max_val,
            # the first solution found at a given depth might not be optimal
            # We continue searching

    # Start DFS
    dfs(0, target_pf, [], 0, 0)

    # Fallback: return [N] if no better solution found
    return best_factors

import time
from math import factorial

_START = time.time()
_DEADLINE = _START + 85  # safety margin before 90s kill


def construct():
    target = factorial(12)  # 479001600 = 2^10 * 3^5 * 5^2 * 7^1 * 11^1

    # Generate all divisors of target
    divisors = set()
    for a in range(11):  # 2^a, a=0..10
        for b in range(6):  # 3^b, b=0..5
            for c in range(3):  # 5^c, c=0..2
                for d in range(2):  # 7^d, d=0..1
                    for e in range(2):  # 11^e, e=0..1
                        val = (2**a) * (3**b) * (5**c) * (7**d) * (11**e)
                        divisors.add(val)

    # Filter: must be > 12
    valid_divs = sorted([d for d in divisors if d > 12])

    best_max = [target]  # initial: single factor
    best_factors = [target]

    def dfs(remaining, start_idx, current_factors, current_max):
        nonlocal best_max, best_factors

        # Check deadline
        if time.time() > _DEADLINE:
            return

        # If remaining is 1, we've found a valid decomposition
        if remaining == 1:
            if len(current_factors) >= 1 and len(current_factors) <= 12:
                if current_max < best_max[0]:
                    best_max[0] = current_max
                    best_factors = list(current_factors)
            return

        # Pruning: if current_max >= best_max[0], no improvement possible
        # (we want to minimize the largest factor)
        if current_max >= best_max[0]:
            return

        # Try divisors starting from start_idx to ensure distinct and sorted
        for i in range(start_idx, len(valid_divs)):
            d = valid_divs[i]
            if d > remaining:
                break
            if remaining % d != 0:
                continue

            new_remaining = remaining // d
            new_max = max(current_max, d)

            # Pruning: if new_max >= best_max[0], skip
            if new_max >= best_max[0]:
                continue

            # Also, we can prune if even using the smallest possible remaining factors
            # can't beat best_max. But let's keep it simple for now.

            current_factors.append(d)
            dfs(new_remaining, i + 1, current_factors, new_max)
            current_factors.pop()

    # Start DFS
    dfs(target, 0, [], 0)

    return best_factors

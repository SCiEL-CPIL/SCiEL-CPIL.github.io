import time
from math import factorial

def construct():
    start_time = time.time()
    deadline = 85.0

    target = factorial(12)  # 479001600

    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1

    # Precompute divisors of target
    all_divs = []
    for i in range(1, int(target**0.5) + 1):
        if target % i == 0:
            all_divs.append(i)
            if i != target // i:
                all_divs.append(target // i)
    all_divs = sorted(set(all_divs))

    # Filter to divisors > 12
    divs_gt12 = [d for d in all_divs if d > 12]

    # Cache for divisors of a number that are > threshold
    divs_cache = {}

    def get_divs_gt(n, threshold):
        key = (n, threshold)
        if key in divs_cache:
            return divs_cache[key]
        result = []
        limit = int(n**0.5) + 1
        for i in range(1, limit):
            if n % i == 0:
                if i > threshold:
                    result.append(i)
                other = n // i
                if other != i and other > threshold:
                    result.append(other)
        result.sort()
        divs_cache[key] = result
        return result

    best_max = target
    best_path = [target]

    def dfs(rem, last, count, path):
        nonlocal best_max, best_path, start_time

        if time.time() - start_time > deadline:
            return

        if rem == 1:
            if 1 <= count <= 12:
                current_max = path[-1]
                if current_max < best_max:
                    best_max = current_max
                    best_path = list(path)
            return

        if count == 12:
            if rem > last and rem > 12 and rem < best_max:
                best_max = rem
                best_path = path + [rem]
            return

        # Get candidate divisors of rem that are > last and < best_max
        candidates = get_divs_gt(rem, last)

        valid_candidates = []
        for d in candidates:
            if d < best_max:
                valid_candidates.append(d)
            else:
                break

        # Try candidates in ascending order
        for d in valid_candidates:
            # Pruning: if rem // d < d + 1 and rem // d != 1, we can't continue
            if rem // d != 1 and rem // d < d + 1:
                continue

            # Additional pruning: check if we can form enough factors
            # We need (12 - count - 1) more factors, each > d
            # The minimum product for m factors > d is (d+1)*(d+2)*...*(d+m)
            remaining_count = 12 - count - 1
            if remaining_count > 0:
                min_prod = 1
                for j in range(remaining_count):
                    min_prod *= (d + 1 + j)
                    if min_prod > rem // d:
                        break
                if min_prod > rem // d:
                    continue

            dfs(rem // d, d, count + 1, path + [d])

            if time.time() - start_time > deadline:
                return

    dfs(target, 12, 0, [])

    return best_path

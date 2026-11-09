import math
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin within 60s limit

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def verify_solution(denoms):
        """Verify that sum of reciprocals equals exactly 1."""
        if len(denoms) < 1 or len(denoms) > 256:
            return False
        if len(set(denoms)) != len(denoms):
            return False
        for d in denoms:
            if d < 5 or d > 10000000:
                return False
        s = Fraction(0, 1)
        for d in denoms:
            s += Fraction(1, d)
        return s == Fraction(1, 1)

    def search_for_k(num, den, k, min_d, path, used_set, time_limit):
        """
        DFS to find k distinct denominators >= min_d summing to num/den.
        Uses strong pruning.
        """
        if time.time() > time_limit:
            return None

        if k == 0:
            if num == 0 and den == 1:
                return list(path)
            return None

        if k == 1:
            # Need 1/d = num/den, so d = den/num
            if num == 0:
                return None
            if den % num == 0:
                d = den // num
                if d >= min_d and d <= 10000000 and d not in used_set:
                    return list(path) + [d]
            return None

        # Bounds on d:
        # 1/d <= num/den  =>  d >= den/num
        # Also, with k terms all >= d, max sum is k/d, so k/d >= num/den => d <= k*den/num
        d_min = max(min_d, math.ceil(den / num) if num > 0 else min_d)
        d_max = int(math.floor(k * den / num)) if num > 0 else 10000000

        if d_max > 10000000:
            d_max = 10000000

        if d_min > d_max:
            return None

        # Pruning: check if max possible sum with k terms starting from d_min is >= num/den
        # Max sum is achieved by smallest denominators: d_min, d_min+1, ..., d_min+k-1
        # But we can use a simpler bound: k/d_min >= num/den is already checked by d_max
        # Additional pruning: min possible sum with k terms starting from d is k/(d+k-1) roughly
        # Actually, the minimum sum with k distinct terms >= d is 1/(d+k-1) + ... no, it's 1/d + 1/(d+1) + ... but that's max
        # The MINIMUM sum with k distinct terms >= d_min is when we pick the largest possible, but that's unbounded.
        # Better pruning: if even the max sum (smallest denominators) is too small, prune.
        # But we already have d_max ensuring k*den/num >= d_min, which means d_min <= k*den/num, so k/d_min >= num/den.
        
        # Iterate d from d_min to d_max
        for d in range(d_min, d_max + 1):
            if time.time() > time_limit:
                return None
            if d in used_set:
                continue

            # After picking d, remaining sum is num/den - 1/d = (num*d - den)/(den*d)
            n2 = num * d - den
            if n2 <= 0:
                # 1/d >= num/den, so this d is too small (or equal if n2==0 but then we need 0 more terms)
                if n2 == 0 and k == 1:
                    if d >= min_d and d <= 10000000 and d not in used_set:
                        return list(path) + [d]
                continue

            d2 = den * d

            # Reduce fraction
            g = gcd(n2, d2)
            n2_red = n2 // g
            d2_red = d2 // g

            used_set.add(d)
            result = search_for_k(n2_red, d2_red, k - 1, d + 1, path + [d], used_set, time_limit)
            used_set.remove(d)

            if result is not None:
                return result

        return None

    # Known good solutions for fallback
    hardcoded_candidates = [
        [5, 6, 7, 10, 14, 20, 42, 70, 105, 210],
        [5, 6, 7, 10, 14, 21, 35, 42, 70, 105],
    ]

    best_solution = None
    for candidate in hardcoded_candidates:
        if verify_solution(candidate):
            best_solution = candidate
            break

    if best_solution is None:
        # Fallback: construct via DP on 2520
        target = 2520
        reached = {0: []}
        for d in range(5, target + 1):
            if target % d:
                continue
            w = target // d
            for total, values in list(reached.items()):
                if total + w <= target and total + w not in reached:
                    reached[total + w] = values + [d]
            if target in reached:
                best_solution = reached[target]
                break

    if best_solution is None:
        best_solution = [5, 6, 7, 10, 14, 20, 42, 70, 105, 210]

    best_k = len(best_solution)

    # Try to find shorter solutions, starting from k=8 (theoretical minimum)
    for k in range(8, best_k):
        if time.time() > deadline:
            break

        # For each k, try to find a solution
        # Use a time budget per k value
        time_budget = deadline - (time.time() - start_time)
        if time_budget < 1.0:
            break

        # Allocate time: more time for smaller k
        k_time_limit = start_time + min(deadline, time.time() + max(1.0, time_budget * 0.6))

        solution = search_for_k(1, 1, k, 5, [], set(), k_time_limit)
        if solution is not None:
            if verify_solution(solution):
                best_solution = solution
                best_k = k
                break

    return best_solution
# EVOLVE-BLOCK-END

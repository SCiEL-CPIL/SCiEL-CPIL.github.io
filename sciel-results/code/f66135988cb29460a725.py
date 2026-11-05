import math
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    def time_ok():
        return time.time() < deadline

    def dfs(rem_num, rem_den, min_den, terms_left, current_denoms):
        if not time_ok():
            return None

        if terms_left == 0:
            if rem_num == 0:
                return current_denoms
            return None

        if terms_left == 1:
            if rem_num == 0:
                return None
            if rem_den % rem_num == 0:
                d = rem_den // rem_num
                if d >= min_den and d <= 10000000:
                    return current_denoms + [d]
            return None

        # Pruning: max possible sum with terms_left distinct denominators >= min_den
        # Upper bound: terms_left / min_den (since each 1/d <= 1/min_den)
        if rem_num * min_den > rem_den * terms_left:
            return None

        # Lower bound for next denominator: 1/d <= rem => d >= ceil(rem_den/rem_num)
        # Also d >= min_den
        lower_bound = max(min_den, (rem_den + rem_num - 1) // rem_num)

        # Upper bound: 1/d >= rem/terms_left => d <= terms_left * rem_den / rem_num
        upper_bound = (terms_left * rem_den) // rem_num

        if lower_bound > upper_bound:
            return None

        if upper_bound > 10000000:
            upper_bound = 10000000

        # Try denominators in order: start from lower_bound
        # For better pruning, we can try a few strategies
        for d in range(lower_bound, upper_bound + 1):
            if not time_ok():
                return None

            # 1/d must be <= rem, so rem_num*d >= rem_den
            if rem_num * d < rem_den:
                continue

            new_num = rem_num * d - rem_den
            if new_num < 0:
                continue

            new_den = rem_den * d
            g = math.gcd(new_num, new_den)
            new_num //= g
            new_den //= g

            # Additional pruning: remaining sum must be achievable
            # with (terms_left-1) terms, each >= d+1
            # Max remaining sum: (terms_left-1)/(d+1)
            if new_num > 0 and (terms_left - 1) > 0:
                if new_num * (d + 1) > new_den * (terms_left - 1):
                    continue

            result = dfs(new_num, new_den, d + 1, terms_left - 1, current_denoms + [d])
            if result is not None:
                return result

        return None

    # We've proven k=1..8 are impossible, so start from k=9
    for k in range(9, 257):
        if not time_ok():
            break
        result = dfs(1, 1, 5, k, [])
        if result is not None:
            # Verify the solution
            total = Fraction(0, 1)
            for d in result:
                total += Fraction(1, d)
            if total == 1 and all(5 <= d <= 10000000 for d in result) and len(set(result)) == len(result):
                return result

    # Fallback: greedy Egyptian fraction decomposition
    def greedy():
        rem = Fraction(1, 1)
        denoms = []
        last_d = 4
        while rem > 0 and len(denoms) < 256:
            if not time_ok():
                return None
            d = max(last_d + 1, math.ceil(rem.denominator / rem.numerator))
            if d > 10000000:
                return None
            denoms.append(d)
            last_d = d
            rem -= Fraction(1, d)
        if rem == 0:
            return denoms
        return None

    baseline = greedy()
    if baseline:
        return baseline

    raise RuntimeError("no representation found")

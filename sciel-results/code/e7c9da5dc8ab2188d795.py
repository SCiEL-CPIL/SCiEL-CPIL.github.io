"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
import time
import math

# EVOLVE-BLOCK-START

def construct():
    start = time.time()
    deadline = start + 75.0  # safety margin before 90s kill
    MAX_DEN = 10**7

    def remaining_time():
        return deadline - time.time()

    # ------------------------------------------------------------------
    # Try to find a solution with k terms using DFS with strong pruning.
    # We search k from 5 upward.
    # ------------------------------------------------------------------

    def dfs(rem, min_den, terms_left, used):
        """
        rem: Fraction remaining to sum to 1
        min_den: next denominator must be >= min_den (and not in used)
        terms_left: how many more terms we can use
        used: frozenset of denominators already chosen
        Returns list of chosen denominators or None
        """
        if rem == 0:
            return []
        if terms_left == 0:
            return None
        if time.time() > deadline:
            return None

        # Lower bound on next denominator: 1/d <= rem => d >= ceil(1/rem)
        # 1/rem = rem.den / rem.num
        den_lb = max(min_den, math.ceil(rem.denominator / rem.numerator))

        if den_lb > MAX_DEN:
            return None

        # Pruning: even taking the largest possible terms (smallest denominators),
        # we must be able to reach rem. The max sum with terms_left terms starting
        # from den_lb is at most terms_left / den_lb (each term <= 1/den_lb).
        if terms_left * Fraction(1, den_lb) < rem:
            return None

        # Upper bound on den: 1/den >= rem / terms_left => den <= terms_left / rem
        den_ub_frac = terms_left * rem.denominator // rem.numerator
        den_ub = min(MAX_DEN, max(den_lb, den_ub_frac))

        # Cap the search range for efficiency
        # Use a tighter cap based on the structure of the problem
        search_end = min(den_ub, den_lb + 200000)

        # For the last term, we can directly compute
        if terms_left == 1:
            # Need 1/d = rem, so d = 1/rem = rem.den / rem.num
            if rem.denominator % rem.numerator == 0:
                d = rem.denominator // rem.numerator
                if d >= min_den and d <= MAX_DEN and d not in used:
                    return [d]
            return None

        # For two terms left: 1/a + 1/b = rem => b = a*rem.den / (a*rem.num - rem.den)
        # Actually: 1/a + 1/b = rem => (a+b)/(ab) = rem => b = a / (rem*a - 1)
        # Let's derive: rem = (a+b)/(ab) => rem*a*b = a + b => rem*a*b - b = a => b(rem*a - 1) = a
        # => b = a / (rem*a - 1)
        # For b to be integer: a must be divisible by (rem*a - 1)
        # rem*a - 1 = (rem.numerator * a) / rem.denominator - 1 = (rem.numerator * a - rem.denominator) / rem.denominator
        # So b = a * rem.denominator / (rem.numerator * a - rem.denominator)
        # Let num = rem.numerator, den = rem.denominator
        # b = a * den / (num * a - den)
        # For b to be integer, (num * a - den) must divide (a * den)
        # Let g = gcd(num * a - den, a * den)
        # This is complex; let's just iterate but with a tighter bound.
        if terms_left == 2:
            num = rem.numerator
            den = rem.denominator
            # b = a * den / (num * a - den)
            # num * a - den > 0 => a > den / num = 1/rem
            # So a >= ceil(den/num) + 1 = den_lb + 1 (since den_lb = ceil(den/num))
            # Also b > a => a * den / (num * a - den) > a
            # => den / (num * a - den) > 1
            # => den > num * a - den
            # => 2 * den > num * a
            # => a < 2 * den / num
            a_max = min(den_ub, (2 * den) // num - 1)
            a_max = max(a_max, den_lb)
            for a in range(den_lb, a_max + 1):
                if time.time() > deadline:
                    return None
                if a in used:
                    continue
                numerator_b = a * den
                denominator_b = num * a - den
                if denominator_b <= 0:
                    continue
                if numerator_b % denominator_b != 0:
                    continue
                b = numerator_b // denominator_b
                if b <= a or b > MAX_DEN or b in used:
                    continue
                # Verify
                if Fraction(1, a) + Fraction(1, b) == rem:
                    return [a, b]
            return None

        # General case: iterate over candidate denominators
        for den in range(den_lb, search_end + 1):
            if time.time() > deadline:
                return None
            if den in used:
                continue
            term = Fraction(1, den)
            if term > rem:
                continue
            new_rem = rem - term
            if new_rem < 0:
                continue
            # Quick prune: new_rem must be achievable with terms_left-1 terms
            if terms_left - 1 > 0:
                # Max possible: (terms_left-1) / (den+1)
                if (terms_left - 1) * Fraction(1, den + 1) < new_rem:
                    continue
            result = dfs(new_rem, den + 1, terms_left - 1, used | {den})
            if result is not None:
                return [den] + result
        return None

    # Try k from 5 upward
    for k in range(5, 10):
        if time.time() > deadline:
            break
        result = dfs(Fraction(1, 1), 5, k, frozenset())
        if result is not None:
            # Verify solution
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and len(set(result)) == len(result) and all(5 <= d <= MAX_DEN for d in result):
                return result

    # ------------------------------------------------------------------
    # Fallback: use a known construction or DP approach
    # ------------------------------------------------------------------
    # Try a simple greedy-like approach with backtracking on small denominators
    # Or use the telescoping series idea: 1 = 1/2 + 1/3 + 1/6 but 2,3,6 not all >= 5
    # Another identity: 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + ... 
    # Let's just do a DP over divisors of a highly composite number

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
            return reached[target]

    # Try a larger target
    target = 27720
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]

    raise RuntimeError("no valid representation found")

# EVOLVE-BLOCK-END

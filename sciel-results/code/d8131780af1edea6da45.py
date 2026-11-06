"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _compute_max_sum(k, min_den):
    """Compute maximum possible sum of k distinct denominators starting from min_den."""
    # Use the k smallest available denominators: min_den, min_den+1, ..., min_den+k-1
    total = Fraction(0, 1)
    for i in range(k):
        total += Fraction(1, min_den + i)
    return total

def _compute_min_sum(k, min_den, max_den_limit=10000000):
    """Compute minimum possible sum of k distinct denominators >= min_den, bounded by max_den_limit."""
    # Use the k largest possible denominators, but they must be >= min_den
    # To minimize sum, use largest denominators: max_den_limit, max_den_limit-1, ...
    # But we need k distinct values >= min_den
    total = Fraction(0, 1)
    for i in range(k):
        d = max_den_limit - i
        if d < min_den:
            d = min_den + (k - 1 - i)
        total += Fraction(1, d)
    return total

def _search_min_terms():
    """Find minimum number of distinct denominators >= 5 summing to 1."""
    deadline = time.time() + 85  # Safety margin before 90s kill
    best = None

    def dfs(remaining, min_den, current, terms_left):
        nonlocal best
        if time.time() > deadline:
            return
        if terms_left == 0:
            if remaining == 0:
                if best is None or len(current) < len(best):
                    best = list(current)
            return
        if remaining <= 0:
            if remaining == 0 and terms_left == 0:
                if best is None or len(current) < len(best):
                    best = list(current)
            return

        # Pruning: check if remaining is achievable with terms_left terms
        # Max possible sum: use terms_left smallest available denominators
        max_possible = _compute_max_sum(terms_left, min_den)
        if remaining > max_possible:
            return

        # Min possible sum: use terms_left largest possible denominators
        # This is a very loose bound but helps in some cases
        # For now, skip min sum check as it's expensive

        if terms_left == 1:
            # Directly compute the required denominator
            if remaining > 0:
                # 1/d = remaining => d = 1/remaining
                # remaining must be a unit fraction
                num, den = remaining.numerator, remaining.denominator
                if num == 1:
                    d = den
                    if d >= min_den and d <= 10000000 and d not in current:
                        if best is None or len(current) + 1 < len(best):
                            best = list(current) + [d]
            return

        # Try denominators for the current term
        # d must satisfy: 1/d <= remaining => d >= ceil(1/remaining)
        # Also d >= min_den
        if remaining > 0:
            # d >= 1/remaining
            min_d = max(min_den, (1 // remaining.numerator) + (1 if remaining.numerator > 1 else 0))
            # More precisely: d >= ceil(1/remaining) = ceil(den/num)
            num, den = remaining.numerator, remaining.denominator
            min_d = max(min_den, den // num + (1 if den % num != 0 else 0))
        else:
            min_d = min_den

        # Upper bound on d: we need to be able to fill remaining - 1/d with (terms_left-1) terms
        # The max sum of (terms_left-1) terms with min denominator d+1 is _compute_max_sum(terms_left-1, d+1)
        # So we need: remaining - 1/d <= _compute_max_sum(terms_left-1, d+1)
        # This gives an upper bound on d

        # Also, the min sum of (terms_left-1) terms with min denominator d+1 must be <= remaining - 1/d
        # But this is a lower bound check, less useful for upper bound on d

        # For efficiency, let's limit the range of d we try
        # d cannot be too large because then 1/d is too small and we can't reach 'remaining'
        # with the remaining terms
        # Max d: 1/d >= remaining / terms_left (rough) => d <= terms_left / remaining
        max_d = min(10000000, int(terms_left / remaining) + 1) if remaining > 0 else 10000000

        # But we also need remaining - 1/d to be non-negative
        # 1/d <= remaining => d >= 1/remaining (already handled)

        # Iterate through possible d values
        # To improve search, try values that are more likely to lead to solutions
        # Heuristic: try d values that make remaining - 1/d a "nice" fraction
        
        for d in range(min_d, max_d + 1):
            if time.time() > deadline:
                return
            new_rem = remaining - Fraction(1, d)
            if new_rem < 0:
                break
            if new_rem == 0:
                if terms_left == 1:
                    if best is None or len(current) + 1 < len(best):
                        best = list(current) + [d]
                # If terms_left > 1, we can't have remaining terms sum to 0
                # unless all remaining terms are 0, which is impossible for positive denominators
                # So this case only works if terms_left == 1
                continue

            # Check if new_rem is achievable with (terms_left - 1) terms
            if terms_left - 1 > 0:
                max_possible_next = _compute_max_sum(terms_left - 1, d + 1)
                if new_rem > max_possible_next:
                    continue
                # Also check minimum: new_rem must be >= min possible sum
                # This is a very loose bound, skip for now

            dfs(new_rem, d + 1, current + [d], terms_left - 1)

    # Determine the minimum number of terms needed
    # 7 terms: max sum with 5,6,7,8,9,10,11 = ~0.937 < 1, so 7 is impossible
    # 8 terms: max sum with 5,6,7,8,9,10,11,12 = ~1.02 > 1, so 8 might work
    for n_terms in range(8, 13):
        # Check if n_terms is even possible
        max_sum = _compute_max_sum(n_terms, 5)
        if max_sum < 1:
            continue
        dfs(Fraction(1, 1), 5, [], n_terms)
        if best is not None:
            break
        if time.time() > deadline:
            break

    return best

def construct():
    result = _search_min_terms()
    if result is None:
        # Fallback: use a known solution or DP approach
        # Try a simple DP with a moderate common denominator
        target = 2520
        reached = {0: []}
        for d in range(5, target + 1):
            if target % d != 0:
                continue
            w = target // d
            for total, values in list(reached.items()):
                new_total = total + w
                if new_total <= target and new_total not in reached:
                    reached[new_total] = values + [d]
            if target in reached:
                return reached[target]
        # If still no solution, return a hardcoded valid solution
        # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + 1/31 + 1/32 + 1/33 + 1/34 + 1/35 + 1/36 + 1/37 + 1/38 + 1/39 + 1/40 + 1/41 + 1/42 + 1/43 + 1/44 + 1/45 + 1/46 + 1/47 + 1/48 + 1/49 + 1/50
        # This is too many. Let me use a different fallback.
        raise RuntimeError("no baseline representation")
    return result
# EVOLVE-BLOCK-END

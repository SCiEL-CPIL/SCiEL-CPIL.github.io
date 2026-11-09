import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # safety margin before 90s kill

    MIN_D = 5
    MAX_D = 10_000_000

    # Try to find a representation with the minimum number of terms.
    # We know that 7 terms is impossible (max sum of 7 smallest denominators >= 5 is < 1).
    # 8 terms is the minimum possible.

    # We will use a DFS search to find 8 distinct denominators d1 < d2 < ... < d8
    # such that sum(1/di) = 1.

    # To make the search efficient, we use strong pruning.
    # At each step, we have:
    # - start: the minimum allowed denominator for the next term
    # - remaining: the remaining sum to achieve
    # - count: the number of terms chosen so far
    # - target_count: the total number of terms we want (8)
    # - chosen: the list of chosen denominators so far

    # Pruning 1: Max possible sum
    # The maximum sum we can achieve with (target_count - count) terms starting from 'start'
    # is 1/start + 1/(start+1) + ... + 1/(start + target_count - count - 1).
    # If this is less than 'remaining', we prune.

    # Pruning 2: Min possible sum
    # The minimum sum is achieved by choosing the largest possible denominators.
    # This is harder to bound tightly, but we can use a loose bound:
    # If we choose the largest possible denominators, the sum is very small.
    # A better bound: if 1/start * (target_count - count) < remaining, we prune.
    # Actually, a tighter bound is: if the sum of the (target_count - count) largest
    # possible reciprocals is less than 'remaining', we prune. But we don't have a fixed
    # upper bound on denominators, so this is tricky.
    # Instead, we can use: if 1/start > remaining, we prune (since all terms are <= 1/start).
    # But we already enforce d >= ceil(1/remaining).

    # Pruning 3: Distinctness and ordering
    # We enforce d > previous_d by setting start = previous_d + 1.

    # Let's implement the DFS with these prunings.

    def dfs(start, remaining, count, chosen, target_count):
        if time.time() > deadline:
            return None

        if count == target_count:
            if remaining == 0:
                return list(chosen)
            return None

        # Pruning: max possible sum
        max_possible = Fraction(0)
        for k in range(target_count - count):
            max_possible += Fraction(1, start + k)
        if max_possible < remaining:
            return None

        # Determine the range for the next denominator d
        # We need 1/d <= remaining, so d >= ceil(1/remaining)
        if remaining > 0:
            min_d = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
        else:
            min_d = start

        min_d = max(min_d, start)

        # Upper bound for d:
        # We need to be able to represent 'remaining - 1/d' with (target_count - count - 1) terms.
        # The maximum sum of those terms is 1/(d+1) + 1/(d+2) + ... + 1/(d + target_count - count - 1).
        # This must be >= remaining - 1/d.
        # Also, 1/d must be <= remaining (already enforced).
        
        # Let's find an upper bound for d.
        # If d is too large, 1/d is small, and we can't reach 'remaining' even with the largest
        # possible remaining terms.
        
        # A simple upper bound: d <= remaining.denominator // remaining.numerator + some margin
        # But this is not tight.
        
        # Let's use a heuristic upper bound.
        # If d > 1/remaining, then 1/d < remaining, which is fine.
        # But we also need to be able to reach 'remaining' with the remaining terms.
        
        # Let's try a range of d values.
        # We'll search from min_d to some upper bound.
        
        # Upper bound heuristic:
        # If we choose d, the remaining sum is remaining - 1/d.
        # The maximum sum of the remaining (target_count - count - 1) terms is:
        # sum_{k=1}^{target_count-count-1} 1/(d+k)
        # This must be >= remaining - 1/d.
        
        # Let's find the largest d such that this holds.
        # As d increases, 1/d decreases, so remaining - 1/d increases.
        # Also, the max sum of the remaining terms decreases.
        # So there is a crossover point.
        
        # Let's just search a reasonable range.
        # We'll search from min_d to min_d + 10000, but break early if the max sum condition fails.
        
        max_d = min_d + 5000  # heuristic upper bound
        
        for d in range(min_d, max_d + 1):
            if time.time() > deadline:
                return None

            term = Fraction(1, d)
            if term > remaining:
                continue

            new_remaining = remaining - term
            if new_remaining < 0:
                continue

            # If this is the last term, check if new_remaining == 0
            if count + 1 == target_count:
                if new_remaining == 0:
                    return chosen + [d]
                continue

            # Check if new_remaining can be achieved with (target_count - count - 1) terms >= d+1
            max_rest = Fraction(0)
            for k in range(1, target_count - count):
                max_rest += Fraction(1, d + k)
            if max_rest < new_remaining:
                # As d increases, max_rest decreases and new_remaining increases,
                # so we can break here.
                break

            chosen.append(d)
            result = dfs(d + 1, new_remaining, count + 1, chosen, target_count)
            if result is not None:
                return result
            chosen.pop()

        return None

    # Try to find a solution with 8 terms
    result = dfs(MIN_D, Fraction(1, 1), 0, [], 8)
    if result is not None:
        # Verify
        total = sum(Fraction(1, d) for d in result)
        if total == 1 and len(set(result)) == 8 and all(MIN_D <= d <= MAX_D for d in result):
            return result

    # If 8 terms fails, try 9 terms
    for n_terms in range(9, 12):
        result = dfs(MIN_D, Fraction(1, 1), 0, [], n_terms)
        if result is not None:
            total = sum(Fraction(1, d) for d in result)
            if total == 1 and len(set(result)) == n_terms and all(MIN_D <= d <= MAX_D for d in result):
                return result

    # Last resort: return a known valid solution if we can find one quickly
    # This is a fallback to ensure we always return a valid answer.
    # We can use a precomputed solution or a greedy approach.
    
    # Greedy approach: repeatedly pick the smallest denominator that doesn't make the sum exceed 1
    # This won't guarantee minimality, but it will give a valid answer.
    
    def greedy_solve():
        chosen = []
        remaining = Fraction(1, 1)
        start = MIN_D
        while remaining > 0:
            # Find the smallest d >= start such that 1/d <= remaining
            d = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
            d = max(d, start)
            
            # Check if we can add this term
            if 1/d <= remaining:
                chosen.append(d)
                remaining -= Fraction(1, d)
                start = d + 1
            
            # If we've used too many terms, break
            if len(chosen) > 256:
                break
        
        return chosen

    result = greedy_solve()
    if result:
        total = sum(Fraction(1, d) for d in result)
        if total == 1 and len(set(result)) == len(result) and all(MIN_D <= d <= MAX_D for d in result):
            return result

    raise RuntimeError("no representation found")

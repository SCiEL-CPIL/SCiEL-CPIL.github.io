import time
from fractions import Fraction

def construct():
    """
    Find 1..256 distinct integer denominators in [5,10000000] whose reciprocal sum is exactly 1.
    Minimize their count.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin

    def dfs(remaining_num, remaining_den, start_denom, terms, depth, max_depth):
        """
        remaining: remaining_num / remaining_den
        start_denom: minimum denominator to consider
        terms: list of denominators chosen so far
        depth: current depth
        max_depth: maximum depth allowed
        """
        if time.time() > deadline:
            return None
        
        if remaining_num == 0:
            return terms if len(terms) > 0 else None
        
        if depth == max_depth:
            return None
        
        if remaining_num < 0 or remaining_den <= 0:
            return None
        
        # remaining = remaining_num / remaining_den
        # We need to pick d such that 1/d <= remaining
        # d >= 1/remaining = remaining_den / remaining_num
        
        min_d = max(start_denom + 1, 5)
        
        # Upper bound:
        # (max_depth - depth) * (1/min_d) >= remaining
        # min_d <= (max_depth - depth) / remaining
        # min_d <= (max_depth - depth) * remaining_den / remaining_num
        
        # Also, the smallest possible sum for the remaining (max_depth - depth) terms
        # starting from d is sum(1/d + 1/(d+1) + ...).
        # A simpler bound:
        # The largest term we can pick is 1/d.
        # The sum of the next (max_depth - depth - 1) terms must cover the rest.
        # A loose but effective bound is:
        # (max_depth - depth) / d >= remaining_num / remaining_den
        # d <= (max_depth - depth) * remaining_den / remaining_num
        
        max_d_candidate = ((max_depth - depth) * remaining_den) // remaining_num
        if max_d_candidate < min_d:
            return None
        
        max_d = min(max_d_candidate, 10000000)
        
        for d in range(min_d, max_d + 1):
            # Check if 1/d <= remaining
            # 1/d <= remaining_num/remaining_den
            # remaining_den <= d * remaining_num
            
            if remaining_den > d * remaining_num:
                # 1/d is too small? No, if 1/d < remaining, it's valid as a candidate.
                # Wait, the condition for d to be valid is 1/d <= remaining.
                # If 1/d > remaining, we skip.
                # 1/d > remaining_num/remaining_den
                # remaining_den > d * remaining_num
                # So if this condition is true, 1/d > remaining, so we should NOT use it.
                # We want 1/d <= remaining.
                # So we continue if 1/d > remaining.
                continue
            
            # Subtract 1/d from remaining
            # new_remaining = remaining - 1/d
            # = remaining_num/remaining_den - 1/d
            # = (d * remaining_num - remaining_den) / (d * remaining_den)
            
            new_num = d * remaining_num - remaining_den
            new_den = d * remaining_den
            
            # Reduce fraction
            if new_num == 0:
                # We hit exactly 0.
                # We need to ensure we have enough terms?
                # The problem says "1..256 distinct... minimize count".
                # If we hit 0, we have a valid solution.
                # But we are in a loop for fixed max_depth.
                # If we hit 0 before max_depth, it's a valid solution with fewer terms.
                # But we are iterating k from 11 upwards.
                # So if we find a solution at depth < max_depth, it's even better.
                # However, the DFS structure here fixes max_depth.
                # If we find a solution, we return it.
                return terms + [d]
            
            if new_num < 0:
                # 1/d was too large?
                # We already checked 1/d <= remaining.
                # So new_num should be >= 0.
                # If new_num < 0, it means 1/d > remaining, which we skipped.
                # So this shouldn't happen.
                break
            
            # Recurse
            result = dfs(new_num, new_den, d, terms + [d], depth + 1, max_depth)
            if result is not None:
                return result
        
        return None

    # Try to find solutions with increasing number of terms
    # Theoretical minimum is 11.
    for k in range(11, 20):
        if time.time() > deadline:
            break
        # Start with remaining = 1/1
        result = dfs(1, 1, 4, [], 0, k)
        if result is not None:
            return result

    # Fallback: return a known valid solution if search fails (should not happen)
    # This is a placeholder, the search above should succeed.
    raise RuntimeError("no valid representation found")

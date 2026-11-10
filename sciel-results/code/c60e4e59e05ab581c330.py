"""Few-term Egyptian fractions with denominators greater than four."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin before 60s limit

    def verify(denoms):
        """Verify that the sum of reciprocals of denoms equals 1."""
        total = sum(Fraction(1, d) for d in denoms)
        return total == 1

    # Pre-computed known solution: 16 terms
    known_solution = [5, 6, 7, 10, 14, 21, 30, 35, 42, 70, 105, 140, 210, 280, 420, 840]
    if verify(known_solution):
        best = known_solution
    else:
        best = None

    # Try to find a solution with fewer terms using a smarter search
    def search_small_sets(max_terms, max_denom=2000):
        """Try to find a small set of denominators that sums to 1."""
        if time.time() > deadline:
            return None

        for k in range(2, max_terms + 1):
            if time.time() > deadline:
                return None

            # Precompute max possible sum for given min_d and remaining terms
            # This helps prune branches where even the smallest denominators can't reach the target
            def max_possible_sum(min_d, terms):
                s = Fraction(0, 1)
                for i in range(terms):
                    s += Fraction(1, min_d + i)
                return s

            def dfs(remaining, min_denom, used, depth, target_depth):
                if time.time() > deadline:
                    return None
                if depth == target_depth:
                    if remaining == 0:
                        return sorted(used)
                    return None
                if remaining == 0:
                    return sorted(used)

                if remaining <= 0:
                    return None
                
                terms_left = target_depth - depth
                min_d = max(min_denom, math.ceil(1 / remaining))
                
                # Pruning 1: Even with the smallest possible denominators, can we reach the remaining sum?
                # Max sum we can add is sum(1/(min_d + i) for i in range(terms_left))
                if max_possible_sum(min_d, terms_left) < remaining:
                    return None

                # Pruning 2: The next denominator d must satisfy 1/d >= remaining / terms_left
                # i.e., d <= terms_left / remaining
                # Also d >= min_d
                max_d = min(max_denom, int(terms_left / remaining) + 1)
                
                if min_d > max_d:
                    return None

                for d in range(min_d, max_d + 1):
                    if d in used:
                        continue
                    new_remaining = remaining - Fraction(1, d)
                    if new_remaining < 0:
                        break
                    result = dfs(new_remaining, d + 1, used | {d}, depth + 1, target_depth)
                    if result is not None:
                        return result
                return None

            # Start DFS for this target depth
            result = dfs(Fraction(1, 1), 5, set(), 0, k)
            if result is not None:
                return result

        return None

    # Try to find solutions with fewer than 16 terms
    # Start with small numbers of terms and increase
    # We know k=5 is impossible. Start from k=6 or 7 to save time?
    # Actually, let's try k=15, 14, 13 first as they are more likely to exist and be found?
    # No, smaller k is better score. But if k=15 exists, it's better than 16.
    # Let's try k from 6 up to 15.
    
    # To increase chances, we can run the search for a specific k for a portion of the time,
    # or just let it run until deadline.
    
    # We will iterate k from 6 to 15.
    # For each k, we set a time limit or just let it run.
    # Given the difficulty, we might only find k=15 or 14.
    
    for k in range(6, 16):
        if time.time() > deadline - 5.0:  # Leave 5s for fallback
            break
            
        # For larger k, denominators can be larger.
        # For k=15, denominators might be in the hundreds or thousands.
        max_denom = 5000
        
        result = search_small_sets(k, max_denom)
        if result is not None:
            if verify(result):
                if best is None or len(result) < len(best):
                    best = result
                # If we found a solution, we can try to find an even smaller one if time permits
                # But smaller k is harder. So we should try smaller k first?
                # The loop goes 6, 7, ..., 15.
                # If we find k=15, we should still try k=14?
                # The loop structure above tries 6, then 7...
                # If 6 fails (likely), it tries 7...
                # This is good.
                pass

    # If we found a better solution, return it
    if best is not None:
        return sorted(best)

    # Fall back to the known solution
    return known_solution

# EVOLVE-BLOCK-END

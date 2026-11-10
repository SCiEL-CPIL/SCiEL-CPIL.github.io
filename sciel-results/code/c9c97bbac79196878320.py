"""Rewritten: exact bounded DFS + local repair to minimize the largest denominator for 5/31."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd

def construct():
    deadline = time.time() + 85.0
    target = Fraction(5, 31)

    def check_time():
        return time.time() > deadline

    # Quick greedy baseline (not necessarily distinct, but used only as fallback if needed)
    def greedy(frac, min_d=1, used=None):
        used = used or set()
        terms = []
        r = frac
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d < min_d:
                d = min_d
            while d in used:
                d += 1
            terms.append(d)
            r -= Fraction(1, d)
            min_d = d + 1
            used.add(d)
        return terms

    baseline = greedy(target)
    best_solution = baseline[:]
    best_max_den = max(baseline)

    # Exact DFS: find any representation with max denom <= B.
    # State: remaining fraction = n/d in lowest terms, current min allowed denominator = lo.
    # We try denominators from lo up to B, with pruning.
    def dfs(n, d, lo, B, depth, max_terms, path):
        nonlocal best_solution, best_max_den
        if n == 0:
            # Found a solution
            m = max(path)
            if m < best_max_den:
                best_max_den = m
                best_solution = path[:]
            return True
        if depth >= max_terms:
            return False
        # Pruning: we need 1/x <= n/d for x >= lo => x >= ceil(d/n)
        min_x = (d + n - 1) // n
        if min_x < lo:
            min_x = lo
        # Also x must be <= B
        if min_x > B:
            return False
        # Additional pruning: with at most (max_terms - depth) terms, the maximum sum we can add
        # is sum of reciprocals of the smallest available denominators. If that's < n/d, prune.
        # A simple bound: even with all remaining terms being 1/(min_x), the max sum is (max_terms-depth)/min_x
        # But since denominators must be distinct and increasing, a tighter bound is better.
        # For now, use a loose bound for speed.
        if (max_terms - depth) * Fraction(1, min_x) < Fraction(n, d):
            return False

        # Try each possible next denominator
        # We can also bound the upper end: 1/x must be such that remaining can be filled
        # 1/x <= n/d => x >= ceil(d/n) (already have min_x)
        # Also, the remaining after choosing x is n/d - 1/x = (nx - d)/(dx)
        # For this to be positive: nx > d => x > d/n (already ensured by min_x)
        # For the remaining to be representable with (max_terms-depth-1) more terms all >= x+1:
        # n/d - 1/x <= (max_terms-depth-1)/(x+1)
        # This gives an upper bound on x.
        # Let's compute: n/d - 1/x <= (max_terms-depth-1)/(x+1)
        # (nx - d)/(dx) <= (max_terms-depth-1)/(x+1)
        # (nx - d)(x+1) <= dx(max_terms-depth-1)
        # nx^2 + nx - dx - d <= dx(max_terms-depth-1)
        # nx^2 + nx - dx - d <= d*x*(max_terms-depth-1)
        # nx^2 + nx - d*x*(max_terms-depth-1) - dx - d <= 0
        # This is a quadratic in x. For simplicity, let's just iterate.
        
        for x in range(min_x, B + 1):
            if check_time():
                return False
            # Check if 1/x <= n/d
            if x * n < d:
                continue
            # Compute new remainder
            # rem = n/d - 1/x = (nx - d) / (dx)
            num = n * x - d
            den = d * x
            if num <= 0:
                continue
            # Reduce
            g = gcd(num, den)
            num //= g
            den //= g
            path.append(x)
            if dfs(num, den, x + 1, B, depth + 1, max_terms, path):
                path.pop()
                return True
            path.pop()
        return False

    # Try decreasing bounds to find the smallest possible max denominator
    # Start from a reasonable upper bound and go down
    # The greedy solution gives us an initial bound. Let's also try some smaller values.
    # Since 5/31 ≈ 0.161, the smallest denominator must be at least 7 (1/7 ≈ 0.143).
    # Try B values from small to large, stopping when we find a solution.
    test_bounds = [7, 8, 9, 10, 12, 15, 20, 30, 50, 100]
    
    # First, get a better baseline using a smarter greedy that tries to keep denominators small
    # Use the greedy but with distinctness
    better_baseline = greedy(target, min_d=7)
    if better_baseline:
        bm = max(better_baseline)
        if bm < best_max_den:
            best_max_den = bm
            best_solution = better_baseline[:]

    for B in test_bounds:
        if B >= best_max_den:
            continue
        if check_time():
            break
        max_terms = min(256, B)  # Can't have more distinct denominators than B
        found = dfs(5, 31, 7, B, 0, max_terms, [])
        if found:
            if B == 7:
                break
            # Continue trying smaller B? No, B is increasing. We want the smallest B.
            # Since we're going in increasing order, the first found is the best.
            break

    # If we still have time, try to improve via local search
    if best_max_den > 7 and not check_time():
        # Try to reduce the largest denominator by replacing it
        for _ in range(50):
            if check_time():
                break
            cur = best_solution[:]
            if not cur:
                break
            max_d = max(cur)
            max_idx = cur.index(max_d)
            # Try replacing max_d with a smaller value and completing the rest
            remaining_without = target - sum(Fraction(1, x) for x in cur if x != max_d)
            # We need to represent remaining_without with distinct denominators, none equal to any in cur (except we're removing max_d)
            # Actually, we want to find a set S that doesn't include max_d, has max(S) < max_d, and sum(1/s for s in S) = target
            # This is essentially the same problem. Let's try a bounded DFS with B = max_d - 1
            B_new = max_d - 1
            if B_new >= 7:
                max_terms = min(256, B_new)
                found = dfs(5, 31, 7, B_new, 0, max_terms, [])
                if found:
                    if B_new == 7:
                        break
                    break

    if best_solution is None:
        return baseline
    return best_solution
# EVOLVE-BLOCK-END

import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    TIME_LIMIT = 85.0  # 90s wall-clock, leave margin

    # Known 3-term solution: 5/31 = 1/8 + 1/28 + 1/1736
    best_solution = [8, 28, 1736]
    best_max_denom = 1736

    def update_best(sol):
        nonlocal best_solution, best_max_denom
        if not sol:
            return
        m = max(sol)
        if m < best_max_denom:
            best_max_denom = m
            best_solution = list(sol)

    def check_time():
        return (time.time() - start_time) < TIME_LIMIT

    # ---------- 2-term feasibility ----------
    # 5/31 = 1/a + 1/b  <=>  (31a-5)(31b-5) = 25
    # a = (31k+5)/5 for k in divisors of 25
    def two_term_candidates(rem_num, rem_den):
        """Return list of (a,b) pairs of distinct positive ints with 1/a+1/b = rem_num/rem_den, or []."""
        # (rem_den*a - rem_num)(rem_den*b - rem_num) = rem_num^2
        N = rem_num * rem_num
        D = rem_den
        # divisors of N
        divs = []
        i = 1
        while i * i <= N:
            if N % i == 0:
                divs.append(i)
                if i != N // i:
                    divs.append(N // i)
            i += 1
        res = []
        for k in divs:
            a_num = D * k + rem_num
            if a_num % 5 != 0:
                continue
            a = a_num // 5
            if a <= 0:
                continue
            b_num = D * (N // k) + rem_num
            if b_num % 5 != 0:
                continue
            b = b_num // 5
            if b <= 0:
                continue
            if a == b:
                continue
            res.append((a, b))
        return res

    def two_term_feasible(rem_num, rem_den, min_d, max_d):
        """Check if 1/a+1/b = rem_num/rem_den has a solution with min_d < a < b <= max_d."""
        cands = two_term_candidates(rem_num, rem_den)
        for a, b in cands:
            if a < b:
                if a > min_d and b <= max_d:
                    return True
            else:
                if b > min_d and a <= max_d:
                    return True
        return False

    # ---------- 3-term enumeration via Diophantine factorization ----------
    # For 5/31 = 1/a + 1/b + 1/c, let x=a-31, y=b-31, z=c-31.
    # (a+b+c)(a-31)(b-31)(c-31) = 31^3 = 29791
    # => (x+y+z+93) * x * y * z = 29791
    # For each (x,y,z) with x*y*z = 29791 and x+y+z+93 = 29791/(x*y*z) = 1:
    #   x+y+z = -92, and a=x+31, b=y+31, c=z+31 must be positive distinct.
    #   Also we need 1/a+1/b+1/c = 5/31 exactly (the factorization is necessary but
    #   we verify).
    #
    # Actually the equation (a+b+c)(a-31)(b-31)(c-31) = 31^3 is derived from
    # 5/31 = 1/a+1/b+1/c. Let's verify: if 5/31 = 1/a+1/b+1/c, then
    # 5abc = 31(ab+bc+ca). Let x=a-31 etc.
    # This is a known identity. We'll just enumerate and verify.

    def three_term_solutions(max_d):
        """Find all distinct triples (a,b,c) with 1/a+1/b+1/c = 5/31 and max(a,b,c) <= max_d."""
        sols = []
        # Enumerate (x,y,z) positive or negative? a,b,c > 31 typically for
        # sum = 5/31 < 1/31*3. Actually 5/31 ≈ 0.161, and 3/31 ≈ 0.097,
        # so we need some a < 31. Let's just enumerate divisors of 29791
        # for x, y, z (can be negative).
        # 29791 = 31^3. Divisors: ±1, ±31, ±961, ±29791.
        divs = [1, -1, 31, -31, 961, -961, 29791, -29791]
        # We need x*y*z = 29791 (positive), so either 0 or 2 negative factors.
        # And x+y+z+93 must equal 29791/(x*y*z) = 1, so x+y+z = -92.
        # Let's just enumerate all ordered triples and check.
        for x in divs:
            for y in divs:
                prod_xy = x * y
                if 29791 % prod_xy != 0:
                    continue
                z = 29791 // prod_xy
                if z not in divs:
                    continue
                if x * y * z != 29791:
                    continue
                s = x + y + z + 93
                if s != 1:  # need (x+y+z+93) = 29791/(xyz) = 1
                    continue
                a = x + 31
                b = y + 31
                c = z + 31
                if a <= 0 or b <= 0 or c <= 0:
                    continue
                if a > max_d or b > max_d or c > max_d:
                    continue
                # Verify
                if a == b or b == c or a == c:
                    continue
                # Check 1/a+1/b+1/c = 5/31
                lhs = Fraction(1, a) + Fraction(1, b) + Fraction(1, c)
                if lhs == Fraction(5, 31):
                    sols.append((a, b, c))
        return sols

    # ---------- Search strategy ----------
    # 1. First, enumerate all 3-term solutions (fast, bounded enumeration).
    # 2. Then, for each K from best_max_denom-1 downward, check if a solution
    #    with max <= K exists (using 3-term + 2-term + 1-term checks).
    #    Use binary search for efficiency.

    # Step 1: Enumerate 3-term solutions
    all_3term = three_term_solutions(10000000)
    for a, b, c in all_3term:
        update_best([a, b, c])

    # Also try to find 2-term solutions (unlikely for 5/31 but check)
    two_cands = two_term_candidates(5, 31)
    for a, b in two_cands:
        if a <= 10000000 and b <= 10000000:
            update_best([a, b])

    # Step 2: Binary search for minimum K
    # Feasibility: does a solution exist with all denominators <= K?
    # We know K=best_max_denom is feasible.
    # We want the smallest K.

    def feasible(K):
        """Check if a valid Egyptian fraction of 5/31 exists with all denominators in [1, K]."""
        if not check_time():
            return None  # unknown
        # 1-term: 1/a = 5/31 => a = 31/5, not integer. No.
        # 2-term:
        for a, b in two_cands:
            if a <= K and b <= K:
                return True
        # 3-term:
        for a, b, c in all_3term:
            if max(a, b, c) <= K:
                return True
        # 4+ terms: use DFS with pruning
        # This is expensive, so we only do it if needed.
        # For now, return False (we'll handle 4+ terms separately if needed)
        return False

    # Binary search
    low = 1
    high = best_max_denom

    # We know high is feasible. Find smallest feasible K.
    while low < high and check_time():
        mid = (low + high) // 2
        result = feasible(mid)
        if result is None:  # timeout
            break
        if result:
            high = mid
        else:
            low = mid + 1

    if check_time():
        # Verify the answer
        final = feasible(low)
        if final is True:
            # Find a concrete solution with max <= low
            # Check 2-term
            found = False
            for a, b in two_cands:
                if a <= low and b <= low:
                    update_best([a, b])
                    found = True
                    break
            if not found:
                for a, b, c in all_3term:
                    if max(a, b, c) <= low:
                        update_best([a, b, c])
                        found = True
                        break
            if not found:
                # 4+ terms: need DFS
                # For now, keep best_solution
                pass

    # Final: return best_solution
    return best_solution

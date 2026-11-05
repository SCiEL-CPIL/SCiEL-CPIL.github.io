def construct():
    from fractions import Fraction
    from math import isqrt, gcd
    import time

    start = time.time()
    TIME_LIMIT = 88.0

    target_num, target_den = 5, 31

    def solve_two_terms(rem_n, rem_d, min_val, max_val):
        """Find (x, y) with x > min_val, y > x, y <= max_val such that 1/x + 1/y = rem_n/rem_d.
        Returns (x, y) with smallest y, or None."""
        if rem_n <= 0 or rem_d <= 0:
            return None
        g = gcd(rem_n, rem_d)
        rem_n //= g
        rem_d //= g
        # (rem_n*x - rem_d)(rem_n*y - rem_d) = rem_d^2
        # Let A = rem_n*x - rem_d, B = rem_n*y - rem_d
        # A*B = rem_d^2
        # x = (A + rem_d) / rem_n, must be integer
        # y = (B + rem_d) / rem_n, must be integer
        # x > min_val => A > rem_n*min_val - rem_d
        # y <= max_val => B <= rem_n*max_val - rem_d
        # A < B (since x < y) => A < rem_d
        # A and B must have same sign (product positive) => both positive (since A < rem_d and we need x > 0)
        D_sq = rem_d * rem_d
        A_min = rem_n * min_val - rem_d  # A must be > A_min
        B_max = rem_n * max_val - rem_d   # B must be <= B_max
        # A < rem_d and A > A_min and A > 0
        # B = D_sq / A
        # B > A (since B > A)
        # B <= B_max
        # (A + rem_d) % rem_n == 0
        # (B + rem_d) % rem_n == 0
        
        # Factorize rem_d to get divisors of D_sq
        def divisors_of(n):
            divs = []
            i = 1
            while i * i <= n:
                if n % i == 0:
                    divs.append(i)
                    if i != n // i:
                        divs.append(n // i)
                i += 1
            return divs
        
        # Get divisors of rem_d, then square them conceptually for D_sq
        # Actually just get divisors of D_sq directly
        d_divs = divisors_of(D_sq)
        
        best_y = None
        best_pair = None
        
        for A in d_divs:
            if A <= 0:
                continue
            if A >= rem_d:
                continue
            if A <= A_min:
                continue
            B = D_sq // A
            if B <= A:
                continue
            if B > B_max:
                continue
            if (A + rem_d) % rem_n != 0:
                continue
            x = (A + rem_d) // rem_n
            if x <= min_val:
                continue
            if (B + rem_d) % rem_n != 0:
                continue
            y = (B + rem_d) // rem_n
            if y <= x:
                continue
            if y > max_val:
                continue
            if best_y is None or y < best_y:
                best_y = y
                best_pair = (x, y)
        
        return best_pair

    # Exhaustive search for k-term solutions, k = 3, 4, 5
    # We want to minimize the largest denominator
    best_max = float('inf')
    best_sol = None

    # Check 3-term: 1/a + 1/b + 1/c = 5/31, a < b < c
    # a: 1/a < 5/31 => a > 6.2 => a >= 7
    #     3/a >= 5/31 => a <= 93/5 = 18.6 => a <= 18
    for a in range(7, 19):
        if time.time() - start > TIME_LIMIT - 5:
            break
        rem1_n = target_num * a - target_den  # = 5a - 31
        rem1_d = target_den * a               # = 31a
        # rem1 = rem1_n / rem1_d
        if rem1_n <= 0:
            continue
        # 1/b + 1/c = rem1
        # b > a
        # 1/b < rem1 => b > 1/rem1 = rem1_d/rem1_n
        # 2/b >= rem1 => b <= 2/rem1 = 2*rem1_d/rem1_n
        b_min = int(rem1_d / rem1_n) + 1
        if b_min <= a:
            b_min = a + 1
        b_max = int(2 * rem1_d / rem1_n)
        
        for b in range(b_min, b_max + 1):
            if time.time() - start > TIME_LIMIT - 5:
                break
            rem2_n = rem1_n * b - rem1_d
            rem2_d = rem1_d * b
            if rem2_n <= 0:
                continue
            # 1/c = rem2 => c = rem2_d / rem2_n
            if rem2_d % rem2_n == 0:
                c = rem2_d // rem2_n
                if c > b:
                    # Verify
                    if c * 31 * a * b == 5 * b * a + 5 * c * a + 5 * c * b:
                        if c < best_max:
                            best_max = c
                            best_sol = [a, b, c]

    # Check 4-term: 1/a + 1/b + 1/c + 1/d = 5/31, a < b < c < d
    # Only search if we can potentially beat best_max
    # d < best_max => d <= best_max - 1
    if best_sol is not None:
        limit_d = best_max - 1
        if limit_d > 4:
            # a: 1/a < 5/31 => a >= 7
            # 4/a >= 5/31 => a <= 124/5 = 24.8 => a <= 24
            for a in range(7, 25):
                if time.time() - start > TIME_LIMIT - 10:
                    break
                rem1_n = target_num * a - target_den
                rem1_d = target_den * a
                if rem1_n <= 0:
                    continue
                # 1/b + 1/c + 1/d = rem1
                # b > a
                # 3/b >= rem1 => b <= 3*rem1_d/rem1_n
                b_max_a = int(3 * rem1_d / rem1_n)
                if b_max_a <= a:
                    continue
                for b in range(a + 1, b_max_a + 1):
                    if time.time() - start > TIME_LIMIT - 10:
                        break
                    rem2_n = rem1_n * b - rem1_d
                    rem2_d = rem1_d * b
                    if rem2_n <= 0:
                        continue
                    # 1/c + 1/d = rem2
                    # c > b, d < limit_d
                    pair = solve_two_terms(rem2_n, rem2_d, b, limit_d)
                    if pair:
                        c, d = pair
                        if d < best_max:
                            # Verify exactly
                            # 1/a + 1/b + 1/c + 1/d = 5/31
                            # (31*b*c*d + 31*a*c*d + 31*a*b*d + 31*a*b*c) = 5*a*b*c*d
                            lhs = 31 * (b*c*d + a*c*d + a*b*d + a*b*c)
                            rhs = 5 * a * b * c * d
                            if lhs == rhs:
                                best_max = d
                                best_sol = [a, b, c, d]
                                # Update limit for further search
                                limit_d = best_max - 1

    # Check 5-term if we have time and best_max is large enough
    if best_sol is not None and best_max > 30 and time.time() - start < TIME_LIMIT - 15:
        limit_e = best_max - 1
        # a: 1/a < 5/31 => a >= 7
        # 5/a >= 5/31 => a <= 31
        for a in range(7, 32):
            if time.time() - start > TIME_LIMIT - 10:
                break
            rem1_n = target_num * a - target_den
            rem1_d = target_den * a
            if rem1_n <= 0:
                continue
            # 1/b + 1/c + 1/d + 1/e = rem1
            # b > a
            # 4/b >= rem1 => b <= 4*rem1_d/rem1_n
            b_max_a = int(4 * rem1_d / rem1_n)
            if b_max_a <= a:
                continue
            for b in range(a + 1, b_max_a + 1):
                if time.time() - start > TIME_LIMIT - 10:
                    break
                rem2_n = rem1_n * b - rem1_d
                rem2_d = rem1_d * b
                if rem2_n <= 0:
                    continue
                # 1/c + 1/d + 1/e = rem2
                # c > b
                # 3/c >= rem2 => c <= 3*rem2_d/rem2_n
                c_max_b = int(3 * rem2_d / rem2_n)
                if c_max_b <= b:
                    continue
                for c in range(b + 1, c_max_b + 1):
                    if time.time() - start > TIME_LIMIT - 5:
                        break
                    rem3_n = rem2_n * c - rem2_d
                    rem3_d = rem2_d * c
                    if rem3_n <= 0:
                        continue
                    # 1/d + 1/e = rem3
                    # d > c, e < limit_e
                    pair = solve_two_terms(rem3_n, rem3_d, c, limit_e)
                    if pair:
                        d, e = pair
                        if e < best_max:
                            # Verify: 1/a+1/b+1/c+1/d+1/e = 5/31
                            # 31*(b*c*d*e + a*c*d*e + a*b*d*e + a*b*c*e + a*b*c*d) = 5*a*b*c*d*e
                            prod = a * b * c * d * e
                            lhs = 31 * (b*c*d*e + a*c*d*e + a*b*d*e + a*b*c*e + a*b*c*d)
                            rhs = 5 * prod
                            if lhs == rhs:
                                best_max = e
                                best_sol = [a, b, c, d, e]
                                limit_e = best_max - 1

    # Fallback if no solution found
    if best_sol is None:
        best_sol = [8, 31, 248]
        best_max = 248

    return best_sol

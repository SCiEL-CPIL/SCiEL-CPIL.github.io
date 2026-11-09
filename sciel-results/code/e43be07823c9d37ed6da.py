import time

def construct():
    deadline = time.time() + 80  # 80s safety margin within 90s limit
    answers = [None] * 254  # for n=3..256

    for idx, n in enumerate(range(3, 257)):
        if time.time() > deadline:
            break
        found = None
        # Try x = n + k for k = 1, 2, 3, ... up to some limit
        for k in range(1, 11):
            if time.time() > deadline:
                break
            x = n + k
            a = 4 * x - n  # = 4(n+k) - n = 3n + 4k
            b = n * x      # = n(n+k)
            # Need (ay-b)(az-b) = b^2
            # Let d = ay - b, then az - b = b^2/d
            # So d must divide b^2, and d ≡ -b (mod a), and b^2/d ≡ -b (mod a)
            # Also y = (d+b)/a > x, z = (b^2/d + b)/a > y, all distinct
            b_sq = b * b
            # Enumerate divisors of b^2 by enumerating divisors of b
            # b = n*(n+k), factorize both
            # For efficiency, iterate d over divisors of b_sq where d >= b (so y >= x roughly)
            # Actually we need d such that d ≡ -b (mod a)
            # d = a*m - b for some m, and d divides b^2
            # So a*m - b | b^2
            # Let's iterate m from 1 upward
            # d = a*m - b must be positive and divide b^2
            # d > 0 => a*m > b => m > b/a = n(n+k)/(3n+4k)
            # For k=1: m > n(n+1)/(3n+4) ≈ n/3 for large n
            # z = (b^2/d + b)/a, need z > y > x, all distinct
            # y = (d+b)/a = m, so y = m!
            # Wait: d = a*m - b, so y = (d+b)/a = (a*m - b + b)/a = m
            # So y = m directly!
            # And z = (b^2/d + b)/a where d = a*m - b
            # z = (b^2/(a*m - b) + b)/a
            # Need z to be integer, z > y = m, z != x, z != y
            # b^2 must be divisible by (a*m - b)
            # Start m from max(x+1, floor(b/a)+1)
            m_start = max(x + 1, b // a + 1)
            # Upper bound: d = a*m - b, z = (b^2 + b*(a*m-b))/(a*(a*m-b)) = b*(b + a*m - b)/(a*(a*m-b)) = b*a*m/(a*(a*m-b)) = b*m/(a*m-b)
            # z = b*m / (a*m - b)
            # Need z > m => b*m/(a*m-b) > m => b/(a*m-b) > 1 => b > a*m - b => 2b > a*m => m < 2b/a
            m_end = min(2 * b // a, 10**12 // x)  # z <= 10^12 constraint
            if m_end < m_start:
                continue
            for m in range(m_start, m_end + 1):
                d = a * m - b
                if d <= 0:
                    continue
                if (b_sq) % d != 0:
                    continue
                z = b * m // d
                if z <= m:
                    continue
                if z > 10**12:
                    continue
                # Check distinctness
                if x == y or x == z or m == z:
                    continue
                # y = m
                if x == m:
                    continue
                # Verify the equation
                # 1/x + 1/m + 1/z should equal 4/n
                # Cross multiply: n*(m*z + x*z + x*m) == 4*x*m*z
                lhs = n * (m * z + x * z + x * m)
                rhs = 4 * x * m * z
                if lhs == rhs:
                    found = [x, m, z]
                    break
            if found:
                break
        answers[idx] = found
    return answers

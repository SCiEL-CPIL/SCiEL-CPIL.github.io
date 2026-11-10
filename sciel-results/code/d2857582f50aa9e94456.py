import time
import numpy as np

def construct():
    MAX_DENOM = 10**12
    DEADLINE = time.time() + 85.0
    answers = [None] * 254

    def get_divisors(num):
        if num == 0:
            return [0]
        if num == 1:
            return [1]
        factors = {}
        d = 2
        temp = num
        while d * d <= temp:
            while temp % d == 0:
                factors[d] = factors.get(d, 0) + 1
                temp //= d
            d += 1
        if temp > 1:
            factors[temp] = factors.get(temp, 1)
        divisors = [1]
        for prime, exp in factors.items():
            new_divs = []
            for dd in divisors:
                pwr = 1
                for e in range(exp + 1):
                    new_divs.append(dd * pwr)
                    pwr *= prime
            divisors = new_divs
        return divisors

    for n in range(3, 257):
        idx = n - 3
        if time.time() > DEADLINE:
            answers[idx] = None
            continue

        n_sq = n * n
        divs_n2 = get_divisors(n_sq)
        divs_n2_set = set(divs_n2)

        found = None
        # Try x from smallest valid value upward
        # x > n/4, so x >= floor(n/4) + 1
        x_min = n // 4 + 1
        # Limit x range; for most n, small x works
        for x in range(x_min, n + 200):
            if time.time() > DEADLINE:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            b_sq = b * b  # n^2 * x^2

            # We need d | b_sq such that (d + b) % a == 0 and (b_sq//d + b) % a == 0
            # d = d1 * d2 where d1 | n^2, d2 | x^2
            # Enumerate divisors of x^2 (small) and combine with divisors of n^2
            divs_x2 = get_divisors(x * x)

            # Build candidate d values: for each d2 in divs_x2, d = d1 * d2 for d1 in divs_n2
            # But that's too many. Instead, iterate d2 and check divisibility conditions.
            # Actually, let's just iterate over divs_x2 and for each, the corresponding d1 must satisfy
            # d = d1 * d2 divides b_sq = n^2 * x^2. Since d1|n^2 and d2|x^2, d|b_sq automatically.
            # But we also need d to be a divisor, which it is.
            # The key constraint is (d + b) % a == 0 and (b_sq//d + b) % a == 0

            # For each d2 in divs_x2, d = d1 * d2. We need (d1*d2 + b) % a == 0.
            # So d1*d2 ≡ -b (mod a). This is a modular constraint on d1.
            # This is complex. Let's just enumerate all divisors of b_sq for small b_sq,
            # or use the factorization approach more efficiently.

            # Alternative: enumerate divisors of b_sq directly using combined factorization
            # Factor b_sq = n^2 * x^2. We have factorizations of n^2 and x^2.
            # Merge them.
            # For efficiency, let's get divisors of x^2 (small) and for each d2,
            # the divisors d1 of n^2 such that d1*d2 gives valid y,z.

            # Let's try: for each d2 in divs_x2, compute what d1 must satisfy.
            # d = d1 * d2
            # (d + b) % a == 0  =>  (d1*d2 + b) % a == 0
            # (b_sq//d + b) % a == 0  =>  (n^2*x^2/(d1*d2) + b) % a == 0
            #   = (n^2 * (x^2/d2) / d1 + b) % a == 0
            #   Since d2 | x^2, let x2_over_d2 = x*x//d2 (integer)
            #   = (n^2 * x2_over_d2 / d1 + b) % a == 0

            # This is getting complex. Let's just enumerate divisors of b_sq for reasonable sizes.
            # For n up to 256, x up to ~456, b_sq up to (256*456)^2 ≈ 1.36e11, which is manageable.

            # Actually, let's use a different approach: iterate over possible y values.
            # From 1/y + 1/z = a/b, we get (ay-b)(az-b) = b^2.
            # So ay - b must be a positive divisor of b^2 (since y > b/a for 1/z > 0).
            # y > b/a = n*x/(4x-n). For x close to n/4, this is large.
            # Let's iterate over divisors of b^2.

            # Get divisors of b_sq
            divs_bs = get_divisors(b_sq)

            # For each divisor d of b_sq, check if (d + b) % a == 0
            # Use numpy for vectorized check
            d_arr = np.array(divs_bs, dtype=np.int64)
            # Filter: d + b must be divisible by a, and d + b > 0 (always true since d>0, b>0)
            mask = (d_arr + b) % a == 0
            valid_d = d_arr[mask]

            if len(valid_d) == 0:
                continue

            # For each valid d, compute y and z
            # y = (d + b) // a
            # z = (b_sq // d + b) // a
            # Check z is integer: b_sq // d must be integer (it is, since d | b_sq)
            # and (b_sq//d + b) % a == 0

            # Vectorize z check
            # b_sq // d for each d in valid_d
            # Use integer division
            rem = b_sq // valid_d  # This works since d | b_sq
            z_mask = (rem + b) % a == 0
            valid_d2 = valid_d[z_mask]
            rem2 = rem[z_mask]

            if len(valid_d2) == 0:
                continue

            y_vals = (valid_d2 + b) // a
            z_vals = (rem2 + b) // a

            # Filter: y > x, z > y (or just y != x, z != x, z != y), y,z <= MAX_DENOM
            # We want distinct: x, y, z all different
            # Let's just check: y > 0, z > 0, y != x, z != x, y != z, y <= MAX_DENOM, z <= MAX_DENOM
            # And verify the equation.

            y_arr = y_vals
            z_arr = z_vals

            # Basic filters
            mask2 = (y_arr > 0) & (z_arr > 0) & (y_arr != x) & (z_arr != x) & (y_arr != z_arr)
            mask2 &= (y_arr <= MAX_DENOM) & (z_arr <= MAX_DENOM)

            if not np.any(mask2):
                continue

            y_cand = y_arr[mask2]
            z_cand = z_arr[mask2]

            # Verify the equation: 1/x + 1/y + 1/z = 4/n
            # x*y*z*4 == n*(y*z + x*z + x*y)
            # Pick the first candidate and verify
            for i in range(len(y_cand)):
                yv = int(y_cand[i])
                zv = int(z_cand[i])
                if x * yv * zv * 4 == n * (yv * zv + x * zv + x * yv):
                    found = [x, yv, zv]
                    break

            if found:
                break

        answers[idx] = found

    return answers

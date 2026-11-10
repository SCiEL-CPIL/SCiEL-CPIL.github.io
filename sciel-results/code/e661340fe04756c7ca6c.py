import time

# EVOLVE-BLOCK-START
def construct():
    MAX_DENOM = 10**12
    deadline = time.time() + 50.0  # safety margin before 60s limit
    answers = [None] * 254

    for i, n in enumerate(range(3, 257)):
        if time.time() > deadline - 0.5:
            break
        found = None
        # x must satisfy: x >= n/4 + 1 (so 4x - n > 0), and y_range nonempty
        # y_min = ceil(2nx/(4x-n)), y_max = floor(nx/(2x-n))
        # Need y_min < y_max (strict, since z>y requires y > 2nx/(4x-n))
        # x range: from floor(n/4)+1 up to 2n-1 (where 2x-n > 0)
        x_start = n // 4 + 1
        if n % 4 == 0:
            x_start = n // 4 + 1  # 4x - n > 0 requires x > n/4
        x_end = 2 * n - 1

        for x in range(x_start, x_end + 1):
            if time.time() > deadline - 0.5:
                break
            a = 4 * x - n  # > 0
            b = n * x
            # y_min = ceil(2*b / a), y_max = floor(b / (2*x - n))
            two_x_minus_n = 2 * x - n
            if two_x_minus_n <= 0:
                continue
            y_min = (2 * b + a - 1) // a  # ceil(2b/a)
            y_max = b // two_x_minus_n     # floor(b/(2x-n))
            if y_min >= y_max:
                continue
            # Also need z > y and z <= MAX_DENOM
            # z = b*y / (a*y - b), need a*y - b > 0 i.e. y > b/a
            # y_min already ensures y > 2b/a > b/a, so a*y - b > 0
            for y in range(y_min, y_max + 1):
                denom = a * y - b  # = b*(y - b/(a*y))... actually a*y - b
                if denom <= 0:
                    continue
                num = b * y
                if num % denom == 0:
                    z = num // denom
                    if z > y and z <= MAX_DENOM and x <= MAX_DENOM and y <= MAX_DENOM:
                        # Verify: 1/x + 1/y + 1/z == 4/n
                        # Cross multiply: n*(y*z + x*z + x*y) == 4*x*y*z
                        lhs = n * (y * z + x * z + x * y)
                        rhs = 4 * x * y * z
                        if lhs == rhs:
                            found = [x, y, z]
                            break
            if found:
                break
        answers[i] = found

    return answers
# EVOLVE-BLOCK-END

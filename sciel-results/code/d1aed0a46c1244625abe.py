import time

def construct():
    deadline = time.time() + 85.0
    answers = [None] * 254
    BOUND = 10**12

    def find_solution_for_n(n, x_start, x_end, max_k_per_x, check_time=True):
        """Find a distinct unit fraction triple for 1/x+1/y+1/z=4/n."""
        for x in range(x_start, x_end + 1):
            if check_time and time.time() > deadline - 3.0:
                return None

            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x

            # k = y, where (k*a - b) must divide b^2
            # k > b/a and k < 2b/a for z > y
            k_lo = max(x + 1, b // a + 1)
            k_hi = (2 * b - 1) // a  # largest integer < 2b/a

            if k_hi < k_lo:
                continue

            # Cap the range to avoid excessive iteration
            if k_hi - k_lo > max_k_per_x:
                k_hi = k_lo + max_k_per_x

            b_sq = b * b

            for k in range(k_lo, k_hi + 1):
                d = k * a - b
                if d <= 0:
                    continue
                if b_sq % d != 0:
                    continue
                e = b_sq // d
                z_num = e + b
                if z_num % a != 0:
                    continue
                z = z_num // a
                y = k

                if y <= x or z <= y:
                    continue
                if x > BOUND or y > BOUND or z > BOUND:
                    continue

                # Verify with cross multiplication
                if n * y * z + n * x * z + n * x * y == 4 * x * y * z:
                    if x != y and y != z and x != z:
                        return [x, y, z]

        return None

    # First pass: quick search for each n
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline - 5.0:
            break

        x_min = n // 4 + 1
        found = find_solution_for_n(n, x_min, x_min + 100, 5000)
        answers[idx] = found

    # Second pass: extended search for unsolved n's
    for n in range(3, 257):
        idx = n - 3
        if answers[idx] is not None:
            continue
        if time.time() > deadline - 15.0:
            break

        x_min = n // 4 + 1
        found = find_solution_for_n(n, x_min, x_min + 500, 20000)
        answers[idx] = found

    # Third pass: even more extended search for remaining unsolved n's
    for n in range(3, 257):
        idx = n - 3
        if answers[idx] is not None:
            continue
        if time.time() > deadline - 2.0:
            break

        x_min = n // 4 + 1
        found = find_solution_for_n(n, x_min, x_min + 2000, 100000)
        answers[idx] = found

    return answers

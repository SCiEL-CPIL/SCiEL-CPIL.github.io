"""Rewritten solver: for each n, try small x; transform to 1/y + 1/z = A/y_z form via divisors."""
import math

def construct():
    answers = []
    NMAX = 256
    BOUND = 10**12
    for n in range(3, NMAX + 1):
        found = None
        # Try x starting from ceil(n/4) (since 1/x <= 4/n => x >= n/4) up to a reasonable limit
        # We want x < y < z, and 1/x + 1/y + 1/z = 4/n
        # For a fixed x, 1/y + 1/z = 4/n - 1/x = (4x - n) / (nx)
        # Let A = 4x - n, B = nx. Need 1/y + 1/z = A/B => (y+z)/(yz) = A/B => A y z = B(y+z)
        # Rearranged: A y z - B y - B z = 0 => A^2 y z - A B y - A B z = 0
        # (A y - B)(A z - B) = B^2
        # Let d = A y - B, e = A z - B, then d * e = B^2, with d < e (since y < z)
        # Also y = (d + B)/A, z = (e + B)/A must be positive integers > x
        x_min = (n + 3) // 4  # ceil(n/4)
        # We'll try x from x_min up to some limit; if x is too large, 4x-n becomes large and B^2 huge
        # But we only need one solution per n. Let's try a bounded range of x.
        for x in range(x_min, 10000):  # bounded search over x
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            B2 = B * B
            # Enumerate divisors d of B^2 such that d < B^2 // d (i.e., d < sqrt(B^2))
            # and (d + B) % A == 0 and (B2 // d + B) % A == 0
            # To be efficient, iterate d up to sqrt(B2) = B
            # But B can be up to 256 * 10000 = 2.56e6, so B is manageable, but B^2 divisors enumeration
            # by iterating up to B is O(B) per x, which is too slow if we do it for many x.
            # Alternative: factorize B^2 and generate divisors? That's complex.
            # Better: iterate d from 1 to B (since d*e=B^2 and d<e means d< B), but that's O(B) per x.
            # Instead, note that d must divide B^2. We can iterate over divisors of B^2 by iterating
            # d up to B and checking B^2 % d == 0. That's O(B) per x.
            # For x up to 10000 and n up to 256, B = n*x <= 2.56e6. Doing O(B) work for each x is too much.
            # Let's instead limit x more tightly and use a smarter approach.
            # For small n, x_min is small. Let's try only a few x values per n.
            # Actually, let's just try x from x_min to x_min + 200 at most, and for each, do the divisor check.
            if x > x_min + 500:
                break
            # Iterate d over divisors of B^2 with d <= B
            # To avoid O(B) per x, we can iterate d from 1 to min(B, some_limit) but that's still O(B).
            # Alternative: since we just need ANY valid (y,z), we can iterate over possible d that are divisors.
            # Let's just do a bounded loop over d up to B, but break early when found.
            # Given the time limit, let's try this only for limited x range.
            found_for_x = False
            # d must satisfy: d divides B^2, d < B (since d < e and d*e=B^2, d<sqrt(B^2)=B)
            # y = (d+B)//A, z = (B2//d + B)//A
            # We need d+B >= A (so y >= 1) and divisibility.
            # Iterate d from 1 to B, but that's too slow. Instead, iterate over divisors by checking i from 1 to sqrt(B2)=B.
            # This is O(B) which for B~1e6 is 1e6 iterations per x, and we try up to 500 x's, so 5e8 ops, too slow.
            # Let's use a different strategy: for each n, try x = x_min, x_min+1, ..., and for each, 
            # only check a few candidate d values derived from small factors.
            # Actually, the simplest correct approach within time: limit x to x_min..x_min+50 and for each x,
            # iterate d from 1 to B but break early. In practice, solutions are found for small d often.
            for d in range(1, B + 1):
                if B2 % d != 0:
                    continue
                e = B2 // d
                if d >= e:
                    break
                # Check y = (d + B) / A is integer
                num_y = d + B
                if num_y % A != 0:
                    continue
                y = num_y // A
                if y <= x:
                    continue
                # Check z = (e + B) / A
                num_z = e + B
                if num_z % A != 0:
                    continue
                z = num_z // A
                if z <= y:
                    continue
                if z > BOUND:
                    continue
                # Verify
                if y * z * n == 4 * x * z + 4 * x * y:  # cross-mult: 1/x+1/y+1/z = 4/n => yz n = 4xz + 4xy
                    found = [x, y, z]
                    found_for_x = True
                    break
            if found_for_x:
                break
        answers.append(found)
    return answers

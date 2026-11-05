import math
import time

def construct():
    deadline = time.time() + 80  # 80s out of 90s limit, safety margin
    answers = [None] * 254  # indices 0..253 correspond to n=3..256
    solved = 0

    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline - 5:
            break

        found = None
        # x must satisfy 1/x < 4/n, so x > n/4
        # Also x < n/4 * 3 (since y,z > x means 1/y+1/z < 2/x, so 1/x > 4/n - 2/x => 3/x > 4/n => x < 3n/4)
        # But more tightly: since y > x and z > y, we have 1/x + 1/y + 1/z < 1/x + 1/x + 1/x = 3/x
        # So 4/n < 3/x => x < 3n/4
        # Also 1/x + 1/y + 1/z > 1/x (since y,z positive), so 4/n > 1/x => x > n/4
        
        # For distinct x < y < z:
        # 4/n = 1/x + 1/y + 1/z
        # => (4x - n)/x = 1/y + 1/z
        # Let A = 4x - n, B = nx
        # Then A/y + A/z = B, i.e., A(y+z) = B*y*z
        # => A*y*z - A*y - A*z = 0
        # => (A*y - A)*(A*z - A) = A^2
        # Let u = A*y - A, v = A*z - A, then u*v = A^2, y = (u+A)/A, z = (v+A)/A
        # So y = 1 + u/A, z = 1 + v/A
        # We need y and z to be integers, so A | u and A | v, i.e., u and v are multiples of A
        # u*v = A^2, u = A*k, v = A^2/k for positive integer k
        # y = 1 + k, z = 1 + A^2/(A*k) = 1 + A/k
        # So we need k | A, y = 1+k, z = 1 + A/k
        # Conditions: x < y < z, all distinct, positive integers
        
        x_start = n // 4 + 1
        x_end = min(3 * n // 4 + 1, 10**12)
        
        for x in range(x_start, x_end):
            if time.time() > deadline - 2:
                break
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # y = 1 + k, z = 1 + A/k where k | A
            # Need x < y < z => x < 1+k < 1+A/k
            # => k > x-1 and A/k > k => k^2 < A => k < sqrt(A)
            # Also k | A
            
            # Iterate over divisors k of A where x-1 < k < sqrt(A)
            # y = 1+k, z = 1 + A/k
            # Need z > y => 1 + A/k > 1 + k => A/k > k => k^2 < A
            # Need z integer: k | A (guaranteed by construction)
            # Need all distinct: x != y, x != z, y != z
            # x != y: x != 1+k => k != x-1 (but k > x-1, so ok)
            # x != z: x != 1 + A/k
            # y != z: 1+k != 1+A/k => k != A/k => k^2 != A (but k^2 < A, so ok)
            
            # Find divisors of A in range (x-1, sqrt(A))
            sqrt_A = int(math.isqrt(A))
            # k must satisfy: x-1 < k < sqrt_A and k | A
            # Iterate k from x to sqrt_A-1
            for k in range(max(x, 1), sqrt_A):
                if A % k == 0:
                    y = 1 + k
                    z = 1 + A // k
                    # Check distinctness
                    if x != y and x != z and y != z:
                        # Check bounds
                        if x <= 10**12 and y <= 10**12 and z <= 10**12:
                            found = [x, y, z]
                            break
            if found:
                break
        
        if found:
            answers[idx] = found
            solved += 1
    
    return answers

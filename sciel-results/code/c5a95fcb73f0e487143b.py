import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 84.0

    max_val = 200000
    spf = list(range(max_val + 1))
    for i in range(2, int(math.isqrt(max_val)) + 1):
        if spf[i] == i:
            for j in range(i * i, max_val + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def factorize_fast(num):
        if num <= 1:
            return {}
        factors = {}
        while num > 1:
            if num <= max_val:
                p = spf[num]
            else:
                # Fallback for larger numbers
                p = num
                for d in range(2, int(math.isqrt(num)) + 1):
                    if num % d == 0:
                        p = d
                        break
            count = 0
            while num % p == 0:
                num //= p
                count += 1
            factors[p] = count
        return factors

    def get_divisors_from_factors(factors):
        divisors = [1]
        for p, exp in factors.items():
            new_divs = []
            p_power = 1
            for _ in range(exp + 1):
                for d in divisors:
                    new_divs.append(d * p_power)
                p_power *= p
            divisors = new_divs
        return divisors

    def combine_factors(f1, f2):
        result = dict(f1)
        for p, e in f2.items():
            result[p] = result.get(p, 0) + e
        return result

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def solve_n(n):
        # Find distinct x, y, z > 0 with 1/x + 1/y + 1/z = 4/n
        # Assume x < y < z
        # 1/x + 1/y + 1/z = 4/n
        # (yz + xz + xy) / (xyz) = 4/n
        # n(yz + xz + xy) = 4xyz
        # Rearranging: 4xyz - nxy - nxz - nyz = 0
        # Divide by xyz: 4 - n/z - n/y - n/x = 0
        # 1/x + 1/y + 1/z = 4/n
        
        # For fixed x, solve 1/y + 1/z = 4/n - 1/x = (4x - n) / (nx)
        # Let A = 4x - n, B = nx
        # 1/y + 1/z = A/B
        # (y+z) / (yz) = A/B
        # B(y+z) = Ayz
        # Ayz - By - Bz = 0
        # Multiply by A: A^2 yz - AB y - AB z = 0
        # (Ay - B)(Az - B) = B^2
        # Let u = Ay - B, v = Az - B
        # uv = B^2
        # y = (u+B)/A, z = (v+B)/A
        # We need y, z integers > 0, distinct from x and each other.
        
        # x must be such that 4x - n > 0 => x > n/4
        # Also x < y < z implies 1/x > 1/y > 1/z
        # 3/x > 4/n => x < 3n/4
        # So n/4 < x < 3n/4
        # Also x must be such that A = 4x - n > 0.
        # Minimum x = floor(n/4) + 1
        # Maximum x: since y > x, 1/y < 1/x, so 1/z = 4/n - 1/x - 1/y < 4/n - 1/x - 1/x = 4/n - 2/x
        # This implies z > 1 / (4/n - 2/x) = nx / (4x - 2n) = nx / (2(2x-n))
        # For z to exist, we need 4/n - 1/x - 1/y > 0.
        # Since y > x, 1/y < 1/x, so 4/n - 1/x - 1/y > 4/n - 1/x - 1/x = 4/n - 2/x.
        # We need 4/n - 2/x > 0 => 4x > 2n => x > n/2.
        # Wait, if x <= n/2, then 1/x >= 2/n, so 1/x + 1/y >= 1/x + 1/(x+1) ... 
        # Actually, if x is small, 1/x is large.
        # 1/x + 1/y + 1/z = 4/n.
        # If x < n/4, then 1/x > 4/n, so 1/x + 1/y + 1/z > 4/n. No solution.
        # So x >= n/4 + 1.
        # If x = n/2, 1/x = 2/n. 1/y + 1/z = 2/n.
        # If x > 3n/4, 1/x < 4/(3n). 1/y + 1/z > 4/(3n) (since y,z > x).
        # Max sum = 1/x + 1/x + 1/x = 3/x. If 3/x < 4/n, no solution. x > 3n/4.
        
        min_x = n // 4 + 1
        max_x = (3 * n) // 4
        
        # To maximize chance of finding a solution quickly, we might want to search x in a smart way.
        # The current code iterates x from min_x to max_x.
        # For each x, it iterates divisors of B^2.
        
        n_factors = factorize_fast(n)
        
        for x in range(min_x, max_x + 1):
            A = 4 * x - n
            if A <= 0:
                continue
            
            B = n * x
            B2 = B * B
            
            # We need (Ay-B)(Az-B) = B^2
            # u = Ay - B, v = Az - B
            # u * v = B^2
            # u and v are divisors of B^2.
            # Also u and v must have the same sign as A? No.
            # y = (u+B)/A. Since y > 0 and A > 0, u+B > 0 => u > -B.
            # Since u*v = B^2 > 0, u and v have same sign.
            # If u > 0, v > 0.
            # If u < 0, v < 0. Then y = (u+B)/A < B/A < x (since A = 4x-n, B/A = nx/(4x-n) > x for x > n/2?
            # nx/(4x-n) > x => nx > 4x^2 - nx => 2nx > 4x^2 => n > 2x.
            # If n > 2x, then B/A > x.
            # But we assumed x < y < z.
            # So we should look for u, v > 0.
            
            # u < B (since y > x => (u+B)/A > x => u+B > Ax => u > Ax - B = A(4x-n) - B? No.
            # y > x => (u+B)/A > x => u+B > Ax => u > Ax - B = A(4x-n) - nx = 4x^2 - Ax - nx = 4x^2 - (4x-n)x - nx = 4x^2 - 4x^2 + nx - nx = 0.
            # So u > 0.
            # Similarly z > y => v > u.
            # So we need divisors u of B^2 such that 0 < u < v and u*v = B^2.
            # This means u < B (since u < sqrt(B^2) = B).
            # And u must be such that (u+B) % A == 0.
            
            # Factorize B
            x_factors = factorize_fast(x)
            b_factors = combine_factors(n_factors, x_factors)
            b2_factors = {p: 2 * e for p, e in b_factors.items()}
            
            divisors = get_divisors_from_factors(b2_factors)
            
            # Optimization: only check u < B
            for u in divisors:
                if u >= B:
                    break  # divisors are not sorted, but we can skip large ones? 
                    # Actually get_divisors_from_factors doesn't guarantee sorted order.
                    # Let's just check u < B.
                    continue
                
                if (u + B) % A != 0:
                    continue
                
                v = B2 // u
                if v <= u:
                    continue
                if (v + B) % A != 0:
                    continue
                
                y = (u + B) // A
                z = (v + B) // A
                
                if y > x and z > y and z <= 10**12:
                    return [x, y, z]
        
        return None

    answers = []
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = solve_n(n)
        answers.append(found)
    
    return answers

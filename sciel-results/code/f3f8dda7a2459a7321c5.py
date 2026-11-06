import math
import time

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s kill
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        # We need 1/x + 1/y + 1/z = 4/n with x < y < z (distinct)
        # Assume x < y < z. Then 1/x > 1/3 * (4/n) => x < 3n/4
        # And 3/x > 4/n => x < 3n/4
        # Also 1/x < 4/n => x > n/4
        # So x is in [floor(n/4) + 1, floor(3n/4)]
        
        min_x = n // 4 + 1
        max_x = (3 * n) // 4
        
        # If min_x > max_x, no solution with x<y<z in this order?
        # Wait, for n=3: min_x = 0+1=1, max_x = 9//4=2.
        # x=1: 1/1 + 1/y + 1/z = 4/3 => 1/y + 1/z = 1/3.
        # y > 1. 1/y < 1/3 => y > 3.
        # Try y=4: 1/4 + 1/z = 1/3 => 1/z = 1/12 => z=12.
        # 1, 4, 12 are distinct. 1/1+1/4+1/12 = 12/12+3/12+1/12 = 16/12 = 4/3. Correct.
        
        for x in range(min_x, max_x + 1):
            if time.time() > deadline:
                break
            if found:
                break
            
            # 1/y + 1/z = 4/n - 1/x = (4x - n) / (nx)
            # Let A = 4x - n, B = nx
            # We need A > 0, so 4x > n => x > n/4. This is guaranteed by min_x.
            A = 4 * x - n
            B = n * x
            
            if A <= 0:
                continue
            
            # 1/y + 1/z = A / B
            # (y+z)/(yz) = A/B => B(y+z) = Ayz => Ayz - By - Bz = 0
            # Multiply by A: A^2 yz - AB y - AB z = 0
            # (Ay - B)(Az - B) = B^2
            # Let u = Ay - B, v = Az - B. Then uv = B^2.
            # Since y < z, u < v.
            # y = (u + B) / A, z = (v + B) / A.
            # We need u, v to be divisors of B^2.
            # Also y > x => (u+B)/A > x => u+B > Ax => u > Ax - B = A(x) - B = A(x) - nx.
            # Ax - B = A(x) - nx = (4x-n)x - nx = 4x^2 - nx - nx = 4x^2 - 2nx = 2x(2x-n).
            # Also z > y is guaranteed if v > u.
            # Also y, z must be integers.
            # So u must be a divisor of B^2 such that:
            # 1. u < v = B^2/u => u < B
            # 2. (u + B) % A == 0
            # 3. (v + B) % A == 0
            # 4. y = (u+B)//A > x
            # 5. z = (v+B)//A > y
            # 6. y != z (guaranteed by u != v, i.e., u != B)
            # 7. x, y, z distinct. x < y is enforced. y < z is enforced.
            
            # Iterate through divisors u of B^2 with u < B
            # B can be up to 256 * 192 ~ 49152. B^2 ~ 2.4e9.
            # Finding divisors of B^2 might be slow if done naively for each x.
            # However, x range is small (n/4 to 3n/4, width ~ n/2).
            # For n=256, width is 64.
            # Total iterations of x is sum over n of ~n/2 ~ 256^2/4 ~ 16k.
            # For each x, we factor B^2? Or iterate u from 1 to sqrt(B^2)=B?
            # Iterating u from 1 to B is O(B). B ~ nx ~ n^2.
            # Sum of n^2 for n=3..256 is roughly 256^3/3 ~ 5.5e6.
            # This is very fast in Python.
            
            B_sq = B * B
            # We need u | B_sq and u < B.
            # We can iterate u from 1 to B-1? No, that's too slow if B is large.
            # Better: factorize B, generate divisors of B_sq, filter u < B.
            # Or just iterate u from 1 to int(sqrt(B_sq)) = B?
            # Actually, we can iterate u from 1 to B. If B^2 % u == 0, check conditions.
            # But B can be ~ 50000. Looping 50000 times * 16000 x's = 800M ops. Too slow.
            
            # Better approach: Factorize B.
            # B = n * x.
            # Let's find divisors of B^2.
            # divisors of B^2 are of the form p1^e1 ... pk^ek where B = p1^f1 ... pk^fk and 0 <= ei <= 2fi.
            # Number of divisors is prod(2fi+1).
            # For B ~ 50000, number of divisors is small (usually < 1000).
            # So generating divisors is fast.
            
            # Function to get divisors of a number
            def get_divisors(num):
                divs = []
                i = 1
                while i * i <= num:
                    if num % i == 0:
                        divs.append(i)
                        if i != num // i:
                            divs.append(num // i)
                    i += 1
                return divs
            
            # But get_divisors(B_sq) might be slow if B_sq is large and we do it for each x.
            # B_sq is up to ~ 2.5e9. sqrt is ~ 50000.
            # Looping 50000 times per x is too slow.
            
            # Alternative:
            # u must divide B^2.
            # Let u = k. We need k | B^2.
            # We can iterate k from 1 to B.
            # Wait, is there a faster way?
            # Note that u = Ay - B. Since y > x, u > 2x(2x-n).
            # Also y < z => u < B.
            # So u is in (2x(2x-n), B).
            # And u | B^2.
            
            # Let's try iterating u from max(1, 2*x*(2*x-n)+1) to B-1.
            # And check if B^2 % u == 0.
            # The range size is B - 2x(2x-n).
            # B = nx. 2x(2x-n) = 4x^2 - 2nx.
            # Range size = nx - (4x^2 - 2nx) = 3nx - 4x^2 = x(3n - 4x).
            # Since x < 3n/4, 3n - 4x > 0.
            # Max range size is when x is small?
            # If x ~ n/4, range ~ (n/4)(3n - n) = n^2/2.
            # If x ~ 3n/4, range ~ (3n/4)(3n - 3n) = 0.
            # So for small x, the range is large.
            # For n=256, x=64, range ~ 64 * (768-256) = 64*512 = 32768.
            # Checking B^2 % u == 0 for 32768 values.
            # Total sum of range sizes over all x and n?
            # Sum over n of Sum over x of x(3n-4x).
            # This is roughly integral of x(3n-4x) dx from n/4 to 3n/4.
            # = [1.5n x^2 - 4/3 x^3] ...
            # It scales as n^3.
            # Sum n^3 for n=3..256 is ~ 256^4/4 ~ 1e9.
            # This might be too slow in Python.
            
            # Let's go back to factorization.
            # For each x, compute B = nx.
            # Factorize B.
            # Generate divisors of B^2.
            # Filter divisors u < B.
            # Check conditions.
            
            # Factorization of B (up to 50000) is fast.
            # We can precompute smallest prime factor (SPF) up to 100000.
            
            # Let's precompute SPF up to 100000.
            
        # Precompute SPF
        MAX_VAL = 100000
        spf = list(range(MAX_VAL + 1))
        for i in range(2, int(MAX_VAL**0.5) + 1):
            if spf[i] == i:
                for j in range(i*i, MAX_VAL + 1, i):
                    if spf[j] == j:
                        spf[j] = i
        
        def factorize(num):
            factors = {}
            while num > 1:
                p = spf[num]
                cnt = 0
                while num % p == 0:
                    num //= p
                    cnt += 1
                factors[p] = cnt
            return factors
        
        # We need divisors of B^2.
        # If B = prod p_i^e_i, then B^2 = prod p_i^(2e_i).
        # Divisors are prod p_i^k_i where 0 <= k_i <= 2e_i.
        
        def get_divisors_from_factors(factors):
            divs = [1]
            for p, e in factors.items():
                max_e = 2 * e
                new_divs = []
                for d in divs:
                    for k in range(max_e + 1):
                        new_divs.append(d * (p ** k))
                divs = new_divs
            return divs

        # Re-loop with this approach
        answers = []
        for n in range(3, 257):
            if time.time() > deadline:
                answers.append(None)
                continue
            
            found = None
            min_x = n // 4 + 1
            max_x = (3 * n) // 4
            
            for x in range(min_x, max_x + 1):
                if time.time() > deadline:
                    break
                if found:
                    break
                
                A = 4 * x - n
                B = n * x
                if A <= 0:
                    continue
                
                # Factorize B
                B_factors = factorize(B)
                divs = get_divisors_from_factors(B_factors)
                
                # For each divisor u of B^2:
                # u < B (since u < v and uv=B^2)
                # (u + B) % A == 0
                # y = (u + B) // A
                # y > x
                # z = (B*B//u + B) // A
                # z > y
                # x, y, z distinct
                
                for u in divs:
                    if u >= B:
                        continue
                    if (u + B) % A != 0:
                        continue
                    y = (u + B) // A
                    if y <= x:
                        continue
                    v = B * B // u
                    if (v + B) % A != 0:
                        continue
                    z = (v + B) // A
                    if z <= y:
                        continue
                    # Check distinctness
                    if x != y and y != z and x != z:
                        found = [x, y, z]
                        break
            
            answers.append(found)
        
        return answers

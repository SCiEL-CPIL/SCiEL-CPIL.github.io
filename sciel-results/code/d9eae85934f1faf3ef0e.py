import time

def factorize(n):
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def get_divisors_of_square(factors):
    # Factors of B are p^e. Factors of B^2 are p^(2e).
    # We generate all divisors of B^2.
    divisors = [1]
    for p, e in factors.items():
        max_exp = 2 * e
        new_divs = []
        for d in divisors:
            p_pow = 1
            for k in range(max_exp + 1):
                new_divs.append(d * p_pow)
                p_pow *= p
        divisors = new_divs
    return divisors

def construct():
    deadline = time.time() + 85.0
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            # If time runs out, pad with None
            while len(answers) < 254:
                answers.append(None)
            return answers
            
        found = None
        # x must satisfy 1/x + 1/y + 1/z = 4/n
        # Since x < y < z, 3/x > 4/n => x < 3n/4
        # Also 1/x < 4/n => x > n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4  # x < 3n/4, so max integer is floor(3n/4) - 1 if 3n/4 is integer?
        # Actually x < 3n/4. So x <= ceil(3n/4) - 1.
        # Let's just loop up to 3n/4 + 1 to be safe, but bound is strict.
        # If 3n/4 is integer, x < 3n/4 means x <= 3n/4 - 1.
        # If 3n/4 is not integer, x <= floor(3n/4).
        
        for x in range(x_min, x_max + 2):
            if time.time() > deadline:
                break
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # (ay-b)(az-b) = b^2
            # Let U = ay-b, V = az-b. UV = b^2.
            # y < z => U < V => U < b.
            # y > x => ay-b > ax-b => U > ax-b.
            # Also U must be a divisor of b^2.
            # And U ≡ -b (mod a).
            
            lower_bound = a * x - b
            # U must be > lower_bound.
            # U must be < b.
            # U must divide b^2.
            
            # To avoid generating all divisors for every x (which might be slow if factorization is repeated),
            # we can just iterate U from max(1, lower_bound+1) to b-1?
            # b is up to ~50,000. Range is ~50,000.
            # This is 50k iterations per x.
            # 128 x's per n. 6.4M ops per n.
            # 254 n's. 1.6B ops. Too slow.
            
            # We MUST use divisors.
            # Factorize b.
            # b = n * x. n is small. x is small.
            # We can factorize b quickly.
            
            fac_b = factorize(b)
            divs_b2 = get_divisors_of_square(fac_b)
            
            target_mod = (-b) % a
            
            # Filter divisors
            for u in divs_b2:
                if u >= b:
                    continue
                if u <= lower_bound:
                    continue
                if u % a != target_mod:
                    continue
                
                # Valid U
                # y = (u + b) / a
                # z = (v + b) / a where v = b^2 / u
                # Check divisibility
                num_y = u + b
                if num_y % a != 0:
                    continue
                y = num_y // a
                
                v = (b * b) // u
                num_z = v + b
                if num_z % a != 0:
                    continue
                z = num_z // a
                
                # Check distinctness and bounds
                if y <= x or z <= y:
                    continue
                if x == y or y == z or x == z:
                    continue
                if z > 10**12:
                    continue
                
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers.append(found)
        
    return answers

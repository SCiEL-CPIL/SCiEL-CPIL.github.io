import math

def construct():
    answers = [None] * 254
    
    # Precompute small primes for fast factorization up to max B ~ 256 * 192 = 49152
    # Actually B = n*x. Max n=256, max x < 3*256/4 = 192. Max B = 256*192 = 49152.
    # We need factorization of B.
    
    def get_prime_factors(n):
        factors = {}
        d = 2
        while d * d <= n:
            if n % d:
                d += 1
            else:
                nf = 0
                while not n % d:
                    n //= d
                    nf += 1
                factors[d] = nf
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors

    def generate_divisors(factors):
        divs = [1]
        for p, e in factors.items():
            new_divs = []
            for d in divs:
                for i in range(e + 1):
                    new_divs.append(d * (p ** i))
            divs = new_divs
        return divs

    for idx, n in enumerate(range(3, 257)):
        found = None
        # x must be > n/4 and < 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            A = 4 * x - n
            B = n * x
            
            # We need (Ay - B)(Az - B) = B^2
            # Let u = Ay - B. Then u divides B^2.
            # y = (u + B) / A must be integer.
            # z = (B^2/u + B) / A must be integer.
            # We assume y < z, which implies u < B^2/u, so u < B.
            
            # Factorize B to get divisors of B^2
            # Note: divisors of B^2 are p^(2e) for each p^e in B
            factors_B = get_prime_factors(B)
            
            # Construct factors for B^2
            factors_B2 = {}
            for p, e in factors_B.items():
                factors_B2[p] = 2 * e
            
            divs = generate_divisors(factors_B2)
            
            for u in divs:
                if u >= B:
                    continue
                
                num_y = u + B
                if num_y % A != 0:
                    continue
                y = num_y // A
                
                # Calculate v = B^2 / u
                v = (B * B) // u
                
                num_z = v + B
                if num_z % A != 0:
                    continue
                z = num_z // A
                
                # Check conditions
                # y and z must be positive (guaranteed by u>0, B>0, A>0)
                # Distinctness: x, y, z must be distinct.
                # We enforced y < z by u < B.
                if x == y or x == z:
                    continue
                
                # Denominator bound check (10^12)
                if x > 10**12 or y > 10**12 or z > 10**12:
                    continue
                
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers[idx] = found
        
    return answers

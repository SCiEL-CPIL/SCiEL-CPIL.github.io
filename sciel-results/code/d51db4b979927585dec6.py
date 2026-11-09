import time

def construct():
    answers = [None] * 254
    t0 = time.time()
    DEADLINE = 85.0  # safety margin under 90s wall-clock

    def get_prime_factors(num):
        factors = {}
        d = 2
        while d * d <= num:
            if num % d:
                d += 1
            else:
                nf = 0
                while not num % d:
                    num //= d
                    nf += 1
                factors[d] = nf
        if num > 1:
            factors[num] = factors.get(num, 0) + 1
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
        if time.time() - t0 > DEADLINE:
            break
        
        found = None
        # x must be > n/4 and < 3n/4
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() - t0 > DEADLINE:
                break
                
            A = 4 * x - n
            B = n * x
            
            if A <= 0:
                continue
                
            # We need (Ay - B)(Az - B) = B^2
            # Let u = Ay - B. Then u divides B^2.
            # y = (u + B) / A must be integer.
            # z = (B^2/u + B) / A must be integer.
            # We assume y < z, which implies u < B^2/u, so u < B.
            
            # Factorize B to get divisors of B^2
            factors_B = get_prime_factors(B)
            
            # Construct factors for B^2
            factors_B2 = {}
            for p, e in factors_B.items():
                factors_B2[p] = 2 * e
            
            divs = generate_divisors(factors_B2)
            
            # Sort divisors to find small u first? Not strictly necessary, 
            # but smaller u leads to smaller y, which is often preferred or just arbitrary.
            # We just need ANY valid distinct triplet.
            
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

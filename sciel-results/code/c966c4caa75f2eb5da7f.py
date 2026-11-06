import time
import math

def _factorize(num):
    """Return prime factorization as dict {prime: exponent}"""
    factors = {}
    d = 2
    while d * d <= num:
        while num % d == 0:
            factors[d] = factors.get(d, 0) + 1
            num //= d
        d += 1
    if num > 1:
        factors[num] = factors.get(num, 0) + 1
    return factors

def _divisors_from_factorization(factorization):
    """Generate all divisors from prime factorization dict"""
    if not factorization:
        yield 1
        return
    primes = list(factorization.keys())
    exponents = [factorization[p] for p in primes]
    result = [1]
    for p, e in zip(primes, exponents):
        new_result = []
        for val in result:
            for i in range(e + 1):
                new_result.append(val * (p ** i))
        result = new_result
    for d in result:
        yield d

def construct():
    """
    Construct distinct three-term unit-fraction representations of 4/n for n=3..256.
    Returns 254 rows; each is None or [x, y, z] with x<y<z, all distinct, <= 10^12.
    """
    deadline = time.time() + 85
    MAX_DEN = 10**12
    answers = []
    
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        found = None
        start_x = n // 4 + 1
        end_x = (3 * n) // 4
        
        for x in range(start_x, end_x + 1):
            if time.time() > deadline:
                break
            
            A = 4 * x - n
            B = n * x
            
            if A <= 0:
                continue
            
            # Need (A*y - B)(A*z - B) = B^2
            # u = A*y - B, v = A*z - B, u*v = B^2
            # u < B (since y < z implies u < v and u*v = B^2)
            # y = (u + B) / A must be integer, z = (v + B) / A must be integer
            # y > x, z > y, all distinct, all <= MAX_DEN
            
            # Factor B^2 = (n*x)^2
            # Get factorization of B, then double exponents for B^2
            B_factors = _factorize(B)
            B2_factors = {p: 2*e for p, e in B_factors.items()}
            
            # Enumerate divisors u of B^2 with u <= B
            # For each, check conditions
            found_for_x = False
            for u in _divisors_from_factorization(B2_factors):
                if u > B:
                    continue
                v = B * B // u
                if v < u:
                    continue  # we want u <= v, i.e., u <= B
                
                # Check y = (u + B) / A is integer
                if (u + B) % A != 0:
                    continue
                y = (u + B) // A
                
                # Check z = (v + B) / A is integer
                if (v + B) % A != 0:
                    continue
                z = (v + B) // A
                
                # Check constraints
                if y <= x or z <= y:
                    continue
                if y > MAX_DEN or z > MAX_DEN:
                    continue
                if x == y or y == z or x == z:
                    continue
                
                found = [x, y, z]
                found_for_x = True
                break
            
            if found_for_x:
                break
        
        answers.append(found)
    
    return answers

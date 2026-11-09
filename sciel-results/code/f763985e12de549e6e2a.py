import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 60s
    
    N = 210
    MAX_VAL = 10210
    MIN_VAL = 211  # strictly larger than 210
    
    best = None
    best_raw = float('inf')
    
    def is_perfect_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n
    
    def square_free_kernel(n):
        """Return the square-free part of n (product of primes with odd exponent)."""
        result = 1
        temp = n
        d = 2
        while d * d <= temp:
            count = 0
            while temp % d == 0:
                count += 1
                temp //= d
            if count % 2 == 1:
                result *= d
            d += 1
        if temp > 1:
            result *= temp
        return result
    
    # Strategy 1: Search for 2-element solutions
    # a * b * 210 must be a perfect square
    # So a * b = 210 * k^2 for some k
    # We want to minimize max(a, b) with a, b > 210, a != b, a, b <= 10210
    
    # For a given k, 210*k^2 must factor as a*b where both > 210 and <= 10210
    # To minimize max(a,b), we want a and b as close as possible to sqrt(210*k^2) = k*sqrt(210)
    
    found_2elem = []
    for k in range(1, 200):  # k up to ~200 should cover values up to 10210
        if time.time() > deadline:
            break
        target = N * k * k  # a * b must equal this
        # Find factor pairs (a, b) of target where a*b = target, a <= b
        # a must be >= MIN_VAL, b must be <= MAX_VAL, a != b
        # a ranges from ceil(target/MAX_VAL) to sqrt(target)
        a_min = max(MIN_VAL, (target + MAX_VAL - 1) // MAX_VAL)  # ceil(target/MAX_VAL)
        a_max = math.isqrt(target)
        
        for a in range(a_min, a_max + 1):
            if time.time() > deadline:
                break
            if target % a != 0:
                continue
            b = target // a
            if b < a:
                break
            if b > MAX_VAL:
                continue
            if a == b:
                continue
            if a < MIN_VAL or b < MIN_VAL:
                continue
            # Verify: a * b * 210 is a perfect square
            prod = a * b * N
            if is_perfect_square(prod):
                raw = b - N  # since b >= a, max is b
                if raw < best_raw:
                    best_raw = raw
                    best = [a, b]
    
    # Strategy 2: Search for 3-element solutions if no good 2-element found
    # a * b * c * 210 must be a perfect square
    # This is harder; let's try small a, b and solve for c
    # c = (210 * k^2) / (a * b) for some k
    # Actually: a*b*c*210 = square => c = square / (a*b*210)
    # Let's try: for small a, b > 210, check if there exists c > 210, c != a,b, c <= 10210
    # such that a*b*c*210 is a perfect square.
    # a*b*210 * c = m^2 => c = m^2 / (a*b*210)
    # So (a*b*210) must divide m^2, meaning the square-free part of a*b*210 must be 1
    # OR c must supply the missing factors.
    
    # More systematically: let sf = square_free_kernel(a * b * N)
    # Then c must have square_free_kernel(c) == sf (so that sf * c has even exponents)
    # Wait, that's not quite right. Let me think again.
    # a*b*N * c = square
    # Let a*b*N = s^2 * sf where sf is square-free
    # Then c must be sf * t^2 for some t, so that s^2 * sf * sf * t^2 = (s*ft)^2
    # So c = sf * t^2 for some positive integer t
    # We need c > 210, c != a, c != b, c <= 10210
    
    if best is None or best_raw > 50:  # only try 3-element if 2-element is poor
        for a in range(MIN_VAL, min(MIN_VAL + 200, MAX_VAL + 1)):
            if time.time() > deadline:
                break
            for b in range(a + 1, min(a + 200, MAX_VAL + 1)):
                if time.time() > deadline:
                    break
                sf = square_free_kernel(a * b * N)
                # c = sf * t^2
                # c > 210, c <= 10210, c != a, c != b
                # t^2 > 210/sf => t > sqrt(210/sf)
                # t^2 <= 10210/sf => t <= sqrt(10210/sf)
                t_min = max(1, math.isqrt(210 // sf) + (1 if (210 // sf) > 0 and math.isqrt(210//sf)**2 <= 210//sf else 0))
                # Actually: need sf * t^2 > 210 => t^2 > 210/sf => t > sqrt(210/sf)
                t_min = math.isqrt(210 // sf)
                if sf * t_min * t_min <= 210:
                    t_min += 1
                t_max = math.isqrt(MAX_VAL // sf)
                
                for t in range(t_min, t_max + 1):
                    if time.time() > deadline:
                        break
                    c = sf * t * t
                    if c <= 210 or c > MAX_VAL:
                        continue
                    if c == a or c == b:
                        continue
                    # Verify
                    prod = a * b * c * N
                    if is_perfect_square(prod):
                        raw = max(a, b, c) - N
                        if raw < best_raw:
                            best_raw = raw
                            best = [a, b, c]
    
    # Fallback: trivial solution
    if best is None:
        # 4*210 = 840, so [840] gives 840*210 = 210^2 * 4 = (2*210)^2 = 420^2
        best = [4 * N]
        best_raw = 3 * N
    
    # Ensure types are correct
    result = [int(x) for x in best]
    return result

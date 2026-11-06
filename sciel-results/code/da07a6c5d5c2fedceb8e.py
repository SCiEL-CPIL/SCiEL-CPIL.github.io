import time
import math

def construct():
    answers = [None] * 254
    LIMIT = 10**12
    start_time = time.time()
    DEADLINE = 55.0 # Leave 5s margin
    
    def get_divisors_of_square(n):
        """
        Returns list of divisors of n^2.
        n is small enough to factor quickly.
        """
        if n == 1:
            return [1]
        
        # Factorize n
        factors = {}
        temp = n
        d = 2
        while d * d <= temp:
            while temp % d == 0:
                factors[d] = factors.get(d, 0) + 1
                temp //= d
            d += 1
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
            
        # Generate divisors of n^2
        # If n = prod p_i^e_i, n^2 = prod p_i^(2e_i)
        # Divisors are prod p_i^(k_i) where 0 <= k_i <= 2e_i
        divs = [1]
        for p, e in factors.items():
            max_exp = 2 * e
            current_divs = []
            for d in divs:
                p_power = 1
                for _ in range(max_exp + 1):
                    current_divs.append(d * p_power)
                    p_power *= p
            divs = current_divs
        return divs

    for idx, n in enumerate(range(3, 257)):
        if time.time() - start_time > DEADLINE:
            break
            
        found = None
        # x range: n/4 < x < 3n/4
        start_x = n // 4 + 1
        end_x = (3 * n) // 4 - 1 # x < 3n/4 strictly
        
        for x in range(start_x, end_x + 1):
            A = 4 * x - n
            B = n * x
            if A <= 0:
                continue
                
            # We need U such that U * V = B^2, U < V
            # U = A*y - B
            # y = (U+B)/A
            # y > x => U+B > Ax => U > Ax - B
            # z = (V+B)/A
            # z > y => V > U
            # z <= LIMIT => (V+B)/A <= LIMIT => V <= A*LIMIT - B
            
            # Get divisors of B^2
            divs = get_divisors_of_square(B)
            
            # Filter divisors U
            # U must be < B (since U < V and UV=B^2)
            # Also U > Ax - B (for y > x)
            lower_bound_U = A * x - B
            upper_bound_U = B - 1 # U < B
            
            for U in divs:
                if U > upper_bound_U:
                    continue
                if U <= lower_bound_U:
                    continue
                    
                # Check if y is integer
                if (U + B) % A != 0:
                    continue
                y = (U + B) // A
                if y <= x:
                    continue
                
                V = (B * B) // U
                # Check if z is integer
                if (V + B) % A != 0:
                    continue
                z = (V + B) // A
                if z <= y:
                    continue
                if z > LIMIT:
                    continue
                
                # Found a valid solution
                found = [x, y, z]
                break
            
            if found:
                break
        
        answers[idx] = found

    return answers

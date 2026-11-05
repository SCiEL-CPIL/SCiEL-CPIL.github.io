import time
from fractions import Fraction
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    target = Fraction(5, 31)
    max_terms_allowed = 256
    max_denom_limit = 10000000
    
    best_max = float('inf')
    best_solution = None

    def update_best(denoms, current_max):
        nonlocal best_max, best_solution
        if current_max < best_max:
            best_max = current_max
            best_solution = list(denoms)

    def solve_two_terms(rem, min_den, min_den_strict=True):
        # Solve 1/x + 1/y = rem, with x < y and x >= min_den
        # rem = N/D
        N = rem.numerator
        D = rem.denominator
        if N == 0:
            return []
        
        # (Nx - D)(Ny - D) = D^2
        # X = Nx - D, Y = Ny - D
        # X * Y = D^2
        # x = (X + D) / N
        # y = (Y + D) / N
        # x < y => X < Y => X < D
        # x >= min_den => (X + D) / N >= min_den => X >= N * min_den - D
        
        D_sq = D * D
        min_X = N * min_den - D
        if min_X < 1:
            min_X = 1
            
        # Iterate divisors X of D_sq
        # X must be <= D
        # X must be >= min_X
        # X must satisfy (X + D) % N == 0
        
        sols = []
        # Find divisors of D_sq
        # D can be large, but D_sq is square.
        # We can iterate i up to isqrt(D_sq) = D
        # But D can be up to 10^7? No, D comes from 5/31 - sum of unit fractions.
        # Denominators in intermediate steps can grow, but for k=3,4 they are manageable.
        
        # Optimization: Instead of iterating all divisors, we can iterate potential x values?
        # No, divisor enumeration is better if D is small.
        # If D is large, this might be slow.
        # However, for k=3, D is small.
        # For k=4, D is from 5/31 - 1/a - 1/b.
        
        # Let's use a fast divisor check.
        # We only care about X in [min_X, D].
        
        # If D is very large, we might time out.
        # But for 5/31, D is small.
        
        limit = isqrt(D_sq) # which is D
        # We need X <= D.
        # So we iterate divisors up to D.
        
        # Actually, we can just iterate i from 1 to D? No, D can be big.
        # But we only need divisors.
        # Standard divisor enumeration is O(sqrt(N)).
        
        divs = []
        for i in range(1, isqrt(D_sq) + 1):
            if D_sq % i == 0:
                divs.append(i)
                if i != D_sq // i:
                    divs.append(D_sq // i)
                
        for X in divs:
            if X > D:
                continue
            if X < min_X:
                continue
            if (X + D) % N != 0:
                continue
            x = (X + D) // N
            y = (D_sq // X + D) // N
            
            # Check distinctness and order
            if x >= y:
                continue
            if x < min_den:
                continue
            if y <= x: # redundant
                continue
            # Check if x, y are valid denominators
            if x <= max_denom_limit and y <= max_denom_limit:
                sols.append((x, y))
                
        return sols

    # Search for k=3
    # 1/a + 1/b + 1/c = 5/31
    # a < b < c
    # 3/a > 5/31 => a < 18.6 => a <= 18
    # 1/a < 5/31 => a > 6.2 => a >= 7
    for a in range(7, 19):
        rem = target - Fraction(1, a)
        if rem <= 0:
            continue
        sols = solve_two_terms(rem, a + 1)
        for b, c in sols:
            update_best([a, b, c], c)
            
    # Search for k=4
    # 1/a + 1/b + 1/c + 1/d = 5/31
    # a < b < c < d
    # 4/a > 5/31 => a < 24.8 => a <= 24
    # 1/a < 5/31 => a > 6.2 => a >= 7
    for a in range(7, 25):
        if time.time() > deadline:
            break
        rem_a = target - Fraction(1, a)
        if rem_a <= 0:
            continue
        # 3/b > rem_a => b < 3 / rem_a
        # 1/b < rem_a => b > 1 / rem_a
        b_min = a + 1
        b_max = int(3 / rem_a)
        
        for b in range(b_min, b_max + 1):
            if time.time() > deadline:
                break
            rem_ab = rem_a - Fraction(1, b)
            if rem_ab <= 0:
                continue
            sols = solve_two_terms(rem_ab, b + 1)
            for c, d in sols:
                update_best([a, b, c, d], d)

    # If no solution found, use fallback
    if best_solution is None:
        best_solution = [8, 28, 1736]
        best_max = 1736
    else:
        # Validate
        if 0 < len(best_solution) <= 256:
            if all(1 <= d <= 10000000 for d in best_solution):
                if len(set(best_solution)) == len(best_solution):
                    s = sum(Fraction(1, d) for d in best_solution)
                    if s != target:
                        best_solution = [8, 28, 1736]
                        best_max = 1736
                else:
                    best_solution = [8, 28, 1736]
                    best_max = 1736
            else:
                best_solution = [8, 28, 1736]
                best_max = 1736
        else:
            best_solution = [8, 28, 1736]
            best_max = 1736
            
    return best_solution

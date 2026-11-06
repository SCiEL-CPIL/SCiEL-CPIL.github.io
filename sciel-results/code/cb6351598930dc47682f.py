"""Solve 1/x + 1/y + 1/z = 4/n for n=3..256 with distinct denominators <= 10^12."""
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 70.0  # safety margin before 90s kill
    MAX_DEN = 10**12
    
    def get_divisors(n):
        """Return list of divisors of n."""
        divs = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
            i += 1
        return divs

    def find_solution(n):
        """Find distinct x<y<z with 1/x+1/y+1/z=4/n, denominators <= MAX_DEN."""
        x_min = n // 4 + 1
        x_upper = 3 * n + 10  # generous upper bound
        
        for x in range(x_min, x_upper + 1):
            if time.time() > deadline:
                return None
            
            a = 4 * x - n
            if a <= 0:
                continue
            b = n * x
            
            # We need d = a*y - b to be a divisor of b^2.
            # Conditions on d:
            # 1. d > a*x - b (from y > x)
            # 2. d < b (from z > y, which implies y < 2b/a => d < b)
            # 3. (d + b) % a == 0 (so y is integer)
            # 4. z = b*y/d is integer and <= MAX_DEN
            
            d_min = a * x - b
            d_max = b - 1
            
            if d_min >= d_max:
                continue
            
            # Get divisors of b^2
            # b can be up to 256 * (3*256+10) ~ 200k. b^2 ~ 4e10.
            # Iterating up to sqrt(b^2) = b is 200k iterations.
            # This might be too slow if done for every x.
            # However, b is small.
            # Let's optimize divisor generation.
            
            # Instead of generating all divisors of b^2, we can iterate d in range?
            # No, range is large.
            
            # Let's stick to iterating y but with a tighter bound or different strategy?
            # The previous method iterated y. The number of y is O(n).
            # Number of divisors of b^2 is small.
            # Let's generate divisors of b^2.
            
            # To speed up, we can factor b.
            # But get_divisors(b*b) iterates up to b.
            # If b is ~1000, this is fast.
            # If b is ~100000, this is 100000 iterations.
            # For n=256, x~200, b~50000. sqrt(b^2)=50000.
            # 50000 iterations per x. x goes up to 760.
            # Total ~ 38 million iterations. Might be borderline in Python.
            
            # Let's try the divisor method first.
            b_sq = b * b
            divs = get_divisors(b_sq)
            
            for d in divs:
                if d <= d_min or d >= d_max:
                    continue
                if (d + b) % a != 0:
                    continue
                y = (d + b) // a
                # Check y > x is guaranteed by d > d_min
                # Check y < 2b/a is guaranteed by d < b
                # Check z
                if (b * y) % d != 0:
                    continue
                z = (b * y) // d
                if z <= y:
                    continue
                if z > MAX_DEN:
                    continue
                # Check distinctness
                if x == y or x == z or y == z:
                    continue
                return (x, y, z)
            
        return None
    
    answers = []
    for n in range(3, 257):
        if time.time() > deadline:
            answers.append(None)
            continue
        
        sol = find_solution(n)
        answers.append(sol)
    
    return answers

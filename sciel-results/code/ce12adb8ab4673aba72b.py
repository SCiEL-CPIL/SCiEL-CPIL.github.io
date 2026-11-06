import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    MAX_DEN = 10**12
    answers = [None] * 254
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            continue
            
        found = None
        # x must satisfy 1/x < 4/n => x > n/4
        # and x <= y <= z => 3/x >= 4/n => x <= 3n/4
        x_min = n // 4 + 1
        x_max = 3 * n // 4
        
        for x in range(x_min, x_max + 1):
            if found:
                break
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            
            # u = A*y - B
            # v = A*z - B
            # u * v = B^2
            # u <= v => u <= B
            # y > x => A*y - B > A*x - B => u > A*x - B
            # y < z => u < v => u < B (since u,v positive and uv=B^2)
            
            L = A * x - B + 1
            if L < 1:
                L = 1
            R = B - 1
            
            if L > R:
                continue
                
            # u must be a divisor of B^2
            # u = k*A + r, where r = (-B) % A
            r = (-B) % A
            
            # Find k_min, k_max
            # L <= k*A + r <= R
            # (L - r) / A <= k <= (R - r) / A
            
            # Ceiling of (L-r)/A
            if (L - r) % A == 0:
                k_min = (L - r) // A
            else:
                k_min = (L - r) // A + 1
                
            k_max = (R - r) // A
            
            # Iterate k
            # Optimization: if range is large, maybe iterate divisors?
            # But B is at most 256 * 192 ~ 50000. B^2 ~ 2.5e9.
            # Number of divisors is small.
            # Number of k is (R-L)/A ~ B/A ~ nx/(4x-n).
            # For x ~ n/4, A ~ 1, k ~ B ~ 50000.
            # 50000 iterations per x, x up to 192.
            # Total ~ 10 million. Might be slow in Python if not careful.
            
            # Let's check if we can iterate divisors of B^2 instead.
            # Divisors of B^2.
            # B = n*x.
            # Factorize B?
            # Or just iterate u in the arithmetic progression and check divisibility.
            # Divisibility check is fast.
            
            for k in range(k_min, k_max + 1):
                u = k * A + r
                if u < L or u > R:
                    continue
                # Check if u divides B^2
                if (B * B) % u != 0:
                    continue
                v = (B * B) // u
                # u < v is guaranteed by u < B and v = B^2/u > B
                # Actually if u < B, v > B. So u < v.
                # y = (u+B)//A
                # z = (v+B)//A
                # We know u = A*y - B => y = (u+B)/A. Since u ≡ -B (mod A), y is integer.
                y = (u + B) // A
                z = (v + B) // A
                
                # Check bounds and distinctness
                if y > x and z > y and z <= MAX_DEN:
                    found = (x, y, z)
                    break
        answers[idx] = found
        
    return answers

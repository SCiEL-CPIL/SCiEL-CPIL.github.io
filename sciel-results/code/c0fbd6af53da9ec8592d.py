import time
import random
from math import gcd
from bisect import bisect_left
from itertools import product
from operator import mul

def construct():
        start_time = time.time()
        deadline = 55.0
        n = 30
        
        # Calculate LCM
        L = 1
        for i in range(1, n + 1):
            L = L * i // gcd(L, i)
        
        weights = [L // i for i in range(1, n + 1)]
        
        half = 15
        w1 = weights[:half]
        w2 = weights[half:]
        
        # Precompute powers of 3 for packing
        pow3 = [1] * half
        for i in range(1, half):
            pow3[i] = pow3[i-1] * 3
            
        def pack(coeffs):
            p = 0
            for i, c in enumerate(coeffs):
                p += (c + 1) * pow3[i]
            return p
            
        def unpack(p, m):
            coeffs = [0] * m
            for i in range(m):
                coeffs[i] = (p // pow3[i]) % 3 - 1
            return coeffs
        
        # Generate sums for first half
        sums1 = []
        for c in product([-1, 0, 1], repeat=half):
            s = sum(map(mul, c, w1))
            p = pack(c)
            sums1.append((s, p))
            
        # Sort
        sums1.sort(key=lambda x: x[0])
        sums1_vals = [x[0] for x in sums1]
        
        # Generate sums for second half and search
        best_abs = float('inf')
        best_p1 = 0
        best_p2 = 0
        
        for c2 in product([-1, 0, 1], repeat=half):
            s2 = sum(map(mul, c2, w2))
            target = -s2
            pos = bisect_left(sums1_vals, target)
            
            # Check neighbors
            # We want min |s1 + s2| != 0
            candidates = []
            if pos < len(sums1):
                candidates.append(pos)
            if pos > 0:
                candidates.append(pos - 1)
            # Check a few more in case of duplicates or zeros
            if pos + 1 < len(sums1):
                candidates.append(pos + 1)
            if pos - 2 >= 0:
                candidates.append(pos - 2)
                
            for idx in candidates:
                s1, p1 = sums1[idx]
                total = s1 + s2
                if total == 0:
                    continue
                abs_total = abs(total)
                if abs_total < best_abs:
                    best_abs = abs_total
                    best_p1 = p1
                    best_p2 = pack(c2)
                    
                    # If we found 1, it's the best possible integer numerator
                    if best_abs == 1:
                        break
            
            if best_abs == 1:
                break
                
            if time.time() > deadline - 5.0:
                break
        
        # Reconstruct coeffs
        c1 = unpack(best_p1, half)
        c2 = unpack(best_p2, half)
        
        # Combine
        # c1 corresponds to weights 1..15
        # c2 corresponds to weights 16..30
        # The problem asks for coefficient[i-1]/i
        # So index 0 is 1/1, index 14 is 1/15
        # Index 15 is 1/16, index 29 is 1/30
        
        final_coeffs = c1 + c2
        
        return [int(c) for c in final_coeffs]

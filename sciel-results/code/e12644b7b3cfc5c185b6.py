import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    MAX_B = 50000
    spf = list(range(MAX_B + 1))
    for i in range(2, int(MAX_B**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, MAX_B + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def get_divisors(num):
        factors = {}
        n = num
        while n > 1:
            p = spf[n]
            cnt = 0
            while n % p == 0:
                n //= p
                cnt += 1
            factors[p] = cnt
        
        divs = [1]
        for p, e in factors.items():
            current_divs = list(divs)
            mult = 1
            for _ in range(e):
                mult *= p
                for d in current_divs:
                    divs.append(d * mult)
        return divs

    answers = [None] * 254
    
    for n in range(3, 257):
        idx = n - 3
        if time.time() > deadline:
            break
            
        found = None
        x_min = n // 4 + 1
        x_max = (3 * n) // 4
        
        for x in range(x_min, x_max + 1):
            if time.time() > deadline:
                break
                
            A = 4 * x - n
            if A <= 0:
                continue
            B = n * x
            B2 = B * B
            
            divs = get_divisors(B2)
            
            target_mod = (-B) % A
            min_u = 4 * x * x - 2 * n * x
            
            for u in divs:
                if u > B:
                    break
                if u < min_u:
                    continue
                if u % A != target_mod:
                    continue
                
                v = B2 // u
                if v % A != target_mod:
                    continue
                
                y = (u + B) // A
                z = (v + B) // A
                
                if y > x and z > y:
                    if z <= 1000000000000:
                        found = [x, y, z]
                        break
            
            if found:
                break
                
        answers[idx] = found
        
    return answers

import math
import time
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 115
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = x * x % n
                if x == n - 1:
                    break
            else:
                return False
        return True
    
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(1, n - 1)
            c = random.randint(1, n - 1)
            m = random.randint(1, n - 1)
            g, r, q = 1, 1, 1
            while g == 1:
                x = y
                for _ in range(r):
                    y = (y * y + c) % n
                k = 0
                while k < r and g == 1:
                    ys = y
                    for _ in range(min(m, r - k)):
                        y = (y * y + c) % n
                        q = q * abs(x - y) % n
                    g = math.gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
            if g != n:
                return g
    
    def largest_prime_factor(n):
        if n <= 1:
            return 1
        max_pf = 1
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            while n % p == 0:
                max_pf = max(max_pf, p)
                n //= p
        if n == 1:
            return max_pf
        if is_prime(n):
            return max(max_pf, n)
        stack = [n]
        while stack:
            x = stack.pop()
            if x == 1:
                continue
            if is_prime(x):
                max_pf = max(max_pf, x)
            else:
                d = pollard_brent(x)
                if d == x:
                    stack.append(x)
                else:
                    stack.append(d)
                    stack.append(x // d)
        return max_pf
    
    def check_run(p, max_k=200):
        k = 0
        for i in range(1, max_k + 1):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            n = p * p + i
            lpf = largest_prime_factor(n)
            if lpf <= p:
                k += 1
            else:
                break
        return k
    
    best_k = 0
    best_p = 2
    
    def is_prime_small(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(math.isqrt(n)) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    for p in range(3, 500, 2):
        if time.time() - start_time > TIME_LIMIT - 50:
            break
        if is_prime_small(p):
            k = check_run(p)
            if k > best_k:
                best_k = k
                best_p = p
    
    for p in range(500, 10000, 2):
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        if is_prime(p):
            k = check_run(p)
            if k > best_k:
                best_k = k
                best_p = p
    
    random.seed(42)
    for _ in range(300):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        p = random.randint(10000, 10**9)
        if p % 2 == 0:
            p += 1
        while not is_prime(p) and p < 10**9:
            p += 2
        if p >= 10**9:
            continue
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p
    
    if best_k < 1:
        best_p = 2
        best_k = 1
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END

import time
import math

def sieve_primes(limit):
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            start = i * i
            step = i
            is_prime[start::step] = bytearray(len(is_prime[start::step]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    primes = sieve_primes(10**6)
    
    def phi(n):
        if n == 1:
            return 1
        result = n
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
                if temp == 1:
                    break
        if temp > 1:
            result -= result // temp
        return result

    def check_run(s, k):
        seen = set()
        for i in range(k):
            n = s + i
            v = phi(n)
            if v in seen:
                return False
            seen.add(v)
        return True

    def find_best_run():
        best_s = 5
        best_k = 3
        for k_try in range(200, 2, -1):
            if time.time() > deadline:
                break
            # Try several starting positions for this k
            candidates = []
            # Positions near 10^12
            for offset in range(0, 5000, 50):
                s_try = 10**12 - k_try + 1 - offset
                if s_try < 1:
                    break
                candidates.append(s_try)
            # Some mid-range positions
            for base in [10**10, 10**11, 5*10**11]:
                for offset in range(0, 2000, 100):
                    s_try = base - offset
                    if s_try < 1:
                        continue
                    candidates.append(s_try)
            
            found = False
            for s_try in candidates:
                if time.time() > deadline:
                    break
                if check_run(s_try, k_try):
                    best_s = s_try
                    best_k = k_try
                    found = True
                    break
            if found:
                break
        return best_s, best_k

    s, k = find_best_run()
    return {'start': s, 'length': k}

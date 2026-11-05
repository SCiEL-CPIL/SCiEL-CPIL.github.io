import time

def construct():
    deadline = time.time() + 55.0
    
    def phi(n):
        if n == 1:
            return 1
        result = n
        m = n
        if m % 2 == 0:
            result -= result // 2
            while m % 2 == 0:
                m //= 2
        p = 3
        while p * p <= m:
            if m % p == 0:
                result -= result // p
                while m % p == 0:
                    m //= p
            p += 2
        if m > 1:
            result -= result // m
        return result

    best = 1000001
    for n in range(1, 1000001):
        if time.time() > deadline:
            break
        if phi(n) % 120 == 0:
            best = n
            break
    return best

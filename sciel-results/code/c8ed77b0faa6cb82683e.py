import random

def construct():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    def smooth_part(x):
        s = 1
        for p in primes:
            while x % p == 0:
                s *= p
                x //= p
        return s
    
    best_n = 1000
    best_count = float('inf')
    
    # Evaluate seed and a large random sample
    candidates = [1000, 1000000]
    # Add random samples
    for _ in range(200000):
        candidates.append(random.randint(1000, 1000000))
    
    for n in candidates:
        smooths = []
        for i in range(1, 25):
            smooths.append(smooth_part(n + i))
        count = len(set(smooths))
        if count < best_count:
            best_count = count
            best_n = n
    
    return best_n

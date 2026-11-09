import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 80
    
    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    primes = sieve(200)
    
    def generate_smooth_numbers(limit, primes):
        smooths = {1}
        for p in primes:
            new_smooths = set()
            for s in list(smooths):
                val = s
                while val <= limit:
                    new_smooths.add(val)
                    if val > limit // p:
                        break
                    val *= p
            smooths.update(new_smooths)
        return smooths
    
    best_result = [1, N - 1]
    
    for y in primes:
        if time.time() > deadline:
            break
        
        primes_le_y = [p for p in primes if p <= y]
        smooth_set = generate_smooth_numbers(N, primes_le_y)
        
        found_pair = None
        for a in smooth_set:
            b = N - a
            if b > 0 and b in smooth_set:
                found_pair = [a, b]
                break
        
        if found_pair:
            return found_pair
    
    return best_result

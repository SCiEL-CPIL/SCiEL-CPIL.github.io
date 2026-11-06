def construct():
    # Precompute primes up to 1000
    limit = 1000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, limit + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False

    def phi(n):
        result = n
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
        if temp > 1:
            result -= result // temp
        return result

    # n must be divisible by 7 and 11
    # So n is a multiple of 77
    for k in range(1, 1000000 // 77 + 1):
        n = 77 * k
        if phi(n) % 120 == 0:
            return n
    return -1

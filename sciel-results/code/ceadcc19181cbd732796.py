    prime_cache = {}
    def is_prime(n):
        if n in prime_cache:
            return prime_cache[n]
        if n < 2:
            prime_cache[n] = False
            return False
        for i in range(2, int(math.isqrt(n)) + 1):
            if n % i == 0:
                prime_cache[n] = False
                return False
        prime_cache[n] = True
        return True

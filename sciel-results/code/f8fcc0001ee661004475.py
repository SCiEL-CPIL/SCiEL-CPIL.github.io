<<<<<<< SEARCH
    def sieve_primes(limit):
        """Sieve primes up to limit using bytearray, with caching."""
        if limit in _prime_cache:
            return _prime_cache[limit]
        if limit < 2:
            return []
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                start = i * i
                step = i
                is_prime[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))
        result = [i for i in range(2, limit + 1) if is_prime[i]]
        _prime_cache[limit] = result
        return result
=======
    def sieve_primes(limit):
        """Sieve primes up to limit using bytearray, with caching."""
        if limit in _prime_cache:
            return _prime_cache[limit]
        if limit < 2:
            return []
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                start = i * i
                step = i
                is_prime[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))
        result = [i for i in range(2, limit + 1) if is_prime[i]]
        _prime_cache[limit] = result
        return result
>>>>>>> REPLACE


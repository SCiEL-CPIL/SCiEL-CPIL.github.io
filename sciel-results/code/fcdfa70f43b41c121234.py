    def construct():
        max_gap = max(hi - lo for lo, hi in P['gaps'])
        # Sieve for small primes up to max_gap
        is_prime = [True] * max_gap
        is_prime[0] = False
        is_prime[1] = False
        for i in range(2, int(max_gap**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, max_gap, i):
                    is_prime[j] = False
        
        small_primes = [i for i, prime in enumerate(is_prime) if prime]
        
        result = []
        for lo, hi in P['gaps']:
            gap_size = hi - lo
            check_primes = [p for p in small_primes if p < gap_size]
            witness = None
            for m in range(lo + 1, hi):
                has_small_factor = False
                for p in check_primes:
                    if m % p == 0:
                        has_small_factor = True
                        break
                if not has_small_factor:
                    witness = m
                    break
            result.append(witness)
        return result
    
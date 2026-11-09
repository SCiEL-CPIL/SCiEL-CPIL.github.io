def construct():
    # Find all prime powers in [1, 22]
    # Plus 1
    
    def is_prime_power(n):
        if n == 1:
            return True
        # Check if n is a power of a prime
        # n = p^k
        # p must be prime
        # k >= 1
        # Simple check for small n
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23]:
            if p > n:
                break
            # Check if n is power of p
            temp = n
            while temp % p == 0:
                temp //= p
            if temp == 1:
                # Check if p is prime
                if p > 1:
                    is_p_prime = True
                    for d in range(2, int(p**0.5)+1):
                        if p % d == 0:
                            is_p_prime = False
                            break
                    if is_p_prime:
                        return True
        return False

    # Actually, simpler: just hardcode or generate
    # Primes: 2, 3, 5, 7, 11, 13, 17, 19
    # Powers:
    # 2: 2, 4, 8, 16
    # 3: 3, 9
    # 5: 5
    # 7: 7
    # 11: 11
    # 13: 13
    # 17: 17
    # 19: 19
    
    candidate_set = [1, 2, 4, 8, 16, 3, 9, 5, 7, 11, 13, 17, 19]
    
    # Verify validity just in case
    def is_valid(subset):
        n = len(subset)
        if n == 0:
            return True
        sums = [Fraction(0)] * (1 << n)
        for mask in range(1, 1 << n):
            lsb = mask & (-mask)
            bit_idx = lsb.bit_length() - 1
            prev = mask ^ lsb
            sums[mask] = sums[prev] + Fraction(1, subset[bit_idx])
        return len(set(sums)) == (1 << n)

    if is_valid(candidate_set):
        return candidate_set
    else:
        # Fallback to greedy if the hardcoded set fails (it shouldn't)
        result = []
        for candidate in range(1, 23):
            trial = result + [candidate]
            if len(trial) <= 15:
                if is_valid(trial):
                    result = trial
        return result

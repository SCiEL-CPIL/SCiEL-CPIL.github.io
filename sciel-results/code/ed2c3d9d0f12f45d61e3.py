    def has_factor_gt_p(n, p):
        if n <= p:
            return False
        
        # Trial division for small n
        if n < 10**6:
            d = 2
            while d * d <= n:
                if n % d == 0:
                    if d > p:
                        return True
                    n //= d
                    while n % d == 0:
                        n //= d
                d += 1 if d == 2 else 2
            # Remaining n is prime
            return n > p

        # For large n
        # Trial divide by small primes
        for i in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % i == 0:
                if i > p:
                    return True
                while n % i == 0:
                    n //= i
                if n == 1:
                    return False
                return has_factor_gt_p(n, p)
        
        if is_prime(n):
            return n > p
        
        factor = pollard_rho(n)
        if has_factor_gt_p(factor, p):
            return True
        return has_factor_gt_p(n // factor, p)

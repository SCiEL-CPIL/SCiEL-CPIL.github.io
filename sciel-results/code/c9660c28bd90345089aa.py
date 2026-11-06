        for d in small_primes:
            if d > p:
                # If d > p and d divides n, then we have a factor > p
                if n % d == 0:
                    return True
                else:
                    continue
            
            while n % d == 0:
                if d > p:
                    return True
                n //= d
            if n == 1:
                return False
            if is_prime(n):
                return n > p

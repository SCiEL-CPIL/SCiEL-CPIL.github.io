    def is_p_smooth(n, p):
        if n == 1:
            return True
        # Remove small prime factors
        for sp in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if n % sp == 0:
                while n % sp == 0:
                    n //= sp
        if n == 1:
            return True
        
        # Now n is either 1, prime, or composite with no small factors.
        # If n is prime, check n <= p
        if is_prime(n):
            return n <= p
        
        # n is composite. Find a factor.
        # If we find a prime factor > p, return False.
        # We can use a recursive check or stack.
        stack = [n]
        while stack:
            x = stack.pop()
            if x == 1:
                continue
            if is_prime(x):
                if x > p:
                    return False
            else:
                d = pollard_brent(x)
                stack.append(d)
                stack.append(x // d)
        return True

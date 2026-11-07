"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _validity_check(S_list):
    """Check if sorted list S_list satisfies: for all distinct a,b,c, a does not divide b*c."""
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            for k in range(j + 1, n):
                c = S_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def construct():
    deadline = time.time() + 80
    n_max = 256
    primes = [x for x in range(2, n_max + 1) if _is_prime(x)]
    
    best_set = primes[:]
    best_size = len(primes)

    # Precompute prime factors for all numbers up to 256
    def get_prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors.add(d)
                n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    # Generate candidate sets by removing first k primes
    # and adding composites in (128, 256] with factors only in removed primes
    for k in range(len(primes) + 1):
        if time.time() > deadline:
            break
        removed = set(primes[:k])
        # Primes remaining
        remaining_primes = [p for p in primes if p not in removed]
        
        # Composites to add: in (128, 256], all prime factors in 'removed'
        composites_to_add = []
        for x in range(129, n_max + 1):
            if _is_prime(x):
                continue
            pf = get_prime_factors(x)
            if pf.issubset(removed):
                composites_to_add.append(x)
        
        S_test = remaining_primes + composites_to_add
        S_test.sort()
        
        # Check validity
        if _validity_check(S_test):
            if len(S_test) > best_size:
                best_size = len(S_test)
                best_set = S_test[:]

    # Also try removing specific small primes not in prefix?
    # Just in case, try removing {2, 3, 5, 7, 11, 13, 17, 19, 23}
    # The prefix loop covers this.

    return [int(x) for x in sorted(best_set)]
# EVOLVE-BLOCK-END

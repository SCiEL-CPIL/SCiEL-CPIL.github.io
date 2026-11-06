<<<<<<< SEARCH
import math
import time
from math import gcd
import random

def _get_primes(limit):
    """Sieve of Eratosthenes."""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, p in enumerate(is_prime) if p]

# We need primes up to a reasonably large number to allow swaps.
# Max prime in product < 10^18. If we have 14 primes, the 15th can be up to 10^18 / P_14 ~ 10^18 / 10^16 = 100.
# If we have 10 primes, the 11th can be up to 10^18 / P_10 ~ 10^18 / 2000 = 5*10^14.
# Let's get primes up to 10^6 to be safe for swaps.
_PRIMES = _get_primes(1000000)

def _tau_perp_fast(n, factorization):
    """Compute tau_perp using factorization for speed.
    factorization: list of (prime, exponent)
    """
    # Generate all divisors from factorization
    divs = [1]
    for p, e in factorization:
        new_divs = []
        for d in divs:
            val = d
            for exp in range(e + 1):
                new_divs.append(val)
                val *= p
        divs = new_divs
    divs.sort()
    count = 0
    for i in range(len(divs) - 1):
        if gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count
=======
import math
import time
from math import gcd
import random
import bisect

def _get_primes(limit):
    """Sieve of Eratosthenes."""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, p in enumerate(is_prime) if p]

# We need primes up to a reasonably large number to allow swaps.
# Let's get primes up to 10^7 to have more options for swaps.
_PRIMES = _get_primes(10000000)

def _tau_perp_fast(n, factorization):
    """Compute tau_perp using factorization for speed.
    factorization: list of (prime, exponent)
    """
    # Generate all divisors from factorization
    divs = [1]
    for p, e in factorization:
        new_divs = []
        for d in divs:
            val = d
            for exp in range(e + 1):
                new_divs.append(val)
                val *= p
        divs = new_divs
    divs.sort()
    count = 0
    for i in range(len(divs) - 1):
        if gcd(divs[i], divs[i+1]) == 1:
            count += 1
    return count
>>>>>>> REPLACE


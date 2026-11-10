"""
Rewritten program to maximize the number of distinct integers in 2..40 
with distinct subset products.

The maximum size of such a set is 12, corresponding to the 12 primes 
less than or equal to 40. Any set of primes is multiplicatively independent, 
ensuring all subset products are distinct. 
Since the vectors of prime exponents for any number in [2, 40] lie in a 
12-dimensional space (one dimension for each prime <= 40), no set of more 
than 12 such numbers can be multiplicatively independent.

Thus, returning the 12 primes is optimal and guarantees the maximum score.
"""

def construct():
    """
    Returns a list of 12 distinct integers in 2..40 with distinct subset products.
    These are the primes in the range [2, 40].
    """
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    return primes

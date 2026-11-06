"""
The task is to find a nonempty subset of [1, 255] containing 255, 
with gcd(a, b) > 1 for all distinct pairs, maximizing cardinality.

255 = 3 * 5 * 17.
Any element in the set must share a factor with 255, so it must be divisible by 3, 5, or 17.
If the set contains the number 3, then every other number in the set must be divisible by 3 
(because gcd(3, x) > 1 implies 3 divides x). 
The set of all multiples of 3 in [1, 255] has size floor(255/3) = 85.
This set is a valid clique because all elements are divisible by 3.

If the set does not contain 3, it must rely on 5 or 17.
If it contains 5, all elements must be divisible by 5 (size 51).
If it contains 17, all elements must be divisible by 17 (size 15).
Mixing "pure" multiples of different primes is impossible for a clique because 
a pure multiple of 3 (e.g., 3) is coprime to a pure multiple of 5 (e.g., 5).
Thus, the largest valid set is the set of all multiples of the smallest prime factor, 3.
"""

def construct():
    return list(range(3, 256, 3))

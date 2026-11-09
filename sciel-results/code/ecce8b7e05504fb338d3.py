"""
Optimized program for the large pairwise noncoprime set task.

The task requires finding a nonempty subset of [1, 255] containing 255 such that
gcd(a, b) > 1 for all distinct pairs, maximizing the cardinality.

Analysis:
1. 255 = 3 * 5 * 17.
2. Any element x in the set must satisfy gcd(x, 255) > 1, so x must be divisible by 3, 5, or 17.
3. If the set contains any number x such that its prime factors are a subset of {3} (e.g., 3, 9, 27, 81),
   then every other number y in the set must share a factor with x. Since x is only divisible by 3,
   y must be divisible by 3.
4. The set of all multiples of 3 in [1, 255] is a valid set because all elements share the factor 3.
   Its size is floor(255 / 3) = 85.
5. If the set does not contain any number whose only prime factor (among 3, 5, 17) is 3,
   it is restricted to numbers divisible by 5 or 17.
   The largest such clique is bounded by the size of the multiples of the next smallest prime, 5,
   which is floor(255 / 5) = 51.
   Mixing sets (e.g., multiples of 3 and 5) fails because gcd(3, 5) = 1.
   Specifically, if we include a multiple of 5 that is not a multiple of 3 (e.g., 5),
   we cannot include 3. If we exclude 3, we must exclude other numbers in the "3-only" part of the multiples of 3
   to maintain pairwise non-coprimality with the "5-only" numbers, which drastically reduces the set size.
   It is a known mathematical result that the maximum size of such a set for n = p1*p2*... is floor(n/p1).
   
Therefore, the optimal strategy is to return all multiples of 3 in [1, 255].
"""

def construct():
    """
    Return a nonempty subset of [1, 255] containing 255, with gcd(a,b)>1 for all distinct pairs.
    Maximize cardinality.
    
    The maximum cardinality is 85, achieved by the set of all multiples of 3.
    """
    # The prime factors of 255 are 3, 5, 17.
    # The smallest prime factor is 3.
    # The set of multiples of 3 is the largest pairwise non-coprime subset containing 255.
    
    # Generate all multiples of 3 in [1, 255]
    # range(start, stop, step)
    # 3, 6, ..., 255
    result = list(range(3, 256, 3))
    
    # Ensure 255 is included (it is, as 255 is divisible by 3)
    # Ensure non-empty (it is, size 85)
    
    return result

import math
import time
import sys

def construct():
    """Return two lists A,B, each containing exactly 3 distinct integers in [1,128].
    All DISTINCT values in A+B must be pairwise coprime.
    Maximize the cardinality of A+B."""
    
    # The theoretical maximum for |A+B| with |A|=|B|=3 is 9.
    # We need 9 distinct sums that are pairwise coprime.
    # The simplest way to guarantee pairwise coprimality is to make all sums prime.
    # If we can find A and B such that all a+b are distinct primes, we achieve score 9.
    
    # Current best known solution from the "current program" context:
    # A = [4, 16, 40], B = [1, 3, 7]
    # Sums:
    # 4+1=5, 4+3=7, 4+7=11
    # 16+1=17, 16+3=19, 16+7=23
    # 40+1=41, 40+3=43, 40+7=47
    # All 9 are distinct primes. Score = 9.
    
    # Since the current program already achieves the theoretical maximum of 9,
    # and the inspirations also converge on solutions with score 9 (or fail to beat it),
    # the most robust strategy is to return this known optimal solution immediately.
    # Any search would only risk timeout or finding an equivalent solution.
    
    A = [4, 16, 40]
    B = [1, 3, 7]
    
    # Quick verification to ensure validity (should always pass)
    sums = [a + b for a in A for b in B]
    distinct_sums = list(set(sums))
    
    if len(distinct_sums) != 9:
        # Fallback if something is wrong (should never happen)
        return [[1, 7, 13], [4, 10, 16]]
        
    # Check pairwise coprimality
    for i in range(len(distinct_sums)):
        for j in range(i + 1, len(distinct_sums)):
            if math.gcd(distinct_sums[i], distinct_sums[j]) > 1:
                return [[1, 7, 13], [4, 10, 16]]
                
    return [A, B]

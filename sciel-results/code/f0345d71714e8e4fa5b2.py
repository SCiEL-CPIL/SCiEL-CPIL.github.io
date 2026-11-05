import time
import random

def construct():
    """
    Constructs a valid subset of [1, 256] maximizing cardinality.
    Uses a greedy approach adding elements from largest to smallest.
    """
    n = 256
    S = []
    
    # We iterate from n down to 1
    # S will contain elements larger than the current x
    for x in range(n, 0, -1):
        # Check if x can be added to S
        # Condition: For all b, c in S with b <= c, x must NOT divide b + c
        # Note: Since we add in descending order, all elements in S are > x.
        # So x will be the smallest element in the new set.
        # The condition "a < b <= c" with a=x implies b,c are from S.
        
        valid = True
        # Optimize: only check pairs b,c in S
        # If S is large, this loop takes time.
        # Max size of S is ~100. Pairs ~ 5000.
        # 256 * 5000 = 1.2M ops. Fast.
        
        # We can break early if found
        for i in range(len(S)):
            b = S[i]
            for j in range(i, len(S)):
                c = S[j]
                # b <= c is guaranteed by loop indices if S is sorted?
                # S is added in descending order, so S[0] > S[1] > ...
                # So b = S[i], c = S[j] with i <= j implies b >= c.
                # The condition is a < b <= c.
                # So we need to check pairs where the smaller is b and larger is c?
                # The set S is a set of numbers. The condition applies to any pair.
                # So we just need to check if x divides sum of any two distinct or same elements in S.
                # Since b and c are just elements from S, order doesn't matter for sum.
                # We just need to check if x | (S[i] + S[j]) for any i, j.
                
                if (b + c) % x == 0:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            S.append(x)
            
    return S

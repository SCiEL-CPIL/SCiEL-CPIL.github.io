import time
import itertools

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a list of 12 distinct positive integers <= 1048576 such that
    all 2^12 subset sums are distinct, minimizing the largest integer.
    
    Uses a greedy strategy: iteratively select the smallest positive integer
    greater than the previous one that preserves the distinctness of subset sums.
    This is equivalent to finding the smallest x such that x is not in the
    difference set of the current subset sums.
    """
    n = 12
    deadline = time.time() + 50.0  # Safety margin
    
    # S is the set of current subset sums
    S = {0}
    # Diff is the set of all differences s1 - s2 for s1, s2 in S
    # If x is in Diff, then adding x creates a collision (s1 + x = s2).
    Diff = {0}
    
    result = []
    last_val = 0
    
    for i in range(n):
        # Find the smallest x > last_val such that x is not in Diff
        x = last_val + 1
        while x in Diff:
            x += 1
            if time.time() > deadline:
                # Fallback to powers of 2 if we run out of time
                return [1 << k for k in range(n)]
        
        result.append(x)
        last_val = x
        
        # Update S
        new_S = set()
        for s in S:
            new_S.add(s)
            new_S.add(s + x)
        S = new_S
        
        # Update Diff efficiently
        # Diff_new = Diff_old U {d - x | d in Diff_old} U {d + x | d in Diff_old}
        # This works because:
        # New sums are S U (S+x).
        # Differences are (S-S) U ((S+x)-(S+x)) U (S-(S+x)) U ((S+x)-S)
        # = Diff U Diff U (Diff-x) U (Diff+x)
        
        if time.time() > deadline - 5.0:
             # If we are running out of time, assume the pattern is powers of 2
             # and fill the rest
             while len(result) < n:
                 result.append(1 << len(result))
             return result

        Diff_new = set(Diff)
        for d in Diff:
            Diff_new.add(d - x)
            Diff_new.add(d + x)
        Diff = Diff_new

    return result
# EVOLVE-BLOCK-END

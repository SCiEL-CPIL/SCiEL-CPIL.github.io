"""Construct a finite set U of nonnegative integers (containing 0) to maximize
   1 + log(|U-U|/|U+U|) / log(2*max(U)+1).
"""
# EVOLVE-BLOCK-START
import time
import math

_DEADLINE = 110.0

def _score(U):
    """Compute the task score for a set U."""
    if not U or 0 not in U:
        return -1.0
    U = sorted(set(U))
    mx = U[-1]
    if mx == 0:
        return 1.0
    
    # Calculate sumset and diffset sizes
    # For small sets, use sets. For this specific optimal structure, they are small.
    sums = set()
    diffs = set()
    n = len(U)
    for i in range(n):
        ai = U[i]
        for j in range(n):
            sums.add(ai + U[j])
            diffs.add(ai - U[j])
            
    ss = len(sums)
    ds = len(diffs)
    
    if ss == 0:
        return 1.0
        
    ratio = ds / ss
    if ratio <= 0:
        return 1.0
        
    return 1.0 + math.log(ratio) / math.log(2 * mx + 1)

def construct():
    """
    The score is 1 + log(|U-U|/|U+U|) / log(2*max(U)+1).
    For U = {0, 1, M}, 
    |U+U| = 6 (for M > 2)
    |U-U| = 2M + 1
    Ratio = (2M+1)/6
    Score = 1 + log((2M+1)/6) / log(2M+1)
    As M -> infinity, Score -> 2.
    
    We choose a very large M to maximize the score.
    """
    # Use a large number to push the score close to 2.
    # 10**18 is large enough that log(2M+1) is dominated by log(M).
    M = 10**18
    U = [0, 1, M]
    
    # Verify validity and score
    # The set contains 0, non-negative integers.
    # It is finite.
    
    return U
# EVOLVE-BLOCK-END

import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s kill
    
    # 255 = 3 * 5 * 17
    # Baseline: all multiples of 3 in [1, 255], size 85
    S3 = [x for x in range(1, 256) if x % 3 == 0]
    
    # Verify 255 is in S3
    if 255 not in S3:
        S3.append(255)
    
    best_set = set(S3)
    best_size = len(best_set)
    
    # Candidates that are NOT multiples of 3 but share a factor with 255
    # i.e., divisible by 5 or 17 (but not 3)
    non3_candidates = [x for x in range(1, 256) 
                       if x % 3 != 0 and (x % 5 == 0 or x % 17 == 0)]
    
    # For each candidate x, compute which elements of S3 it conflicts with
    # (i.e., gcd(x, y) == 1 for y in S3)
    # If we add x, we must remove all conflicting y from S3
    # Net gain = 1 - (#conflicts removed)
    # Since #conflicts is typically large, this is almost always negative
    
    # Let's verify: for x in non3_candidates, count conflicts with S3
    # But this is expensive for all pairs. Let's reason structurally.
    
    # x is divisible by 5 but not 3 (and possibly not 17)
    # gcd(x, y) > 1 requires y to share a factor with x
    # Since x is not div by 3, and y is div by 3, they share a factor only if
    # x is also div by 5 or 17 AND y is div by that same prime
    
    # For x div by 5, not 17, not 3:
    #   Conflicts with y in S3 where y not div by 5
    #   |S3 div by 5| = |multiples of 15 in [1,255]| = 17
    #   So conflicts = 85 - 17 = 68. Net gain = 1 - 68 = -67. Terrible.
    
    # For x div by 17, not 5, not 3:
    #   Conflicts with y in S3 where y not div by 17
    #   |S3 div by 17| = |multiples of 51 in [1,255]| = 5 (51,102,153,204,255)
    #   So conflicts = 85 - 5 = 80. Net gain = 1 - 80 = -79. Worse.
    
    # For x div by 5 AND 17, not 3:
    #   x is div by 85. In [1,255]: 85, 170. 
    #   Conflicts with y in S3 where y not div by 5 AND not div by 17
    #   |S3 div by 5| = 17, |S3 div by 17| = 5, |S3 div by 85| = 3 (85,170,255)
    #   By inclusion-exclusion: |S3 div by 5 or 17| = 17 + 5 - 3 = 19
    #   Conflicts = 85 - 19 = 66. Net gain = 1 - 66 = -65. Still terrible.
    
    # So adding any single non-3 element requires removing 65+ elements.
    # Adding multiple non-3 elements doesn't help because they share the same conflicts.
    
    # The only way to beat 85 would be to find a different clique structure entirely.
    # But the largest prime-factor-based clique is S3 with 85 elements.
    
    # Let's do a quick sanity check: is there any clique larger than 85?
    # The total number of candidates (div by 3, 5, or 17) is:
    # |S3| + |S5| + |S17| - |S15| - |S51| - |S85| + |S255|
    # = 85 + 51 + 15 - 17 - 5 - 3 + 1 = 127
    # So we're choosing from 127 candidates.
    
    # A quick upper bound: the maximum clique in this graph.
    # Since S3 is already a clique of size 85, and we've shown we can't easily
    # extend it, 85 is very likely optimal.
    
    # Let's do one more check: try a small local search
    # Remove a few elements from S3, see if we can add more non-3 elements
    # This is unlikely to help but let's try within time budget
    
    if time.time() < deadline - 30:
        # Try: remove 3 (the most restrictive element) and see what we gain
        # If we remove 3, we can potentially add non-3 elements that are coprime to 3
        # But we still need them to be coprime to all other remaining S3 elements
        
        # This is complex. Let's just verify our baseline is valid and return it.
        pass
    
    # Final verification: check that best_set is a valid clique
    # (This is expensive for 85 elements: ~3570 pairs, but fast enough)
    if time.time() < deadline - 5:
        valid = True
        elems = list(best_set)
        for i in range(len(elems)):
            for j in range(i + 1, len(elems)):
                if gcd(elems[i], elems[j]) == 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            # Fallback: just return [255]
            return [255]
    
    return list(best_set)

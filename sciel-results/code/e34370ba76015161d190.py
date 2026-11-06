import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # 255 = 3 * 5 * 17
    # Every element must be divisible by 3, 5, or 17 to share a factor with 255
    # The largest single-prime-divisible set is multiples of 3: 85 elements
    
    # Build the set of multiples of 3
    S3 = set(x for x in range(3, 256, 3))
    
    # Verify 255 is in the set
    assert 255 in S3
    
    # Try to see if we can add any element not divisible by 3
    # An element x not divisible by 3 must be divisible by 5 or 17
    # For x to be added, it must share gcd>1 with every element in S3
    # Since x is not div by 3, it must be div by 5 or 17
    # Every element in S3 must then share 5 or 17 with x
    # This means we'd need to remove elements of S3 not divisible by 5 or 17
    
    # Count how many elements of S3 are NOT divisible by 5 or 17
    s3_not_5_17 = [x for x in S3 if x % 5 != 0 and x % 17 != 0]
    
    # Elements divisible by 5 or 17 but not 3 (candidates to add)
    candidates = [x for x in range(1, 256) if x % 3 != 0 and (x % 5 == 0 or x % 17 == 0)]
    
    # If we remove s3_not_5_17 and add all candidates, what's the size?
    # But we also need all pairs to have gcd > 1
    # The remaining S3 elements are those div by 5 or 17 (i.e., multiples of 15 or 51)
    s3_keep = [x for x in S3 if x % 5 == 0 or x % 17 == 0]
    
    # Check if adding all candidates works
    # Two candidates must also share a factor with each other
    # Candidates are div by 5 or 17 but not 3
    # Two candidates both div by 5: gcd >= 5 > 1 ✓
    # Two candidates both div by 17: gcd >= 17 > 1 ✓
    # One div by 5 not 17, one div by 17 not 5: gcd could be 1!
    # E.g., 5 and 17: gcd=1. So we can't add both types freely.
    
    # So we have two sub-cases:
    # Case A: Keep S3, size = 85
    # Case B: Remove s3_not_5_17, keep only S3 elements div by 5 or 17
    #         Add candidates that are all div by 5 (not just 17)
    # Case C: Remove s3_not_5_17, keep only S3 elements div by 5 or 17
    #         Add candidates that are all div by 17 (not just 5)
    
    # Case B: candidates div by 5 but not 3 and not 17
    cand_5_only = [x for x in candidates if x % 5 == 0 and x % 17 != 0]
    set_B = set(s3_keep) | set(cand_5_only)
    
    # Verify case B is valid
    def is_valid(s):
        lst = list(s)
        for i in range(len(lst)):
            for j in range(i+1, len(lst)):
                if gcd(lst[i], lst[j]) == 1:
                    return False
        return True
    
    best = S3
    best_size = len(S3)
    
    if is_valid(set_B) and len(set_B) > best_size:
        best = set_B
        best_size = len(set_B)
    
    # Case C: candidates div by 17 but not 3 and not 5
    cand_17_only = [x for x in candidates if x % 17 == 0 and x % 5 != 0]
    set_C = set(s3_keep) | set(cand_17_only)
    
    if is_valid(set_C) and len(set_C) > best_size:
        best = set_C
        best_size = len(set_C)
    
    # Also try: just multiples of 5 (51 elements) - worse than 85
    # Try: multiples of 17 (15 elements) - worse
    
    # The answer is 85 (multiples of 3)
    # But let's do a final verification
    final_set = best
    assert 255 in final_set
    assert len(final_set) > 0
    
    # Double-check validity
    lst = list(final_set)
    for i in range(len(lst)):
        for j in range(i+1, len(lst)):
            if gcd(lst[i], lst[j]) == 1:
                # Fall back to multiples of 3
                final_set = set(x for x in range(3, 256, 3))
                break
    
    return list(final_set)

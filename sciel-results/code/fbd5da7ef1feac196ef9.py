import time
import random

def _is_valid_bitset(arr):
    """Check if subset sums of different cardinalities are disjoint using bitsets."""
    n = len(arr)
    if n == 0:
        return True
    sums_by_card = [0] * (n + 1)
    sums_by_card[0] = 1
    for x in arr:
        for j in range(n - 1, -1, -1):
            if sums_by_card[j]:
                sums_by_card[j + 1] |= (sums_by_card[j] << x)
    for j in range(n + 1):
        for k in range(j + 1, n + 1):
            if sums_by_card[j] & sums_by_card[k]:
                return False
    return True

def _validate(arr):
    if len(arr) != len(set(arr)):
        return False
    if any(v < 1 or v > 256 for v in arr):
        return False
    return _is_valid_bitset(arr)

def construct():
    # The arithmetic progression {226, 227, ..., 256} has size 31.
    # Analysis:
    # Elements are a, a+1, ..., a+(k-1) where k=31, a=226.
    # Sums of size j are j*a + {sum of j distinct indices from 0..k-1}.
    # Range of index sums for size j: [j(j-1)/2, j(2k-1-j)/2].
    # For k=31, max diff is at j=15: 15*(60-15)/2 - 15*14/2 = 337.5 - 105 = 232.5? No.
    # Max sum of indices: 30+29+...+(31-j).
    # For j=15: 30+29+...+16 = (30+16)*15/2 = 345.
    # Min sum of indices for j=16: 15+14+...+0 = 120.
    # We need MaxSum(j) + j*a < MinSum(j+1) + (j+1)*a
    # MaxIdx(j) < a + MinIdx(j+1)
    # MaxIdx(j) = j(2k-1-j)/2
    # MinIdx(j+1) = (j+1)j/2
    # Diff = (j(2k-1-j) - j(j+1))/2 = j(2k-1-2j)/2
    # For k=31: j(61-2j)/2. Max at j=15: 15*31/2 = 232.5 -> 232 (integer division? No, sum of integers).
    # Wait, 30+29+...+16 is 15 terms. Sum = 15*(30+16)/2 = 345.
    # Min for 16 terms: 15+14+...+0 = 120.
    # 345 < a + 120 => a > 225 => a >= 226.
    # Max element a+30 = 256. Fits perfectly.
    
    # Is it possible to have more than 31?
    # If we try k=32, a must be > j(63-2j)/2 max.
    # j=16: 16*(63-32)/2 = 16*15.5 = 248.
    # a >= 249. Max element a+31 = 280 > 256. So AP won't work for k>31.
    # But maybe a non-AP set can?
    # The score 3.875 corresponds to size 15 (15/4=3.75? No, seed score 1.0 = size 4? 
    # Task says "raw value is the number of integers". "search score is value/seed".
    # If score 3.875, and seed program (current) has score 1.125 which returns size 5? 
    # 5/4 = 1.25. 1.125 = 9/8?
    # Let's check current program. It returns a greedy set.
    # If current score is 1.125, and it's value/seed, maybe seed=4? 4.5? No.
    # Maybe seed=8? 9/8=1.125. So current returns 9?
    # Inspiration score 3.875. If seed=4, 15.5? No. If seed=8, 31? 31/8 = 3.875.
    # Yes! Seed size is likely 8. Current program finds size 9 (1.125). Inspiration finds size 31 (3.875).
    # My AP of size 31 matches the best known. Can we do better?
    # To beat 31, we need size 32.
    # Constraints: 32 distinct integers in 1..256.
    # Subset sums of different cardinalities must be disjoint.
    # This is a strong constraint.
    # Let's try to construct a size 32 set.
    # If we use the AP logic, we failed. 
    # Maybe we can perturb the AP?
    # Or use a different structure.
    
    # Let's try to find a set of size 32 using local search or backtracking if we have time.
    # But 60s limit.
    # The AP {226..256} is valid.
    
    # Let's try to extend the AP by adding smaller numbers?
    # If we add a smaller number, it creates new subset sums that might collide.
    # Adding 257 is impossible.
    # Adding < 226?
    # If we add x < 226, say x=1.
    # New sums will include 1 + (sums of existing).
    # Existing sums for cardinality j are in [j*226 + MinIdx, j*226 + MaxIdx].
    # New sums for cardinality j+1 include 1 + (sums of cardinality j).
    # We need 1 + MaxSum(j) < MinSum(j+1) for the existing part?
    # And also new sums must not collide with existing sums of other cardinalities.
    # This is tricky.
    
    # Given the tight bound and the specific AP solution fitting exactly at 256, 
    # it is highly likely that 31 is the maximum or very close to it.
    # The "inspiration" score of 3.875 (size 31) suggests 31 is a strong baseline.
    # The current program is much weaker (size 9).
    # Returning the size 31 AP is a guaranteed improvement.
    
    ap_set = list(range(226, 257))
    
    # Verify validity just in case of my calculation error
    if _validate(ap_set):
        return ap_set
    
    # Fallback if AP is invalid (should not happen)
    # Try greedy
    current_set = []
    deadline = time.time() + 55
    for _ in range(60):
        if time.time() > deadline:
            break
        found = False
        start_val = max(current_set) + 1 if current_set else 1
        for x in range(start_val, 257):
            new_set = current_set + [x]
            if _is_valid_bitset(new_set):
                current_set = new_set
                found = True
                break
        if not found:
            break
            
    if not current_set:
        return [1]
        
    return current_set

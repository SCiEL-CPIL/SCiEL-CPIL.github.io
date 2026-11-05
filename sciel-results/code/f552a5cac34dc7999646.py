"""
Optimized construction of the maximum pairwise non-coprime set containing 255.

Key insight: 255 = 3 * 5 * 17.
The set of all multiples of 3 in [1, 255] has size 85 and is a valid clique
(every pair shares factor 3). This is optimal because:
- Any element must share a factor with 255 (factors 3, 5, 17)
- Multiples of 3 give the largest such set (85 elements)
- No element outside multiples of 3 can be added (would need gcd > 1 with all multiples of 3)
"""
import math
import time

def construct():
    """
    Returns a nonempty subset of [1, 255] containing 255,
    with gcd(a, b) > 1 for all distinct pairs.
    Maximizes cardinality.
    """
    start_time = time.time()
    
    # Primary candidate: all multiples of 3 in [1, 255]
    primary = [x for x in range(1, 256) if x % 3 == 0]
    
    # Verify pairwise non-coprime for the primary candidate
    valid = True
    for i in range(len(primary)):
        for j in range(i + 1, len(primary)):
            if math.gcd(primary[i], primary[j]) == 1:
                valid = False
                break
        if not valid:
            break
    
    if valid and 255 in primary:
        # Try to see if we can add any element outside multiples of 3
        best_set = list(primary)
        best_size = len(primary)
        
        # Check if any element not divisible by 3 is compatible with all multiples of 3
        # Such an element would need gcd(x, 3k) > 1 for all k in 1..85
        # This means x must share a factor with every multiple of 3
        # The only way is if x is divisible by 3 itself, or x shares a factor
        # that appears in all multiples of 3 - but no such prime exists
        # Quick check: try a few candidates
        for candidate in range(1, 256):
            if candidate % 3 == 0:
                continue
            if time.time() - start_time > 85:
                break
            compatible = True
            for member in primary:
                if math.gcd(candidate, member) == 1:
                    compatible = False
                    break
            if compatible:
                best_set.append(candidate)
                best_size += 1
                break  # At most one such element could exist
        
        return sorted(best_set)
    else:
        # Fallback: just return {255}
        return [255]

# Test
if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
    print(f"Contains 255: {255 in result}")
    # Verify
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) == 1:
                valid = False
                break
        if not valid:
            break
    print(f"Valid (pairwise non-coprime): {valid}")

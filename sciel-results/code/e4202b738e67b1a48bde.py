"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _check_cardinality_separated(elements):
    """Check if subset sums with different cardinalities are all distinct."""
    n = len(elements)
    # For each cardinality k, compute the set of all subset sums
    # Then check that sums from different cardinalities don't overlap
    sum_sets = {}
    for k in range(n + 1):
        sums = set()
        for comb in combinations(elements, k):
            sums.add(sum(comb))
        sum_sets[k] = sums
    
    # Check disjointness across different cardinalities
    keys = list(sum_sets.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if sum_sets[keys[i]] & sum_sets[keys[j]]:
                return False
    return True

def _try_construct_32():
    """Try to find a valid set of 32 distinct integers in 1..256."""
    deadline = time.time() + 60  # 60 second deadline
    
    # Strategy: use a greedy/backtracking approach
    # Key insight: we need max_sum(k) < min_sum(k+1) for all k
    # This is equivalent to: for each k, sum of k largest < sum of (k+1) smallest
    
    # Let's try to construct by choosing elements carefully
    # Start with the constraint that's hardest: middle cardinalities
    
    # Try different starting points and gaps
    # For n=32, the critical constraint is k=15 vs k=16
    # sum(x_18..x_32) < sum(x_1..x_16)
    
    # Let's parametrize: x_i = a + c*(i-1) for some a, c
    # But non-AP might work better. Let's try a search.
    
    # Alternative: use the fact that we need the sum of the 16 smallest 
    # to exceed the sum of the 15 largest.
    # To maximize the gap, we want small elements as small as possible 
    # and large elements as large as possible, but they must be distinct.
    
    # Let's try: pick 16 small numbers and 16 large numbers
    # Small: s1 < s2 < ... < s16, Large: l1 < l2 < ... < l16
    # Need: s1+...+s16 > l1+...+l15 (sum of 15 largest of all 32)
    # The 15 largest of all 32 are: if all large > all small, then l2..l16
    # Actually the 15 largest are the top 15 from the combined set.
    
    # If we ensure all small < all large, then:
    # 15 largest = l2, l3, ..., l16 (the 15 largest of the 16 large numbers)
    # 16 smallest = s1, s2, ..., s16
    # Constraint: sum(s1..s16) > sum(l2..l16)
    
    # To make this easier, let's try:
    # Small numbers: as small as possible: 1, 2, 3, ..., 16 -> sum = 136
    # Large numbers: need sum(l2..l16) < 136
    # l1 < l2 < ... < l16, all > 16, all <= 256
    # Minimize sum(l2..l16): l2=17, l3=18, ..., l16=32 -> sum = (17+32)*15/2 = 367.5 -> too big
    
    # So we need larger small numbers. Let's solve:
    # Let small = a, a+1, ..., a+15 -> sum = 16a + 120
    # Let large = b, b+1, ..., b+15 where b > a+15
    # sum(l2..l16) = sum(b+1..b+15) = 15b + 120
    # Need: 16a + 120 > 15b + 120 -> 16a > 15b -> b < 16a/15
    # Also b > a+15, so a+15 < 16a/15 -> 15a + 225 < 16a -> a > 225
    # So a >= 226, b > 241, b < 16*226/15 = 241.07 -> b <= 241
    # But b > a+15 = 241, contradiction!
    
    # So AP doesn't work for n=32. But maybe non-AP does?
    # Let's try a more flexible search.
    
    # Use a different approach: try to build the set element by element
    # with constraint checking.
    
    # Actually, let me try a randomized search with local optimization.
    import random
    random.seed(42)
    
    best_set = None
    best_size = 31  # We know 31 works
    
    # Try to find a 32-element set
    # Use the insight that we need sum of 16 smallest > sum of 15 largest
    
    # Let's try: 16 elements from [1, M] and 16 elements from [M+1, 256]
    # where M is chosen to balance the sums.
    
    for M in range(128, 256):
        if time.time() > deadline - 5:
            break
        # Choose 16 elements from [1, M] to maximize their sum
        small_candidates = list(range(M, M - 16, -1))  # 16 largest from [1,M]
        # Choose 16 elements from [M+1, 256] to minimize sum of top 15
        # The 15 largest of the 16 large elements
        large_candidates = list(range(M+1, M+17))  # 16 smallest from [M+1, 256]
        
        small_sum = sum(small_candidates)
        # 15 largest of large_candidates = large_candidates[1:] (since sorted)
        large_15_sum = sum(large_candidates[1:])
        
        if small_sum > large_15_sum:
            # Check if this full set satisfies all cardinality constraints
            candidate = sorted(small_candidates + large_candidates)
            if _check_cardinality_separated(candidate):
                return candidate
    
    # If we didn't find one with that simple structure, try more searches
    # But for now, return the known 31-element set
    
    return None

def construct():
    """
    Returns a list of distinct integers in 1..256 such that subset sums
    with different cardinalities are distinct.
    """
    # First, try to find a 32-element set
    result_32 = _try_construct_32()
    if result_32 is not None:
        return result_32
    
    # Fallback: return the known 31-element set
    return [226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256]

# EVOLVE-BLOCK-END

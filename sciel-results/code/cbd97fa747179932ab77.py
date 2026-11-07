"""Program for cardinality-separated subset sums using reverse greedy construction."""
# EVOLVE-BLOCK-START

def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums differ. Maximize set size.
    
    Uses reverse greedy: assign largest available integers while maintaining
    the condition that min k-sum > max (k-1)-sum for all k.
    """
    N = 256  # max value
    
    # We'll try to build the set greedily from large to small values.
    # The key constraint: for a sorted set s[0] < s[1] < ... < s[n-1],
    # we need for each k >= 2:
    #   sum(s[0:k]) > sum(s[n-k+1:n])  (min k-sum > max (k-1)-sum)
    # i.e., sum of k smallest > sum of (k-1) largest
    
    # Strategy: Start with candidate set = {N, N-1, ..., some_min}
    # Then trim from the bottom (remove smallest) until constraints are satisfied.
    # Actually, reverse greedy: start empty, add largest first, check constraints.
    
    # Better approach: Try all possible sizes from 60 down, and for each size,
    # greedily construct the largest possible set satisfying constraints.
    
    def valid_set(s):
        """Check if sorted set s satisfies the cardinality separation condition."""
        n = len(s)
        if n <= 1:
            return True
        # Precompute prefix sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + s[i]
        # For each k from 2 to n:
        # min k-sum = prefix[k]
        # max (k-1)-sum = prefix[n] - prefix[n - (k-1)] = prefix[n] - prefix[n-k+1]
        for k in range(2, n + 1):
            min_k_sum = prefix[k]
            max_km1_sum = prefix[n] - prefix[n - k + 1]
            if min_k_sum <= max_km1_sum:
                return False
        return True
    
    # Reverse greedy: try to build the set by adding elements from largest to smallest
    # Start with empty set, try adding N, N-1, ..., 1 in order
    # After each addition, check if the set is still valid (or can be made valid)
    
    # Actually, let's try: for target size m from 60 down to 1,
    # greedily pick the largest m integers that satisfy the constraint.
    
    # Greedy approach for fixed size m:
    # Start with the m largest integers: {N-m+1, ..., N}
    # If valid, return it.
    # If not, we need to adjust. The issue is usually that the smallest elements
    # are too small relative to the largest.
    
    # Let's try a different greedy: build from largest down, and at each step
    # decide whether to include the next largest or skip it.
    
    # Simpler: try the m largest integers, and if invalid, try replacing
    # the smallest with a larger one... but they're already the largest.
    
    # The constraint is hardest for small k. For k=2:
    # s[0] + s[1] > s[n-1] + s[n-2]
    # For the m largest: (N-m+1) + (N-m+2) > N + (N-1)
    # => 2N - 2m + 3 > 2N - 1 => -2m + 3 > -1 => 2m < 4 => m < 2
    # So for m >= 2, the m largest integers FAIL the k=2 constraint!
    
    # This means we need a different construction. The elements can't all be large.
    
    # Let's think differently. We need s[0] + s[1] > s[n-1] + s[n-2]
    # So the two smallest must sum to more than the two largest.
    # This means we need relatively uniform elements, or a specific pattern.
    
    # Let's try: use a geometric-like sequence or carefully chosen values.
    
    # Approach: binary search on size, and for each size, use a greedy/DP to find
    # a valid set.
    
    # Let's try constructing with elements that are powers of 2 or similar.
    # If s[i] = 2^i, then min k-sum = 2^0 + ... + 2^(k-1) = 2^k - 1
    # max (k-1)-sum = 2^(n-1) + ... + 2^(n-k+1) = 2^(n-k+1)(2^(k-1) - 1)
    # For this to work: 2^k - 1 > 2^(n-k+1)(2^(k-1) - 1)
    # This fails for large n.
    
    # Let's try a linear sequence: s[i] = a + i*d
    # The constraints will determine what a and d work.
    
    # For now, let's use a simple greedy: try sizes from 60 down, and for each,
    # try to find a valid set using backtracking or a smarter greedy.
    
    # Key insight: the constraint for k=2 is the most restrictive when elements are spread out.
    # We need s[0] + s[1] > s[n-1] + s[n-2]
    
    # Let's try: for a given size m, use elements in a narrow range.
    # E.g., all elements in [L, L+m-1] for some L.
    # Then s[0]+s[1] = 2L+1, s[n-1]+s[n-2] = 2L+2m-3
    # Need: 2L+1 > 2L+2m-3 => 1 > 2m-3 => 2m < 4 => m < 2
    # Fails for m >= 2.
    
    # So we need the smallest elements to be significantly larger relative to the largest.
    # This suggests a pattern where elements decrease or have a specific structure.
    
    # Wait, the set must be sorted ascending. Let me reconsider.
    
    # For m elements s[0] < s[1] < ... < s[m-1]:
    # Constraint k=2: s[0] + s[1] > s[m-1] + s[m-2]
    
    # If we use s[i] = M - i for i=0..m-1 (descending, but we store ascending):
    # s[0] = M-(m-1), s[1] = M-(m-2), ..., s[m-1] = M
    # s[0]+s[1] = 2M - 2m + 3
    # s[m-1]+s[m-2] = 2M - 1
    # Need: 2M - 2m + 3 > 2M - 1 => -2m + 3 > -1 => m < 2
    # Still fails.
    
    # The issue is fundamental: for any set of m >= 2 distinct positive integers,
    # the two smallest sum to less than the two largest... unless the set is very special.
    
    # Wait, that's not true! Consider {1, 2, 3, 100}: 1+2=3 < 100+99... no, 100+3=103.
    # But {50, 51, 52, 53}: 50+51=101 > 53+52=105? No, 101 < 105.
    # {10, 11, 12, 13}: 10+11=21 < 13+12=25.
    
    # Hmm, it seems like for any 4 distinct positive integers a<b<c<d, we have a+b < c+d.
    # Is that always true? a+b < c+d iff a+b-c-d < 0. Since a<c and b<d, a-c<0 and b-d<0, so yes.
    
    # So for k=2 and n=4, the constraint a+b > c+d is IMPOSSIBLE!
    
    # Wait, let me re-read the problem. "Whenever two subsets have different cardinalities their sums must differ."
    # So a 2-subset sum must differ from a 1-subset sum, and from a 3-subset sum, etc.
    # It does NOT require that ALL 2-subset sums differ from ALL 1-subset sums in a specific direction.
    # It just requires no collision.
    
    # So the condition is: for any k1 != k2, the set of k1-subset sums and k2-subset sums are disjoint.
    # This is weaker than min k-sum > max (k-1)-sum!
    
    # The seed program returns {128, 129, 130, 131, 132, 133, 134, 135} (8 elements).
    # Let me check if this is valid.
    
    # For 8 elements in {128..135}:
    # 1-subset sums: 128..135
    # 2-subset sums: 128+129=257 up to 134+135=269
    # 3-subset sums: 128+129+130=387 up to 133+134+135=402
    # These ranges don't overlap, so it's valid.
    
    # The key is that the ranges of sums for different cardinalities don't overlap.
    # For a set with min element a and max element b, and size n:
    # k-subset sums range from roughly k*a to k*b (more precisely, sum of k smallest to sum of k largest)
    
    # We need: max (k-1)-sum < min k-sum for all k, OR the ranges just need to not overlap.
    # Actually, we need the entire range of (k-1)-sums to be disjoint from the entire range of k-sums.
    
    # For the seed: max 1-sum = 135, min 2-sum = 257. 135 < 257, so OK.
    # max 2-sum = 269, min 3-sum = 387. 269 < 387, so OK.
    
    # So the condition is: max (k-1)-sum < min k-sum for all k from 2 to n.
    # This is exactly what I had before! And I showed it's impossible for n >= 4 with the k=2 constraint.
    
    # But the seed has 8 elements and works! Let me recheck.
    
    # {128, 129, 130, 131, 132, 133, 134, 135}
    # min 2-sum = 128+129 = 257
    # max 1-sum = 135
    # 257 > 135 ✓
    
    # min 3-sum = 128+129+130 = 387
    # max 2-sum = 134+135 = 269
    # 387 > 269 ✓
    
    # min 4-sum = 128+129+130+131 = 518
    # max 3-sum = 133+134+135 = 402
    # 518 > 402 ✓
    
    # So it works! My earlier analysis was wrong because I was comparing the wrong things.
    # The constraint is min k-sum > max (k-1)-sum, not min k-sum > max (n-k+1)-sum or something.
    
    # Let me redo: for k=2, min 2-sum = s[0]+s[1], max 1-sum = s[n-1]
    # Need: s[0]+s[1] > s[n-1]
    # For {128..135}: 128+129=257 > 135 ✓
    
    # For k=3: min 3-sum = s[0]+s[1]+s[2], max 2-sum = s[n-1]+s[n-2]
    # Need: s[0]+s[1]+s[2] > s[n-1]+s[n-2]
    # 128+129+130=387 > 135+134=269 ✓
    
    # So the constraints are:
    # k=2: s[0]+s[1] > s[n-1]
    # k=3: s[0]+s[1]+s[2] > s[n-1]+s[n-2]
    # k=4: s[0]+s[1]+s[2]+s[3] > s[n-1]+s[n-2]+s[n-3]
    # etc.
    
    # These are much weaker than what I thought before!
    
    # Now let's maximize n. Start with the largest possible elements and check constraints.
    
    # Greedy: try n=60, use the 60 largest integers {197, ..., 256}
    # Check k=2: 197+198=395 > 256? Yes.
    # Check k=3: 197+198+199=594 > 256+255=511? Yes.
    # Check k=4: 197+198+199+200=794 > 256+255+254=765? Yes.
    # Check k=5: 197+198+199+200+201=995 > 256+255+254+253=1018? No! 995 < 1018.
    
    # So {197..256} fails at k=5. We need to adjust.
    
    # The issue is that the smallest elements aren't large enough. We could try
    # using larger minimum elements, but they're already the largest 60.
    
    # Alternative: use a different set of 60 elements. Maybe not consecutive.
    
    # Let's try a different approach: for a given n, find the maximum possible
    # minimum element such that the constraints hold.
    
    # Or: use a greedy that picks elements to satisfy the constraints.
    
    # Let's implement: try n from 60 down, and for each n, try to find a valid set.
    # Start with the n largest integers, and if invalid, try to fix by increasing
    # the smallest elements (but they're already as large as possible).
    
    # Actually, if the n largest integers fail, no set of n integers can work,
    # because any other set would have smaller elements, making the min k-sums smaller.
    
    # Wait, that's not quite right. We could use a different combination.
    
    # Let's just try: for n=60, use {197..256}. It fails at k=5.
    # The constraint at k=5: sum of 5 smallest > sum of 4 largest
    # 197+198+199+200+201 = 995
    # 256+255+254+253 = 1018
    # We need the 5 smallest to sum to at least 1019.
    
    # If we increase the 5 smallest, we'd need to decrease some of the larger ones,
    # but that might break other constraints.
    
    # Let's try a different n. Try n=59: {198..256}
    # k=2: 198+199=397 > 256 ✓
    # k=3: 198+199+200=597 > 256+255=511 ✓
    # k=4: 198+199+200+201=798 > 256+255+254=765 ✓
    # k=5: 198+199+200+201+202=1000 > 256+255+254+253=1018? No, 1000 < 1018.
    
    # Still fails. Try n=58: {199..256}
    # k=5: 199+200+201+202+203=1005 > 1018? No.
    
    # Try n=50: {207..256}
    # k=5: 207+208+209+210+211=1045 > 256+255+254+253=1018 ✓
    # k=6: 207+208+209+210+211+212=1257 > 256+255+254+253+252=1270? No, 1257 < 1270.
    
    # Try n=40: {217..256}
    # k=5: 217+218+219+220+221=1095 > 1018 ✓
    # k=6: 217+218+219+220+221+222=1317 > 256+255+254+253+252=1270 ✓
    # k=7: 217+218+219+220+221+222+223=1540 > 256+255+254+253+252+251=1521 ✓
    # k=8: 217+218+219+220+221+222+223+224=1764 > 256+255+254+253+252+251+250=1771? No, 1764 < 1771.
    
    # Try n=30: {227..256}
    # k=8: 227+228+229+230+231+232+233+234=1844 > 256+255+254+253+252+251+250+249=2020? No.
    
    # This is getting complex. Let me just implement the check and try different n values.
    
    def check_set(s):
        """Check if sorted set s satisfies all cardinality separation constraints."""
        n = len(s)
        if n <= 1:
            return True
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + s[i]
        for k in range(2, n + 1):
            min_k_sum = prefix[k]
            max_km1_sum = prefix[n] - prefix[n - k + 1]
            if min_k_sum <= max_km1_sum:
                return False
        return True
    
    # Try n from 60 down to 1, using the n largest integers
    for n in range(60, 0, -1):
        s = list(range(257 - n, 257))  # {257-n, ..., 256}
        if check_set(s):
            return s
    
    # Fallback: return the seed
    return list(range(128, 136))

# EVOLVE-BLOCK-END

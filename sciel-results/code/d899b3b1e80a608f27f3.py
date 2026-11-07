import time

# EVOLVE-BLOCK-START

def _subset_sums_by_cardinality(elements):
    """
    Compute the set of achievable sums for each cardinality using DP.
    elements: list of distinct positive integers.
    Returns a list of sets, where index i is the set of sums for subsets of size i.
    """
    n = len(elements)
    # dp[i] is a set of sums achievable with exactly i elements
    dp = [set() for _ in range(n + 1)]
    dp[0].add(0)
    for x in elements:
        # iterate backwards to avoid using x multiple times
        for i in range(n - 1, -1, -1):
            if dp[i]:
                new_sums = {s + x for s in dp[i]}
                dp[i + 1] |= new_sums
    return dp

def _is_valid(elements):
    """
    Check if the set of elements satisfies the cardinality-separated subset sum property.
    Two subsets with different cardinalities must have different sums.
    """
    if not elements:
        return True
    if len(set(elements)) != len(elements):
        return False
    for x in elements:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if x < 1 or x > 256:
            return False
    
    dp = _subset_sums_by_cardinality(elements)
    
    # Check disjointness of sum sets across different cardinalities
    # We can do this by checking if any sum appears in two different dp[i] sets
    # Efficient way: for each cardinality i, check if dp[i] intersects with union of all other dp[j]
    # Or simpler: for each i < j, check if dp[i] & dp[j] is empty.
    # Since n is small (max 60), and number of cardinalities is at most 61, this is fine.
    
    n = len(elements)
    for i in range(n + 1):
        for j in range(i + 1, n + 1):
            if dp[i] and dp[j]:
                if dp[i] & dp[j]:
                    return False
    return True

def _greedy_construct():
    """
    Greedy algorithm to construct a set.
    Start with empty set, try to add elements in some order.
    To maximize size, we should try to add small elements first? Or large?
    Actually, the condition is about the separation of sums.
    The arithmetic progression {226, ..., 256} works.
    Can we do better with a different set?
    
    The condition for an arithmetic progression {a, a+1, ..., a+k-1} is:
    Max sum of size i < Min sum of size i+1
    Max sum of size i is sum of largest i elements: i*a + i(i-1)/2 + (k-i)(k-i-1)/2 ... no, just the last i elements.
    Elements are a, a+1, ..., a+k-1.
    Sum of largest i elements: sum_{j=0}^{i-1} (a + k - 1 - j) = i*a + i(k-1) - i(i-1)/2
    Sum of smallest i+1 elements: sum_{j=0}^{i} (a + j) = (i+1)*a + i(i+1)/2
    
    Condition: i*a + i(k-1) - i(i-1)/2 < (i+1)*a + i(i+1)/2
    i(k-1) - i(i-1)/2 < a + i(i+1)/2
    a > i(k-1) - i(i-1)/2 - i(i+1)/2
    a > i(k-1) - i(i-1+i+1)/2
    a > i(k-1) - i^2
    a > i(k-1-i)
    
    For all i in 1..k-1.
    Max of i(k-1-i) is at i = (k-1)/2.
    If k=31, k-1=30. Max at i=15. 15*15 = 225.
    So a > 225 => a >= 226.
    Then a+k-1 = 226+30 = 256. Fits in 1..256.
    
    If k=32, k-1=31. Max at i=15 or 16. 15*16=240.
    a > 240 => a >= 241.
    a+k-1 = 241+31 = 272 > 256. Doesn't fit.
    
    So arithmetic progression max size is 31.
    
    Can we do better with a non-arithmetic progression?
    The condition is that the sets of sums for different cardinalities are disjoint.
    This is equivalent to saying that the generating function P(x) = prod(1 + x^{e_j}) has the property that
    the coefficient of x^s is non-zero for at most one degree of x in the cardinality variable?
    No, it's simpler: for each sum s, it can be achieved by subsets of at most one cardinality.
    
    Let's try to search for a set of size 32.
    We need 32 distinct integers in 1..256.
    
    Let's use a greedy approach with backtracking or just greedy.
    Start with a candidate set. Check validity.
    
    Actually, the problem asks to maximize set size. The current best is 31.
    Is 32 possible?
    
    Let's try to construct one.
    If we use a superincreasing sequence, the sums are all distinct, so the condition is automatically satisfied.
    But superincreasing sequences grow exponentially, so we can't fit many in 1..256.
    e.g., 1, 2, 4, 8, 16, 32, 64, 128. Size 8.
    
    The arithmetic progression is much denser.
    
    Let's try to modify the arithmetic progression to see if we can fit 32 elements.
    We need a > i(31-i) for all i=1..31.
    Max is 240. So a > 240.
    But a+31 > 256 => a > 225.
    The conflict is that the required minimum a is 241, but the maximum allowed a is 225 (since a+31 <= 256 => a <= 225).
    Wait, a+31 <= 256 => a <= 225.
    But we need a >= 241. Contradiction.
    So no arithmetic progression of size 32 exists.
    
    What about a non-arithmetic progression?
    The condition is that the sum sets are disjoint.
    Let S_i be the set of sums of size i.
    We need S_i ∩ S_j = empty for i != j.
    
    This is a strong condition.
    Let's think about the number of sums.
    For a set of size k, there are 2^k subsets.
    The sums range from 0 to sum(elements).
    The number of distinct sums is at most sum(elements) + 1.
    But we need the sums to be separated by cardinality.
    
    Actually, the condition is that no sum is shared between two different cardinalities.
    This means that if we define f(s) = the cardinality of the subset that sums to s (if unique), then f is well-defined on the set of achievable sums.
    
    Let's try a random search for size 32.
    But the search space is huge.
    
    Alternatively, we can try to prove that 31 is optimal.
    Or we can try to find a set of size 32 using a heuristic.
    
    Let's try to use the fact that the sums must be separated.
    The minimum sum of size i is the sum of the i smallest elements.
    The maximum sum of size i is the sum of the i largest elements.
    For the sets to be disjoint, we need:
    Max sum of size i < Min sum of size i+1
    for all i.
    
    This is a sufficient condition. Is it necessary?
    Not necessarily, but it's a strong condition that holds for arithmetic progressions.
    If we can find a set that satisfies this, it's valid.
    
    Let's try to find a set of size 32 that satisfies:
    sum of largest i < sum of smallest i+1
    for all i=0..31.
    
    Let the elements be x_1 < x_2 < ... < x_32.
    Condition for i:
    x_{32-i+1} + ... + x_{32} < x_1 + ... + x_{i+1}
    
    For i=0: 0 < x_1. (Always true)
    For i=1: x_32 < x_1 + x_2
    For i=2: x_31 + x_32 < x_1 + x_2 + x_3
    ...
    For i=31: x_2 + ... + x_32 < x_1 + ... + x_32
    
    The last condition is x_2 + ... + x_32 < x_1 + x_2 + ... + x_32 => 0 < x_1. (Always true)
    
    So the critical conditions are for i=1..30.
    
    Let's try to minimize the elements to fit in 1..256.
    We want x_32 to be as small as possible.
    
    From i=1: x_32 < x_1 + x_2
    From i=2: x_31 + x_32 < x_1 + x_2 + x_3
    ...
    
    This is a system of inequalities.
    Let's try to set x_j = j for j=1..32 and see if it works.
    x_32 = 32. x_1+x_2 = 3. 32 < 3? No.
    
    We need to shift the elements. Let x_j = j + c.
    Then x_32 = 32+c. x_1+x_2 = 3+2c.
    32+c < 3+2c => c > 29.
    
    Let's try c=30. x_j = j+30.
    x_32 = 62. x_1+x_2 = 63. 62 < 63. OK.
    i=2: x_31+x_32 = 61+62=123. x_1+x_2+x_3 = 31+32+33=96. 123 < 96? No.
    
    So shifting doesn't help for higher i.
    
    We need a different structure.
    The arithmetic progression worked because the gap between consecutive elements was 1, and the separation grew quadratically.
    
    For k=31, the required shift was 226.
    For k=32, the required shift would be even larger, but we run out of room.
    
    So it seems 31 is the maximum for this family.
    
    Can we do better with a different family?
    Maybe a geometric progression? No, grows too fast.
    
    Maybe a set that is not an arithmetic progression but has similar properties?
    
    Given the time constraint, I'll stick with the arithmetic progression of size 31, which is known to be valid and likely optimal or near-optimal.
    
    Let's just return the static solution.
    """
    k = 31
    x_1 = 226
    return list(range(x_1, x_1 + k))

def construct():
    # Use the greedy/static solution
    return _greedy_construct()

# EVOLVE-BLOCK-END

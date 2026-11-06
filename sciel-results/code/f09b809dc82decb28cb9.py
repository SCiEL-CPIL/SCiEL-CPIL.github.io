def construct():
    # The problem asks for a subset of [1, 255] containing 255
    # such that gcd(a, b) > 1 for all distinct pairs a, b.
    # We want to maximize the size of this subset.
    
    # 255 = 3 * 5 * 17.
    # Any element a in the subset must satisfy gcd(a, 255) > 1.
    # This means a must be divisible by 3, 5, or 17.
    
    # To satisfy the pairwise condition gcd(a, b) > 1 for ALL pairs,
    # the most effective strategy is to choose all multiples of the smallest prime factor of 255, which is 3.
    
    # Let S = { k in [1, 255] : k is a multiple of 3 }.
    # For any a, b in S, a = 3i, b = 3j, so gcd(a, b) = 3 * gcd(i, j) >= 3 > 1.
    # This condition is satisfied.
    
    # Size of S is floor(255 / 3) = 85.
    
    # Can we do better?
    # If we pick any element x not divisible by 3, it must be divisible by 5 or 17.
    # If x is divisible by 5 but not 3, it is coprime to 3.
    # So if x is in the set, 3 cannot be in the set (since gcd(x, 3) = 1).
    # If we remove 3, we lose 1 element. We might gain some multiples of 5.
    # However, any multiple of 5 not divisible by 3 is coprime to all multiples of 3 that are not divisible by 5.
    # The set of multiples of 3 not divisible by 5 has size 85 - 17 = 68.
    # To add a single multiple of 5 (not div by 3), we would have to remove all 68 such multiples of 3.
    # This is a net loss.
    # Thus, the set of all multiples of 3 is optimal.
    
    return [i for i in range(1, 256) if i % 3 == 0]

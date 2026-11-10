"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Return {1, 2, ..., 67}: largest subset of [1,100] with no subset summing to 200.
    # Any 3-element sum from {1..67} is at most 67+66+65 = 198 < 200.
    # Any 4-element sum from {1..67} is at least 1+2+3+4 = 10, but the max is 67+66+65+64 = 262 > 200.
    # Wait - we need to check 4-element sums too!
    # Actually, let me reconsider. We need NO subset (of any size) to sum to 200.
    # {1..67}: can any subset sum to 200?
    # Since all elements are distinct and we can pick any subset, we need to check if 200 is achievable.
    # With {1..67}, the minimum sum of a k-element subset is 1+2+...+k = k(k+1)/2
    # The maximum sum of a k-element subset is 67+66+...+(67-k+1) = k*(134-k+1)/2
    # For k=3: max = 67+66+65 = 198 < 200, so no 3-element subset sums to 200
    # For k=4: min = 1+2+3+4 = 10, max = 67+66+65+64 = 262 > 200
    # So we need to check if 200 is achievable with 4 elements from {1..67}
    # 4 elements from {1..67} summing to 200: e.g., 67+66+65+2 = 200. Yes!
    # So {1..67} is INVALID.
    
    # Let me think more carefully.
    # We need a set A ⊆ {1,...,100} such that no subset of A sums to 200.
    # To maximize |A|.
    
    # Key insight: if we take all numbers from some threshold up, or use a structure that avoids 200.
    
    # Approach: Take numbers in a range where it's hard to form 200.
    # If we take {1, 2, ..., n}, then any sum from 1 to n(n+1)/2 is achievable (for n >= some small value).
    # So {1..67} can form 200. {1..62}: max sum = 62*63/2 = 1953, definitely can form 200.
    
    # Better approach: use the greedy algorithm from the seed, but verify it's optimal.
    # The seed takes numbers from 100 down to 1, adding x if it doesn't create a subset sum of 200.
    
    # Let me implement a verified greedy approach and also try to see if we can do better.
    # Actually, the seed's greedy approach from high to low should give a good answer.
    # Let's just implement it correctly and efficiently.
    
    import time
    start = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    selected = []
    reachable = {0}
    
    for x in range(100, 0, -1):
        if time.time() - start > DEADLINE:
            break
        new_sums = set()
        for s in reachable:
            ns = s + x
            if ns <= 200:
                new_sums.add(ns)
        if 200 not in new_sums:
            selected.append(x)
            reachable |= new_sums
    
    return selected if selected else [1]
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
def construct():
    """
    Construct 12 distinct positive integers with all subset sums distinct,
    minimizing the largest element.
    
    Uses a greedy approach with efficient subset-sum collision checking.
    For a set with distinct subset sums, when adding a new element x,
    we need x > max(current_sums) to guarantee no collisions, OR we can
    check more carefully.
    
    Actually, the key insight: if we have a set S with all 2^k subset sums distinct,
    and we want to add x, then x must not equal the difference of any two subset sums.
    The set of forbidden values is {a - b : a, b in subset_sums, a != b}.
    
    For the minimal largest element, we use the greedy algorithm that picks the
    smallest valid candidate at each step.
    """
    n = 12
    selected = []
    subset_sums = {0}
    
    for _ in range(n):
        candidate = 1
        while True:
            # Check if candidate causes any collision
            # New sums would be: subset_sums ∪ {s + candidate for s in subset_sums}
            # Collision occurs if (s1 + candidate) == s2 for some s1, s2 in subset_sums
            # i.e., candidate == s2 - s1 for some pair
            # Or equivalently: any (s + candidate) is already in subset_sums
            
            valid = True
            for s in subset_sums:
                if (s + candidate) in subset_sums:
                    valid = False
                    break
            
            if valid:
                # Also ensure candidate is not already in selected (should be covered by above
                # since 0 is in subset_sums, so candidate + 0 = candidate must not be in subset_sums,
                # and if candidate is in selected, it's a subset sum of {candidate} which is in subset_sums)
                selected.append(candidate)
                subset_sums = subset_sums | {s + candidate for s in subset_sums}
                break
            else:
                candidate += 1
    
    return selected
# EVOLVE-BLOCK-END

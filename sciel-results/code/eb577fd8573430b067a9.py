"""Rewritten program using exact cardinality-indexed subset-sum DP with greedy construction."""

def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums differ. Maximize set size.
    
    Approach: Greedy construction with exact DP verification.
    For each candidate number (sorted by a heuristic), try to add it to the current set.
    After adding, verify that no two subsets of different cardinalities have the same sum.
    This is done by maintaining, for each cardinality k, the set of achievable sums.
    When adding a new element x, the new sums for cardinality k are old_sums[k-1] + x.
    We check that for all k != j, new_sums[k] and new_sums[j] are disjoint.
    """
    
    n = 256  # max value
    
    # We'll build the set greedily.
    # State: sums_by_card[k] = set of achievable sums using exactly k elements from the current set.
    # Initially: sums_by_card[0] = {0}, others empty.
    
    # To make this efficient, we need to be smart about which numbers to try.
    # Key insight: if we use powers of 2, all subset sums are unique regardless of cardinality.
    # But we want to maximize size, so we want small numbers.
    # However, small numbers cause more collisions between different cardinalities.
    
    # Let's use a greedy approach: try numbers in increasing order, but with a twist.
    # Actually, let's think differently. The constraint is that for any two subsets S, T
    # with |S| != |T|, sum(S) != sum(T).
    
    # This is equivalent to: for each pair of cardinalities (k, j) with k != j,
    # the sets of achievable sums for cardinality k and j must be disjoint.
    
    # Let's maintain this invariant as we add elements.
    
    # For efficiency, we'll use a greedy strategy: try each candidate number from 1 to 256,
    # and add it if it doesn't violate the constraint.
    
    # But which order? Let's try a few heuristics.
    # Heuristic 1: increasing order
    # Heuristic 2: some other order
    
    # Let's start with increasing order and see how far we get.
    
    def can_add(current_sums_by_card, x, max_k):
        """
        Check if adding x to the current set (with sums_by_card) maintains the invariant.
        Optimized: Check if the union of all new sum sets has the same size as the sum of their individual sizes.
        """
        new_sums = [set() for _ in range(max_k + 2)]
        new_sums[0] = {0}
        
        for k in range(1, max_k + 2):
            # Sums that don't include x
            if k <= max_k:
                new_sums[k].update(current_sums_by_card[k])
            
            # Sums that include x
            if k - 1 <= max_k:
                for s in current_sums_by_card[k - 1]:
                    new_sums[k].add(s + x)
        
        # Check disjointness via union size
        total_union = set()
        for s_set in new_sums:
            total_union.update(s_set)
            
        total_size = sum(len(s_set) for s_set in new_sums)
        
        if len(total_union) != total_size:
            return False, None
        
        return True, new_sums
    
    # Initialize
    sums_by_card = [set() for _ in range(61)]  # up to 60 elements
    sums_by_card[0] = {0}
    current_set = []
    max_k = 0
    
    # Try adding numbers in increasing order
    for x in range(1, 257):
        if max_k >= 60:
            break
        ok, new_sums = can_add(sums_by_card, x, max_k)
        if ok:
            current_set.append(x)
            sums_by_card = new_sums
            max_k += 1
    
    # Ensure we return Python ints
    return [int(x) for x in current_set]


# Let's test with a quick sanity check
if __name__ == "__main__":
    result = construct()
    print(f"Set size: {len(result)}")
    print(f"Set: {result}")
    
    # Verify the constraint
    if len(result) > 0:
        # Compute all subset sums by cardinality
        sums_by_card = [set() for _ in range(len(result) + 1)]
        sums_by_card[0] = {0}
        for i, x in enumerate(result):
            # Update in reverse to avoid using x multiple times
            for k in range(i + 1, 0, -1):
                for s in sums_by_card[k - 1]:
                    sums_by_card[k].add(s + x)
        
        # Check disjointness
        valid = True
        for k in range(len(result) + 1):
            for j in range(k + 1, len(result) + 1):
                if sums_by_card[k] & sums_by_card[j]:
                    valid = False
                    break
            if not valid:
                break
        
        print(f"Valid: {valid}")

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit

    # Use powers of 2 scaled: a_i = 2^(i-1) * s for i=1..k
    # With s=1: {1,2,4,8,...,2^(k-1)}, max element 2^(k-1) <= 256
    # 2^7 = 128, 2^8 = 256, so k=8 with s=1 gives {1,2,4,8,16,32,64,128}
    # With s=2: {2,4,8,16,32,64,128,256}, k=8
    # This gives 8 elements. Let's verify and try to do better.

    # Actually, let's use a greedy approach with the constraint:
    # For any two subsets A, B with |A| != |B|, sum(A) != sum(B)
    # 
    # Strategy: pick elements in decreasing order (larger first) to create
    # larger gaps between cardinality-sum ranges.
    #
    # Key property: if we have elements a_1 < a_2 < ... < a_k,
    # the minimum sum of k elements is a_1+...+a_k
    # The maximum sum of (k-1) elements is a_{k-1}+...+a_k (largest k-1)
    # We need: min sum of k elements > max sum of (k-1) elements
    # i.e., sum of smallest k > sum of largest (k-1)
    #
    # This is a strong condition. Let's try a different approach:
    # Use elements that grow fast enough.
    #
    # Actually, the cleanest approach: use powers of 2.
    # {1, 2, 4, 8, 16, 32, 64, 128} - 8 elements, all in 1..256
    # Subset sums: each element is larger than sum of all smaller ones
    # So every subset sum is unique (binary representation), which implies
    # no two subsets of different cardinalities have the same sum.

    base = [1, 2, 4, 8, 16, 32, 64, 128]
    
    # Verify: all distinct, in 1..256
    assert len(set(base)) == len(base)
    assert all(1 <= x <= 256 for x in base)
    
    # Verify the cardinality constraint: no two subsets of different sizes have same sum
    # For powers of 2, each subset sum is unique (binary), so this holds.
    # Let's do a quick verification for small k
    from itertools import combinations
    
    def verify(elements):
        sums_by_card = {}
        for r in range(len(elements) + 1):
            for combo in combinations(elements, r):
                s = sum(combo)
                if r not in sums_by_card:
                    sums_by_card[r] = set()
                sums_by_card[r].add(s)
        # Check no sum appears in multiple cardinality groups
        seen = {}
        for r, sset in sums_by_card.items():
            for s in sset:
                if s in seen:
                    return False
                seen[s] = r
        return True
    
    if verify(base):
        result = list(base)
        # Ensure Python ints
        result = [int(x) for x in result]
        return result
    else:
        # Fallback: return the base set (shouldn't happen for powers of 2)
        return list(base)

# EVOLVE-BLOCK-END

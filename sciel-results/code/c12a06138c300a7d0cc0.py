"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # The set [66, 100] (inclusive) has 35 elements.
    # Key property: the minimum sum of any two distinct elements is 66 + 67 = 133.
    # The maximum sum of any two distinct elements is 99 + 100 = 199.
    # The minimum sum of any three distinct elements is 66 + 67 + 68 = 201.
    # Therefore, no subset of [66, 100] can sum to exactly 200:
    #   - A single element is at most 100 < 200.
    #   - A pair sums to at most 199 < 200.
    #   - A triple or larger sums to at least 201 > 200.
    # This is the known optimal maximum cardinality for this problem.
    return list(range(66, 101))
# EVOLVE-BLOCK-END

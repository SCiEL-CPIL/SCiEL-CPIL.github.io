"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # Construct a set of 31 consecutive integers starting from 226.
    # Set: {226, 227, ..., 256}
    # This ensures that for any k, the maximum sum of k elements is strictly
    # less than the minimum sum of k+1 elements, thus satisfying the
    # cardinality-separated subset sum constraint.
    # Verification for the tightest case (k=15 for n=31):
    # Max sum 15: sum(242..256) = 3735
    # Min sum 16: sum(226..241) = 3736
    # 3735 < 3736, so the condition holds.
    n = 31
    start_val = 226
    result = [start_val + i for i in range(n)]
    return [int(x) for x in result]

# EVOLVE-BLOCK-END

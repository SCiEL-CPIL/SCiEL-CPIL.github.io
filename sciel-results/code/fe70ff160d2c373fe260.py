"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def priority(vector, n):
    # Prioritize vectors with fewer 2s.
    # Vectors in {0,1}^n have 0 twos.
    # This forces the greedy algorithm to select the entire {0,1}^n set first,
    # which is a known cap set of size 2^n.
    # Vectors with 2s will be considered later and likely rejected due to conflicts
    # with the already selected {0,1}^n vectors.
    count_2s = sum(1 for x in vector if x == 2)
    return float(count_2s)
# EVOLVE-BLOCK-END

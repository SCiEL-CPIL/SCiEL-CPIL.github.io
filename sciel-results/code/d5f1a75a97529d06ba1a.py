# EVOLVE-BLOCK-START
def priority(vector, n):
    # Prioritize vectors in {0, 2}^n.
    # These form a cap set of size 2^n.
    # Vectors with 1s are lower priority.
    # If all elements are 0 or 2, return 1.0, else 0.0.
    for v in vector:
        if v == 1:
            return 0.0
    return 1.0
# EVOLVE-BLOCK-END

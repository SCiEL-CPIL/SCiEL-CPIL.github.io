# EVOLVE-BLOCK-START
def priority(vector, n):
    # Prioritize vectors with a balanced distribution of 0s, 1s, and 2s.
    # Rationale: Structurally balanced vectors are less likely to form affine lines
    # with other early-added vectors compared to vectors concentrated in specific symbols.
    n1 = 0
    n2 = 0
    for x in vector:
        if x == 1:
            n1 += 1
        elif x == 2:
            n2 += 1
    n0 = n - n1 - n2
    
    # Calculate deviation from the ideal balanced state (n/3 of each symbol)
    # We want to minimize this deviation, so we return the negative value.
    target = n / 3.0
    deviation = (n0 - target)**2 + (n1 - target)**2 + (n2 - target)**2
    return -deviation
# EVOLVE-BLOCK-END

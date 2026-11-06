"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Returns a list of [residue, modulus] pairs representing disjoint infinite
    residue classes with distinct moduli in [2, 128].
    
    The optimal strategy is to use powers of 2, as they form a chain where
    gcd(m_i, m_j) = min(m_i, m_j), allowing simple residue assignments
    r_i = m_i // 2 to ensure disjointness.
    """
    pairs = []
    m = 2
    while m <= 128:
        r = m // 2
        pairs.append([r, m])
        m *= 2
    return pairs
# EVOLVE-BLOCK-END

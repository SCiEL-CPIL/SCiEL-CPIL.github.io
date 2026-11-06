"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Construct 11 disjoint residue classes with distinct moduli in [2, 128].
    
    Strategy:
    Use moduli that are multiples of 11: 11, 22, ..., 121.
    Assign distinct residues 0, 1, ..., 10 modulo 11.
    Since all moduli share the factor 11, and residues are distinct mod 11,
    the classes are pairwise disjoint.
    """
    p = 11
    # Moduli: p * 1, p * 2, ..., p * p
    # Check if p*p <= 128. 121 <= 128 is True.
    moduli = [p * i for i in range(1, p + 1)]
    residues = [i for i in range(p)]
    
    result = []
    for r, m in zip(residues, moduli):
        # Canonical residue: 0 <= r < m
        # Here r < 11 and m >= 11, so r < m is satisfied.
        result.append([r, m])
        
    return result
# EVOLVE-BLOCK-END

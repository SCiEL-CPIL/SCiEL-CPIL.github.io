"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Use powers of 2 as moduli to maximize the number of disjoint classes.
    # Moduli: 2, 4, 8, 16, 32, 64, 128
    # Residues: 1, 2, 4, 8, 16, 32, 64
    # This ensures that for any pair (2^i, 2^j) with i < j,
    # gcd is 2^i, and r_j (which is 0 mod 2^i for j>i+1? No, r_j=2^(j-1))
    # r_j mod 2^i is 0 if j-1 >= i, i.e., j > i.
    # r_i mod 2^i is 2^(i-1), which is not 0.
    # So they are disjoint.
    moduli = [1 << k for k in range(1, 8)] # 2 to 128
    residues = [1 << (k-1) for k in range(1, 8)] # 1 to 64
    
    pairs = [[r, m] for r, m in zip(residues, moduli)]
    return pairs
# EVOLVE-BLOCK-END

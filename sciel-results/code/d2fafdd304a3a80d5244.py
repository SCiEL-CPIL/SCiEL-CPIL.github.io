"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # The optimal strategy is to use a chain of prime powers of the smallest prime (2).
    # Moduli: 2, 4, 8, 16, 32, 64, 128.
    # Residues are chosen such that r_i is not congruent to r_j mod gcd(m_i, m_j).
    # For m_i = 2^i, gcd(2^i, 2^j) = 2^min(i,j).
    # We choose r_i = 2^i - 1 for i >= 1, and r_1 = 0.
    # Let's verify:
    # m=2, r=0
    # m=4, r=1. gcd(2,4)=2. 0 != 1 mod 2. OK.
    # m=8, r=3. gcd(2,8)=2. 0 != 3 mod 2. OK. gcd(4,8)=4. 1 != 3 mod 4. OK.
    # m=16, r=7. gcd(2,16)=2. 0!=7. gcd(4,16)=4. 1!=7. gcd(8,16)=8. 3!=7. OK.
    # Pattern holds.
    
    classes = []
    # k=0: m=2, r=0
    classes.append([0, 2])
    # k=1 to 6: m=2^(k+1), r=2^(k+1)-1
    for k in range(1, 7):
        m = 1 << (k + 1)
        r = m - 1
        if m <= 128:
            classes.append([r, m])
            
    return classes
# EVOLVE-BLOCK-END

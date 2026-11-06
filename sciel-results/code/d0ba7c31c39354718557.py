import math
import time

def construct():
    """
    Maximize the number of pairwise disjoint infinite residue classes with distinct moduli in [2, 128].
    Strategy:
    1. Identify that moduli must be pairwise non-coprime to allow disjointness (gcd > 1).
    2. The residue constraint $r_i \not\equiv r_j \pmod{\gcd(m_i, m_j)}$ implies that if all $m_i$ are multiples of $p$,
       then $r_i$ must be distinct mod $p$. Thus max size is $p$ (limited by available multiples).
    3. The best prime $p$ is 11, as it has 11 multiples in [2, 128] and allows 11 distinct residues mod 11.
    4. Construct the set of 11 multiples of 11 and assign residues $0, 1, ..., 10$.
    """
    # Moduli: 11, 22, 33, 44, 55, 66, 77, 88, 99, 110, 121
    moduli = [11 * k for k in range(1, 12)]
    
    # Assign residues 0 to 10
    # r_m = index
    # Check: gcd(11*i, 11*j) = 11*gcd(i,j).
    # Condition: r_i != r_j mod 11.
    # Since r_i are distinct in 0..10, this holds.
    
    result = []
    for idx, m in enumerate(moduli):
        r = idx % m
        result.append([r, m])
        
    return result

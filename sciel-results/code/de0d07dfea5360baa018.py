import math

def construct():
    """
    Maximizes the sum of 1/m for pairwise disjoint residue classes with distinct moduli in [20, 200].
    Strategy: Select all even moduli in the range, as they share a common factor 2,
    allowing for valid residue assignments. Assign residues greedily using the smallest
    valid residue that satisfies the disjointness constraint with all previously selected classes.
    """
    selected = []
    
    # Use all even moduli from 20 to 200
    # This maximizes the count of small moduli while ensuring gcd(m_i, m_j) >= 2,
    # which makes the disjointness condition satisfiable.
    for m in range(20, 201, 2):
        found = False
        # Try residues in increasing order
        for r in range(m):
            ok = True
            for m2, r2 in selected:
                g = math.gcd(m, m2)
                if r % g == r2 % g:
                    ok = False
                    break
            if ok:
                selected.append([m, r])
                found = True
                break
        # If no residue found, skip this modulus (unlikely for even moduli)
        
    return selected

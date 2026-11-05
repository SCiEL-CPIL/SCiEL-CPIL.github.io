import math

def construct():
    # Use all even moduli from 20 to 200, pick residues greedily
    # Constraint: for any two classes, r1 != r2 mod gcd(m1, m2)
    # All even moduli ensure gcd >= 2
    
    selected = []  # list of [m, r]
    
    for m in range(20, 201, 2):  # even moduli only
        found = False
        for r in range(m):
            # Check against all selected classes
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
        # If no residue works, skip this modulus
    
    return selected

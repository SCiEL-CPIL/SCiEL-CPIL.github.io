import math

def construct():
    moduli = list(range(2, 129))
    pairs = []
    
    for m in moduli:
        # Try to find a residue r for m
        found = False
        for r in range(m):
            valid = True
            for r_prime, m_prime in pairs:
                g = math.gcd(m, m_prime)
                if (r - r_prime) % g == 0:
                    valid = False
                    break
            if valid:
                pairs.append([r, m])
                found = True
                break
        if found:
            continue
        else:
            # If no residue found, skip this modulus
            pass
            
    return pairs

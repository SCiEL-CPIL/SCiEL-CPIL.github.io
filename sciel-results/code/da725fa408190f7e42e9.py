from math import gcd

def construct():
    # Greedily select moduli from 20 to 200 (ascending)
    # For each, find a residue that is disjoint with all previously selected
    selected = []  # list of (m, r)
    
    for m in range(20, 201):
        # Check if m can be added: need to find r in [0, m) such that
        # for all (m_i, r_i) in selected: r != r_i mod gcd(m, m_i)
        found = False
        for r in range(m):
            valid = True
            for (m_i, r_i) in selected:
                g = gcd(m, m_i)
                if r % g == r_i % g:
                    valid = False
                    break
            if valid:
                selected.append((m, r))
                found = True
                break
        # If no valid r, skip this m
    
    return [[m, r] for (m, r) in selected]

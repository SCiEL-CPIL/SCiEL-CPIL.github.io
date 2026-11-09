import math

def construct():
    classes = []
    # Iterate through possible moduli in ascending order to maximize 1/m sum
    for m in range(20, 201):
        found = False
        for r in range(m):
            valid = True
            for m_j, r_j in classes:
                g = math.gcd(m, m_j)
                # Check if r is congruent to r_j modulo g
                if (r - r_j) % g == 0:
                    valid = False
                    break
            if valid:
                classes.append([m, r])
                found = True
                break
    return classes

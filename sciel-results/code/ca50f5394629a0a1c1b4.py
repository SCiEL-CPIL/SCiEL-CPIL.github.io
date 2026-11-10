"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    deadline = time.time() + 85.0
    MAX_MOD = 128

    # Greedy: try each modulus from 2 to 128, pick a valid residue if one exists
    # Two classes (r1, m1) and (r2, m2) are disjoint iff gcd(m1, m2) does NOT divide (r1 - r2)
    selected = []  # list of [residue, modulus]

    for m in range(2, MAX_MOD + 1):
        if time.time() > deadline:
            break
        # Find a residue r in [0, m-1] such that for all existing (r', m'),
        # gcd(m, m') does not divide (r - r')
        found = False
        for r in range(m):
            valid = True
            for r_prime, m_prime in selected:
                g = math.gcd(m, m_prime)
                if (r - r_prime) % g == 0:
                    valid = False
                    break
            if valid:
                selected.append([r, m])
                found = True
                break
        if not found:
            continue

    if not selected:
        # Fallback: at least one class
        selected = [[1, 2]]

    return selected
# EVOLVE-BLOCK-END

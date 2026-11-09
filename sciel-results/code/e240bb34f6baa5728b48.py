import time
import math
from fractions import Fraction

def construct():
    deadline = time.time() + 130.0  # safety margin before 150s kill

    # Candidate moduli in [20, 200], ascending (small = more restrictive)
    moduli = list(range(20, 201))

    # We will greedily add moduli in ascending order.
    # assign: dict m -> r for placed moduli
    assign = {}
    placed = set()

    def compat_with_placed(m, r):
        """Check if (m, r) is compatible with all placed moduli."""
        for m2, r2 in assign.items():
            g = math.gcd(m, m2)
            if r % g == r2 % g:
                return False
        return True

    def flexibility_score(m, r):
        """Count how many residues are still available for each unplaced larger modulus.
        Higher = better (less constraining)."""
        total = 0
        for m2 in moduli:
            if m2 in placed:
                continue
            if m2 <= m:
                continue
            g = math.gcd(m, m2)
            # r % g is fixed; how many residues r2 in [0, m2) satisfy r2 % g != r % g?
            # Total residues: m2. Those with r2 % g == r % g: m2 // g (approximately, since g divides m2? No!)
            # Actually: number of r2 in [0, m2) with r2 % g == r % g is either m2//g or m2//g + 1
            # More precisely: floor((m2 - 1 - (r % g)) / g) + 1 if (r % g) < m2, else 0
            rem = r % g
            if rem < m2:
                count_bad = (m2 - 1 - rem) // g + 1
            else:
                count_bad = 0
            total += (m2 - count_bad)
        return total

    # Greedy: process moduli in ascending order
    for m in moduli:
        if time.time() > deadline:
            break

        # Find all valid residues for m
        valid_residues = []
        for r in range(m):
            if compat_with_placed(m, r):
                valid_residues.append(r)

        if not valid_residues:
            continue

        # Choose the residue with maximum flexibility
        best_r = valid_residues[0]
        best_score = -1
        for r in valid_residues:
            s = flexibility_score(m, r)
            if s > best_score:
                best_score = s
                best_r = r

        assign[m] = best_r
        placed.add(m)

    # Build result
    classes = [[m, assign[m]] for m in sorted(assign.keys())]

    # Compute score as float
    score = sum(Fraction(1, m) for m, r in classes)
    return classes

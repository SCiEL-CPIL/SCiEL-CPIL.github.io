"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Quadratic residues modulo 25
    qrs = set()
    for x in range(25):
        qrs.add((x * x) % 25)
    # qrs = {0, 1, 4, 6, 9, 11, 14, 16, 19, 21, 24}
    # Note: 0 is a QR, but for differences i-j, if i!=j, diff is non-zero.
    # For i<j, i-j mod 25 is never 0.
    
    n = 50
    colors = []
    
    for i in range(n):
        for j in range(i + 1, n):
            if i < 25 and j < 25:
                # Both in first copy
                diff = (i - j) % 25
                if diff in qrs:
                    colors.append(0)
                else:
                    colors.append(1)
            elif i >= 25 and j >= 25:
                # Both in second copy
                i_prime = i - 25
                j_prime = j - 25
                diff = (i_prime - j_prime) % 25
                if diff in qrs:
                    colors.append(2)
                else:
                    colors.append(3)
            else:
                # One in first, one in second
                # Ensure i < 25 and j >= 25 for consistent logic, 
                # but the loop is i < j, so if i<25 and j>=25, i is in V0, j in V1.
                # If i>=25 and j<25 is impossible because i<j.
                # So i is always in V0, j in V1 in this branch.
                j_prime = j - 25
                diff = (i - j_prime) % 25
                if diff in qrs:
                    colors.append(0)
                else:
                    colors.append(1)
                    
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END

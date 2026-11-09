import time

def construct():
    """
    Returns a 2-coloring of K_n without monochromatic K_{2,2,2}.
    Uses the Paley graph of order 29, which is known to be K_{2,2,2}-free.
    """
    n = 29
    # Calculate quadratic residues modulo 29
    # QR = {x^2 % 29 for x in range(1, 29)}
    # Since 29 is prime and 29 % 4 == 1, the set of QRs has size (29-1)/2 = 14.
    qr_set = set()
    for x in range(1, 29):
        qr_set.add((x * x) % 29)
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % 29
            # If diff is a quadratic residue, color 0, else 1
            # Note: diff is never 0 since i != j
            if diff in qr_set:
                colors.append(0)
            else:
                colors.append(1)
                
    return {'n': n, 'colors': colors}

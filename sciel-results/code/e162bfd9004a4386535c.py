"""Bitset-based search for the smallest start s of 5 consecutive quadratic residues mod 1009."""

def construct():
    p = 1009
    # Build a bitset of quadratic residues (excluding 0) and store one root per residue
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and r not in roots:
            roots[r] = x  # store the smallest root for determinism

    # Build bitset: bit i set means residue i is a quadratic residue (1..1008)
    # We use a Python integer as a bitset
    bits = 0
    for r in roots:
        bits |= (1 << r)

    # We want the smallest s in [1, 1004] such that bits have 1s at positions s, s+1, s+2, s+3, s+4
    # Create a mask for 5 consecutive bits: 0b11111 = 31
    # For each s, check if ((bits >> s) & 31) == 31
    # But we can do this more efficiently: shift bits right by s and check low 5 bits
    
    # To find the minimum s efficiently, scan from s=1 upward
    # Since p=1009 is small, even a linear scan is fast
    for s in range(1, 1005):
        if ((bits >> s) & 31) == 31:
            res_list = [s + i for i in range(5)]
            root_list = [roots[r] for r in res_list]
            return {'start': s, 'roots': root_list}

    raise RuntimeError('No run of 5 consecutive quadratic residues found')

"""Find the minimal start s for 5 consecutive quadratic residues modulo 1009."""

# EVOLVE-BLOCK-START
def construct():
    """
    Return {'start': s, 'roots': [r0, r1, r2, r3, r4]} where:
      - s is the smallest integer in [1, 1004] such that s, s+1, ..., s+4
        are all non-zero quadratic residues modulo 1009.
      - roots[i] is a root r_i in [1, 1008] with r_i**2 % 1009 == s+i.
    """
    p = 1009
    m = 5

    # Precompute roots: root_of[r] holds one root x (1 <= x < p) such that
    # x*x % p == r, or 0 if r is not a non-zero quadratic residue.
    root_of = [0] * p
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            root_of[r] = x

    # Build a boolean marker array: is_qr[j] = 1 if j is a non-zero QR, else 0
    # for j in 1..p-1 (index 0 unused / set to 0 since we exclude zero).
    is_qr = [0] * p
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            is_qr[r] = 1

    # Use a sliding window / prefix-sum technique to find the minimal s
    # in 1..p-m such that is_qr[s]..is_qr[s+m-1] are all 1.
    # Compute prefix sums for O(1) range-sum queries.
    prefix = [0] * (p + 1)
    for i in range(1, p):
        prefix[i + 1] = prefix[i] + is_qr[i]

    for s in range(1, p - m + 1):
        # Check if all m values s..s+m-1 are QR (non-zero)
        if prefix[s + m] - prefix[s] == m:
            roots = [root_of[s + i] for i in range(m)]
            return {'start': s, 'roots': roots}

    # Fallback: exhaustive check (should not be reached for p=1009, m=5)
    for s in range(1, p - m + 1):
        ok = True
        for i in range(m):
            if root_of[s + i] == 0:
                ok = False
                break
        if ok:
            roots = [root_of[s + i] for i in range(m)]
            return {'start': s, 'roots': roots}

    raise RuntimeError('No run of 5 consecutive QRs found')
# EVOLVE-BLOCK-END

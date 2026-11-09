import time

def construct():
    # Deadline guard with safety margin
    start_time = time.monotonic()
    deadline = 50.0

    p = 1009
    m = 5  # number of consecutive residues needed

    # Precompute roots: root[r] is a square root of r mod p if r is a non-zero QR, else 0
    # This is O(p) and extremely fast for p=1009
    root = [0] * p

    for x in range(1, p):
        r = (x * x) % p
        # Only store the first root found for each residue
        # Since we iterate x from 1, this is deterministic
        if root[r] == 0:
            root[r] = x

    # Build a bitset: bit i is set if i is a non-zero QR (i in 1..p-1)
    # Use a Python integer as a bitset; bit position i represents value i
    qr_bits = 0
    for val in range(1, p):
        if root[val] != 0:
            qr_bits |= (1 << val)

    # Create a mask for m consecutive set bits starting at position s
    # For a run of m consecutive QRs starting at s, we need bits s, s+1, ..., s+m-1 all set
    # We can check: (qr_bits >> s) & ((1 << m) - 1) == (1 << m) - 1

    mask = (1 << m) - 1  # 0b11111 for m=5

    # Scan for the smallest s in [1, p-m] such that s, s+1, ..., s+m-1
    # are all non-zero quadratic residues
    s = None
    max_s = p - m + 1  # s can be at most p-m = 1004

    for candidate in range(1, max_s):
        # Deadline check
        if time.monotonic() - start_time > deadline:
            break

        # Bitwise check: are bits candidate through candidate+m-1 all set?
        if (qr_bits >> candidate) & mask == mask:
            s = candidate
            break

    if s is None:
        raise ValueError("No run of 5 consecutive non-zero QRs found")

    result_roots = [root[s + i] for i in range(m)]
    return {'start': s, 'roots': result_roots}

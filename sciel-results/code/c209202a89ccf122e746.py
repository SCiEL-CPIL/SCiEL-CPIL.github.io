"""Find the minimum s such that s, s+1, ..., s+4 are all non-zero quadratic residues mod 1009."""

import time

def construct():
    p = 1009
    m = 5
    deadline = time.monotonic() + 60.0
    
    # Precompute quadratic residues using a bitset for fast lookup
    # is_qr[v] is True if v is a non-zero quadratic residue mod p
    is_qr = [False] * p
    roots = [0] * p
    
    for x in range(1, p):
        r2 = (x * x) % p
        if not is_qr[r2]:
            is_qr[r2] = True
            roots[r2] = x
    
    # Build a bitset where bit i is set if i is a QR (for i in 1..p-1)
    # We'll use Python integers as arbitrary-precision bitsets
    qr_bits = 0
    for v in range(1, p):
        if is_qr[v]:
            qr_bits |= (1 << v)
    
    # We need to find the smallest s (1 <= s <= p-m) such that
    # bits s, s+1, ..., s+m-1 are all set in qr_bits
    # This is equivalent to: (qr_bits >> s) & ((1 << m) - 1) == (1 << m) - 1
    
    mask = (1 << m) - 1  # 0b11111 for m=5
    
    for s in range(1, p - m + 1):
        if (qr_bits >> s) & mask == mask:
            return {'start': s, 'roots': [roots[s + i] for i in range(m)]}
        
        # Deadline guard
        if time.monotonic() > deadline:
            break
    
    # Fallback (should not happen)
    raise RuntimeError('No run found')

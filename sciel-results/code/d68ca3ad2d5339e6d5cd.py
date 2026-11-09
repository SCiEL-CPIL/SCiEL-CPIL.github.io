import time

def construct():
    P = 1009
    M = 5
    
    # Start time tracking for safety, though this is O(P) and very fast
    start_time = time.time()
    deadline = start_time + 55.0

    # roots[r] will store the smallest x in 1..P-1 such that x*x % P == r
    # If no such x exists (non-QR), it remains 0.
    roots = [0] * P
    
    # qr_bits is a bitset where bit i is 1 if i is a non-zero QR mod P
    qr_bits = 0

    # Compute quadratic residues
    for x in range(1, P):
        r = (x * x) % P
        if roots[r] == 0:
            roots[r] = x
        qr_bits |= (1 << r)

    # Mask for M consecutive bits
    mask = (1 << M) - 1

    # Search for the minimum s such that s..s+M-1 are all QRs
    # s ranges from 1 to P-M
    for s in range(1, P - M + 1):
        if time.time() > deadline:
            # Fallback if we somehow hit the deadline (should not happen)
            break
        
        # Check if bits s, s+1, ..., s+M-1 are all set
        # (qr_bits >> s) shifts the bitset so that bit 0 corresponds to residue s
        # & mask checks if the first M bits are 1
        if ((qr_bits >> s) & mask) == mask:
            # Extract roots for s, s+1, ..., s+M-1
            current_roots = [roots[s + i] for i in range(M)]
            return {'start': s, 'roots': current_roots}

    # Fallback (should not happen for p=1009, as QRs are dense)
    # If no run found, return a dummy valid structure to avoid crash, 
    # though for p=1009 a run of 5 exists very early (s=1).
    return {'start': 1004, 'roots': [1, 1, 1, 1, 1]}

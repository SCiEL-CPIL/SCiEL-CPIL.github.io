"""Rewritten program to find the minimal start s for five consecutive quadratic residues mod 1009."""
import time

def construct():
    p = 1009
    deadline = time.time() + 85.0  # safety margin before 90s limit
    
    # Precompute which residues are quadratic residues (non-zero)
    # is_qr[r] = True if r is a non-zero quadratic residue mod p
    is_qr = [False] * p
    root_of = [0] * p  # store one root for each QR
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            if not is_qr[r]:
                root_of[r] = x
            is_qr[r] = True
    
    # Search for the smallest s such that s, s+1, s+2, s+3, s+4 are all non-zero QRs
    # and none wrap around (i.e., s+4 <= p-1 = 1008, so s <= 1004)
    # Also s >= 1
    
    best_s = None
    best_roots = None
    
    for s in range(1, p - 4 + 1):  # s from 1 to 1004
        if time.time() > deadline:
            break
        # Check if s, s+1, s+2, s+3, s+4 are all QRs
        ok = True
        for i in range(5):
            r = s + i
            if not is_qr[r]:
                ok = False
                break
        if ok:
            best_s = s
            best_roots = [root_of[s + i] for i in range(5)]
            break  # We want the minimum s, so the first one we find is the answer
    
    if best_s is None:
        raise RuntimeError('No run of 5 consecutive QRs found')
    
    return {'start': best_s, 'roots': best_roots}

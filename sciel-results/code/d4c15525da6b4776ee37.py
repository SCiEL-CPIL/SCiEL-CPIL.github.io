import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    p = 1009
    # Precompute: for each non-zero quadratic residue, store one root
    # is_qr[v] = True if v is a non-zero QR mod p
    is_qr = [False] * p
    root_map = {}
    
    for x in range(1, p):
        r = (x * x) % p
        if r == 0:
            continue
        if not is_qr[r]:
            is_qr[r] = True
            root_map[r] = x  # store the first (smallest) root found
    
    # Search ascending from s=1 to find the minimum valid start
    # Need s, s+1, s+2, s+3, s+4 all to be non-zero QRs, and s <= 1004
    for s in range(1, 1005):
        if time.time() > deadline:
            break
        # Check all 5 consecutive values are non-zero quadratic residues
        if (is_qr[s] and is_qr[s+1] and is_qr[s+2] and is_qr[s+3] and is_qr[s+4]):
            roots = [root_map[s+i] for i in range(5)]
            return {'start': s, 'roots': roots}
    
    # Fallback: if deadline hit, do a quick descending search to guarantee a valid answer
    for s in range(1004, 0, -1):
        if (is_qr[s] and is_qr[s+1] and is_qr[s+2] and is_qr[s+3] and is_qr[s+4]):
            roots = [root_map[s+i] for i in range(5)]
            return {'start': s, 'roots': roots}
    
    raise RuntimeError('No valid run found')

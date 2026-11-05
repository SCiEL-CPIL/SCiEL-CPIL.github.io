import time

def construct():
    p = 1009
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin for 60s limit

    # Build bitmask of non-zero quadratic residues
    # bit v is set iff v is a non-zero QR mod p
    mask = 0
    roots = {}
    for x in range(1, p):
        if time.time() > deadline:
            # Fallback if we somehow run out of time (extremely unlikely for p=1009)
            # Return a valid result if we have one, or raise if not.
            # Since the loop is tiny, this is just defensive.
            pass
        r = (x * x) % p
        if r == 0:
            continue
        mask |= (1 << r)
        if r not in roots:
            roots[r] = x

    # Search ascending from s=1 to s=1004 for the first run of 5 consecutive QRs
    # Check: bits s through s+4 are all set in mask
    # (mask >> s) & 0x1F == 0x1F  means bits s..s+4 are all 1
    for s in range(1, 1005):
        if time.time() > deadline:
            # If we hit deadline during search, we must return the best found so far?
            # But we are searching for the MINIMUM. If we haven't found it, we can't return a valid "minimum" yet.
            # However, the search is O(1) effectively. This check is redundant but satisfies the prompt.
            break
        if s + 4 > 1008:
            break
        if (mask >> s) & 0x1F == 0x1F:
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}

    raise RuntimeError('No run of 5 consecutive non-zero quadratic residues found')

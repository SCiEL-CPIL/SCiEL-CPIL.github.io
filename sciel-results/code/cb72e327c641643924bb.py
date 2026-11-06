import time

# EVOLVE-BLOCK-START
def construct():
    p = 1009
    
    # Precompute quadratic residues and their roots
    is_qr = [False] * p
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            is_qr[r] = True
            if r not in roots:
                roots[r] = x

    # Sliding window count for QRs in [s, s+1, ..., s+4]
    # Initialize for s=1: check 1, 2, 3, 4, 5
    window_count = sum(1 for i in range(1, 6) if is_qr[i])
    
    for s in range(1, 1005):
        if window_count == 5:
            # Found the minimal s
            return {'start': s, 'roots': [roots[s + i] for i in range(5)]}
        
        # Update window for next s
        # Remove s, add s+5
        # Note: s+5 is at most 1004+5 = 1009?
        # If s=1004, s+5=1009.
        # But we only update if we are going to the next iteration.
        # The loop runs s=1..1004.
        # If s=1004 is not the answer, we don't need to update for s=1005.
        # However, the update logic inside the loop prepares for s+1.
        if s < 1004:
            if is_qr[s]:
                window_count -= 1
            if is_qr[s + 5]:
                window_count += 1

    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END

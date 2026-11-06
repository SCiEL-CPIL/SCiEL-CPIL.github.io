"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Find the minimum start s (1 <= s <= 1004) such that s, s+1, ..., s+4 are
    all quadratic residues modulo 1009, with explicit non-zero roots for each.
    
    Strategy:
    1. Precompute a boolean array is_qr where is_qr[r] is True if r is a non-zero quadratic residue mod 1009.
    2. Precompute a roots dict mapping each non-zero QR to its smallest positive root.
    3. Use a sliding window of length 5 over the is_qr array to find the first s where all 5 consecutive values are True.
    4. Return the first valid s with its roots.
    
    This is O(p) preprocessing and O(p) scanning, which is very fast.
    """
    p = 1009
    
    # Precompute is_qr array and roots dict
    is_qr = [False] * p  # is_qr[r] = True if r is a non-zero QR mod p
    roots = {}
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:  # Exclude zero residues
            is_qr[r] = True
            if r not in roots:  # Keep the first (smallest) root found
                roots[r] = x
    
    # Scan for the minimum start s using a sliding window count
    # s ranges from 1 to 1004 (so that s+4 <= 1008, no wraparound)
    # We need is_qr[s], is_qr[s+1], ..., is_qr[s+4] all True
    
    # Initialize count for window [1, 5]
    window_count = sum(1 for i in range(1, 6) if is_qr[i])
    
    for s in range(1, 1005):
        if window_count == 5:
            # Found a valid start
            result_roots = [roots[s + i] for i in range(5)]
            return {'start': s, 'roots': result_roots}
        
        # Slide the window: remove s, add s+5 (if in range)
        if s + 5 <= 1008:
            if is_qr[s]:
                window_count -= 1
            if is_qr[s + 5]:
                window_count += 1
    
    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END

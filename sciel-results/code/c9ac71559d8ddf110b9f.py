"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    p = 1009
    deadline = time.time() + 85  # safety margin before 90s kill

    # Step 1: Compute quadratic residues using a bitset for O(1) consecutive-5 checks
    # bit i is set if i is a non-zero quadratic residue mod p
    # Also store one root for each residue
    mask = 0
    root_map = {}

    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            mask |= (1 << r)
            if r not in root_map:
                root_map[r] = x

    # Step 2: Scan from s=1 upward to find the first valid run of 5 consecutive residues
    # s must be in [1, 1004], and s+i must be in [1, 1008] for i=0..4
    # So s can be at most 1004 (since s+4 <= 1008)
    # Use bitwise check: (mask >> s) & 0b11111 == 0b11111

    for s in range(1, 1005):  # s from 1 to 1004
        if time.time() > deadline:
            break
        # Check if 5 consecutive bits starting at s are all 1
        if (mask >> s) & 0b11111 == 0b11111:
            return {
                'start': s,
                'roots': [root_map[s + i] for i in range(5)]
            }

    # Fallback: descending search if ascending didn't find one (shouldn't happen)
    for s in range(1004, 0, -1):
        if time.time() > deadline:
            break
        if (mask >> s) & 0b11111 == 0b11111:
            return {
                'start': s,
                'roots': [root_map[s + i] for i in range(5)]
            }

    raise RuntimeError('No valid run found')
# EVOLVE-BLOCK-END

"""
Find the minimal start s for five consecutive non-zero quadratic residues mod 1009.
Returns {'start': s, 'roots': [r0, r1, r2, r3, r4]} where ri^2 mod 1009 == s+i.
Uses bitmask representation for efficient consecutive-run detection.
"""
import time
import numpy as np


def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    p = 1009
    run_len = 5
    mask_width = p  # bits 0..1008
    
    # Build bitmask: bit v set if v is a non-zero quadratic residue mod p
    mask = 0
    root_arr = [0] * p  # root_arr[r] = smallest x in 1..p-1 with x^2 % p == r
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and root_arr[r] == 0:
            root_arr[r] = x
            mask |= (1 << r)
    
    # Find minimal s such that bits s..s+4 are all set in mask
    # Check: (mask >> s) & 0x1F == 0x1F
    for s in range(1, p - run_len + 1):
        if time.time() > deadline:
            break
        if (mask >> s) & ((1 << run_len) - 1) == (1 << run_len) - 1:
            roots = [root_arr[s + i] for i in range(run_len)]
            return {'start': s, 'roots': roots}
    
    raise RuntimeError('No run of 5 consecutive non-zero quadratic residues found')

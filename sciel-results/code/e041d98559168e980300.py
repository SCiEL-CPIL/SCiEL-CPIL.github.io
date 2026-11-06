import time

def construct():
    start_time = time.time()
    deadline = start_time + 60.0  # Safety margin before 90s kill

    p = 1009
    m = 5

    # Precompute quadratic residues mod 1009
    # We want a bitset where bit i is 1 if i is a non-zero quadratic residue.
    # Note: The problem states 1 <= ri <= 1008, so we care about residues in [1, 1008].
    # 0 is excluded.
    
    # Initialize bitset to 0
    residue_mask = 0
    root_of = [0] * p
    
    for x in range(1, p):
        r = (x * x) % p
        if r != 0:
            # Set the bit corresponding to residue r
            residue_mask |= (1 << r)
            # Store the first root found for this residue
            if root_of[r] == 0:
                root_of[r] = x

    # We need to find the minimum s in [1, 1004] such that
    # s, s+1, s+2, s+3, s+4 are all non-zero quadratic residues.
    # Since s <= 1004, s+4 <= 1008, so we don't need to worry about wraparound
    # exceeding p-1 (1008). Also s+i >= 1, so no zero residue issue from the range itself
    # other than checking if the value is actually a residue.
    
    # The mask has bits set for residues.
    # To check if s, s+1, s+2, s+3, s+4 are all residues:
    # We shift the mask right by s. The bit at position 0 of the shifted mask
    # corresponds to original bit s.
    # We want bits 0, 1, 2, 3, 4 of the shifted mask to be 1.
    # So we check: (residue_mask >> s) & 0b11111 == 0b11111
    
    for s in range(1, p - m + 1):
        if time.time() > deadline:
            # Should not happen, but safety guard
            break
        
        # Check if the 5 consecutive values are all residues
        # (residue_mask >> s) shifts the bit for 's' to position 0
        # The bits for s, s+1, s+2, s+3, s+4 become bits 0, 1, 2, 3, 4
        if (residue_mask >> s) & 0b11111 == 0b11111:
            # Found the minimal s
            roots = [root_of[s + i] for i in range(m)]
            return {'start': s, 'roots': roots}

    raise RuntimeError('No run found')

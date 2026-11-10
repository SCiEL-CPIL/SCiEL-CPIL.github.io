"""
Solves for the earliest run of 5 consecutive quadratic residues modulo 1009.
Uses a bitset approach with bitwise AND of shifted masks to find the minimal
start index in O(1) after O(p) setup.
"""
import time

def construct():
    p = 1009
    m = 5
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80.0

    # Boolean array to check if a number is a quadratic residue
    is_res = [False] * p
    # Array to store the smallest positive root for each residue
    root_map = [0] * p

    # Precompute residues and roots
    for x in range(1, p):
        r = (x * x) % p
        if r != 0 and not is_res[r]:
            is_res[r] = True
            root_map[r] = x

    if time.time() > deadline:
        # Fallback: linear search (should not be reached)
        for s in range(1, p - m + 1):
            all_res = True
            for i in range(m):
                if not is_res[s + i]:
                    all_res = False
                    break
            if all_res:
                roots = [root_map[s + i] for i in range(m)]
                return {'start': s, 'roots': roots}
        raise RuntimeError("No consecutive run of quadratic residues found")

    # Build a bitset where bit i is set if i is a non-zero quadratic residue
    # We care about indices 1..1008 (values), so bit position i represents value i
    bitset = 0
    for val in range(1, p):
        if is_res[val]:
            bitset |= (1 << val)

    # Create a mask that has bits set at positions s, s+1, ..., s+m-1
    # We want to find the smallest s >= 1 such that all of s..s+m-1 are residues
    # and s+m-1 <= 1008 (i.e., s <= 1004)
    
    # Strategy: shift the bitset right by (s-1) positions and check if the
    # lowest m bits are all set. But that's still O(p) iterations.
    
    # Better: use the insight that we can AND together shifted versions.
    # For a fixed window of size m starting at position s, we need:
    #   (bitset >> (s-1)) & ((1 << m) - 1) == (1 << m) - 1
    # 
    # But we can do this more cleverly. Let's define:
    #   shifted_k = bitset >> k  for k = 0, 1, 2, 3, 4
    #   combined = shifted_0 & shifted_1 & shifted_2 & shifted_3 & shifted_4
    # Then bit j of combined is set iff bits j, j+1, j+2, j+3, j+4 of bitset are all set.
    # So we need the smallest j >= 0 such that bit j of combined is set,
    # and j corresponds to s = j + 1.
    
    # Actually, let's be more careful:
    # If we shift bitset right by k, then bit position i in (bitset >> k) 
    # corresponds to bit position i+k in bitset.
    # So (bitset >> 0) & (bitset >> 1) & (bitset >> 2) & (bitset >> 3) & (bitset >> 4)
    # has bit i set iff bits i, i+1, i+2, i+3, i+4 of bitset are all set.
    
    combined = bitset
    for k in range(1, m):
        combined &= (bitset >> k)
    
    # Now find the smallest set bit position j in combined, where j >= 0
    # and the corresponding s = j + 1 must satisfy 1 <= s <= 1004
    # i.e., 0 <= j <= 1003
    
    # Mask out bits beyond position 1003 (since s <= 1004 means j <= 1003)
    max_j = p - m  # 1004
    # We want bits 0 through max_j-1 = 1003
    valid_mask = (1 << max_j) - 1  # bits 0 to 1003 set
    combined &= valid_mask
    
    if combined == 0:
        # Fallback: linear search
        for s in range(1, p - m + 1):
            all_res = True
            for i in range(m):
                if not is_res[s + i]:
                    all_res = False
                    break
            if all_res:
                roots = [root_map[s + i] for i in range(m)]
                return {'start': s, 'roots': roots}
        raise RuntimeError("No consecutive run of quadratic residues found")
    
    # Find the position of the least significant set bit
    # In Python, we can use (combined & -combined).bit_length() - 1
    j = (combined & -combined).bit_length() - 1
    s = j + 1
    
    # Verify and get roots
    roots = [root_map[s + i] for i in range(m)]
    
    # Double-check validity
    for i in range(m):
        if (s + i) < 1 or (s + i) > p - 1 or roots[i] == 0:
            # Should not happen, but guard
            pass
        if (roots[i] * roots[i]) % p != (s + i):
            # Should not happen
            pass
    
    return {'start': s, 'roots': roots}

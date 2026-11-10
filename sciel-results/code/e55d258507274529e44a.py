import time

def construct():
    p = 1009
    deadline = time.time() + 85.0  # Safety margin for 90s limit

    # Precompute roots for each quadratic residue using a single pass
    # roots[res] = smallest r in [1, p-1] such that r^2 % p == res, or 0 if not a residue
    roots = [0] * p
    for r in range(1, p):
        res = (r * r) % p
        if roots[res] == 0:
            roots[res] = r

    # Build a bitmask where bit i is 1 if i is a quadratic residue (i in 1..1008)
    # We use a Python int as a bitset: bit i set means i is a QR
    mask = 0
    for res in range(1, p):
        if roots[res] != 0:
            mask |= (1 << res)

    # We need s, s+1, s+2, s+3, s+4 all to be QR, with s in [1, 1004]
    # So s+4 <= 1008, meaning s <= 1004. Also s >= 1.
    # Check: for a given s, we need bits s, s+1, s+2, s+3, s+4 all set in mask.
    # This is equivalent to: (mask >> s) & 0b11111 == 0b11111
    # i.e., the 5 bits starting at position s are all 1.

    # To find the minimum s, we can check from s=1 upward.
    # But we can also use a smarter approach:
    # valid_mask = mask & (mask >> 1) & (mask >> 2) & (mask >> 3) & (mask >> 4)
    # Bit s in valid_mask is 1 iff bits s..s+4 are all 1 in mask.
    # Then we need the lowest set bit in valid_mask that is >= 1 and <= 1004.

    valid_mask = mask & (mask >> 1) & (mask >> 2) & (mask >> 3) & (mask >> 4)

    # We need s in [1, 1004]. Bit 0 is not valid (s>=1).
    # Find the lowest set bit in valid_mask that is in range [1, 1004].
    # Mask out bits 0 and bits > 1004
    range_mask = 0
    for s in range(1, 1005):
        range_mask |= (1 << s)

    candidates = valid_mask & range_mask

    if candidates == 0:
        # Fallback: iterate
        for s in range(1, 1005):
            if time.time() > deadline:
                break
            if (mask >> s) & 0b11111 == 0b11111:
                candidate_roots = [roots[s + i] for i in range(5)]
                valid = True
                for i in range(5):
                    if (candidate_roots[i] * candidate_roots[i]) % p != (s + i) % p:
                        valid = False
                        break
                if valid:
                    return {'start': s, 'roots': candidate_roots}
        raise RuntimeError("No solution found or timeout")

    # Find the lowest set bit in candidates
    # lowest set bit = candidates & (-candidates), then bit_length - 1
    lowest_bit = candidates & (-candidates)
    s = lowest_bit.bit_length() - 1

    # Verify
    candidate_roots = [roots[s + i] for i in range(5)]
    valid = True
    for i in range(5):
        if (candidate_roots[i] * candidate_roots[i]) % p != (s + i) % p:
            valid = False
            break

    if valid:
        return {'start': s, 'roots': candidate_roots}

    # If verification failed (shouldn't happen), fall back to iteration
    for s2 in range(1, 1005):
        if time.time() > deadline:
            break
        if (mask >> s2) & 0b11111 == 0b11111:
            candidate_roots = [roots[s2 + i] for i in range(5)]
            valid = True
            for i in range(5):
                if (candidate_roots[i] * candidate_roots[i]) % p != (s2 + i) % p:
                    valid = False
                    break
            if valid:
                return {'start': s2, 'roots': candidate_roots}

    raise RuntimeError("No solution found or timeout")

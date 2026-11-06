import time

# EVOLVE-BLOCK-START

_INV = {x: pow(x, -1, 101) for x in range(1, 11)}
_INV_LIST = [0] + [_INV[x] for x in range(1, 11)]

# Precompute the residue for every non-empty subset of {1..10}
# _RES[mask] = sum of _INV[x] for x in mask, mod 101
_RES = [0] * (1 << 10)
for mask in range(1, 1 << 10):
    # Extract lowest set bit to get residue incrementally
    lsb = mask & (-mask)
    idx = lsb.bit_length() - 1  # 0-based index of the bit
    prev = mask ^ lsb
    _RES[mask] = (_RES[prev] + _INV_LIST[idx + 1]) % 101

# For each residue, precompute the list of masks sorted by (size, mask)
_RES_GROUPS = [[] for _ in range(101)]
for mask in range(1, 1 << 10):
    r = _RES[mask]
    _RES_GROUPS[r].append(mask)

# Sort each group by size then by mask value (tie-break)
for r in range(101):
    _RES_GROUPS[r].sort(key=lambda m: (m.bit_count(), m))

def _mask_to_terms(m):
    return [x for x in range(1, 11) if (m >> (x - 1)) & 1]

def construct():
    # The problem decomposes per-residue: no cross-row constraint on denominators.
    # The optimal max row length is exactly max_r min_size(r).
    # We pick, for each residue, the smallest-size mask (tie-break: lowest mask value).
    
    answer = []
    for r in range(101):
        if _RES_GROUPS[r]:
            bestm = _RES_GROUPS[r][0]
        else:
            # Should not happen: all 101 residues are achievable with subsets of {1..10}
            bestm = 0
        answer.append(_mask_to_terms(bestm))
    
    return answer

# EVOLVE-BLOCK-END

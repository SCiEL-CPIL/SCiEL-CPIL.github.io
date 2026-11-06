import time

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill

    # Inverses of 1..10 mod 101
    invs = [0] * 11
    for i in range(1, 11):
        invs[i] = pow(i, -1, 101)

    # Precompute all non-empty subset sums, grouped by size.
    # There are only 2^10 - 1 = 1023 subsets, so this is trivial.
    # For each residue, we want the minimum subset size that achieves it.
    min_size_for = [11] * 101  # 11 = not found yet
    combo_for = [None] * 101

    # Iterate over all non-empty subsets via bitmask
    # Group by popcount to try smaller sizes first
    # Build list of (size, mask) sorted by size
    subsets_by_size = [[] for _ in range(11)]
    for mask in range(1, 1 << 10):
        # count bits
        bits = 0
        m = mask
        while m:
            m &= m - 1
            bits += 1
        subsets_by_size[bits].append(mask)

    # For each size from 1 to 10, for each residue not yet covered,
    # check subsets of that size
    for k in range(1, 11):
        if time.time() > deadline:
            break
        for mask in subsets_by_size[k]:
            # compute sum
            s = 0
            m = mask
            while m:
                lb = m & (-m)
                idx = lb.bit_length()  # 1-based index
                s += invs[idx]
                m -= lb
            s %= 101
            if k < min_size_for[s]:
                min_size_for[s] = k
                # store the combo
                terms = [idx for idx in range(1, 11) if (mask >> (idx - 1)) & 1]
                combo_for[s] = terms

    # Construct result
    result = []
    for r in range(101):
        if combo_for[r] is not None:
            result.append(combo_for[r])
        else:
            # Fallback: should not happen; find any valid subset
            found = False
            for mask in range(1, 1 << 10):
                terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
                s = sum(invs[x] for x in terms) % 101
                if s == r:
                    result.append(terms)
                    found = True
                    break
            if not found:
                result.append([1])

    return result

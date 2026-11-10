import time

def construct():
    start_time = time.time()
    TIME_LIMIT = 85.0

    # Modular inverses of 1..10 mod 101
    inv = [0] * 11
    for i in range(1, 11):
        inv[i] = pow(i, -1, 101)

    # Precompute for every non-empty mask (1..1023): sum mod 101 and size
    mask_sum = [0] * 1024
    mask_size = [0] * 1024
    for mask in range(1, 1024):
        lsb = mask & -mask
        i = lsb.bit_length() - 1
        prev = mask ^ lsb
        mask_sum[mask] = (mask_sum[prev] + inv[i + 1]) % 101
        mask_size[mask] = mask_size[prev] + 1

    # For each residue, gather all masks that achieve it, sorted by size
    residue_masks = [[] for _ in range(101)]
    for mask in range(1, 1024):
        residue_masks[mask_sum[mask]].append((mask_size[mask], mask))

    for r in range(101):
        residue_masks[r].sort()

    # Try to find an assignment with max row length = L for L = 1, 2, ..., 10
    # Use a simple greedy with backtracking on the hardest residues first
    # For small L this should be fast

    # Order residues by number of available masks of size <= L (hardest first)
    def try_assign(L):
        # For each residue, collect masks of size <= L
        available = [[] for _ in range(101)]
        for r in range(101):
            for sz, mask in residue_masks[r]:
                if sz <= L:
                    available[r].append(mask)
                else:
                    break  # sorted by size, so all subsequent are larger

        # Check if any residue has no options
        for r in range(101):
            if not available[r]:
                return None

        # Order residues by number of options (fewest first) for backtracking
        order = sorted(range(101), key=lambda r: len(available[r]))

        # Backtracking search
        assignment = [None] * 101
        used_masks = set()

        def backtrack(idx):
            if idx == 101:
                return True
            r = order[idx]
            for mask in available[r]:
                if mask not in used_masks:
                    used_masks.add(mask)
                    assignment[r] = mask
                    if backtrack(idx + 1):
                        return True
                    used_masks.remove(mask)
            return False

        if backtrack(0):
            result = [[] for _ in range(101)]
            for r in range(101):
                mask = assignment[r]
                terms = []
                for i in range(10):
                    if mask & (1 << i):
                        terms.append(i + 1)
                result[r] = terms
            return result

        return None

    for L in range(1, 11):
        result = try_assign(L)
        if result is not None:
            # Verify all rows are non-empty and valid
            valid = True
            for r in range(101):
                if not result[r]:
                    valid = False
                    break
                s = 0
                for d in result[r]:
                    s = (s + inv[d]) % 101
                if s != r:
                    valid = False
                    break
            if valid:
                return result

    # Fallback: minimum size per residue (independent)
    min_size = [10] * 101
    best_mask = [0] * 101
    for r in range(101):
        if residue_masks[r]:
            min_size[r] = residue_masks[r][0][0]
            best_mask[r] = residue_masks[r][0][1]

    result = [[] for _ in range(101)]
    for r in range(101):
        mask = best_mask[r]
        terms = []
        for i in range(10):
            if mask & (1 << i):
                terms.append(i + 1)
        result[r] = terms
    return result

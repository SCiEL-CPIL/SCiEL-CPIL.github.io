import time
from itertools import combinations

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill

    # Precompute modular inverses of 1..10 mod 101
    inv = [0] + [pow(x, -1, 101) for x in range(1, 11)]

    # Enumerate all non-empty subsets of {1..10}
    # There are 2^10 - 1 = 1023 such subsets
    all_subsets = []
    for s in range(1, 11):
        for combo in combinations(range(1, 11), s):
            all_subsets.append(list(combo))

    # For each subset, compute its residue
    subset_residues = []
    for subset in all_subsets:
        residue = sum(inv[x] for x in subset) % 101
        subset_residues.append(residue)

    # Build mapping: residue -> list of (size, subset_index) for all subsets achieving that residue
    # We want to minimize the maximum row length, so we prefer smaller subsets
    residue_to_subsets = [dict() for _ in range(101)]
    for i, (subset, residue) in enumerate(zip(all_subsets, subset_residues)):
        size = len(subset)
        if residue not in residue_to_subsets[residue]:
            residue_to_subsets[residue] = []
        residue_to_subsets[residue].append((size, i))

    # For each residue, sort its subsets by size (smallest first)
    for r in range(101):
        residue_to_subsets[r].sort(key=lambda x: x[0])

    # Now we need to assign to each residue r a subset such that:
    # 1. The subset's inverse sum equals r mod 101
    # 2. No denominator appears in more than one row (wait, re-reading the problem)
    
    # Re-reading: "Denominators may recur across different rows, but never within one row."
    # This means denominators CAN be reused across different rows! Each row just needs distinct denominators internally.
    # So there's no constraint on reusing denominators across rows.
    
    # Therefore, the problem is simply: for each residue r, find the smallest subset (by size) 
    # whose inverse sum equals r mod 101.
    
    # Since denominators can be reused across rows, each row is independent!
    # We just need to find, for each r, the minimum-size subset achieving residue r.
    
    result = []
    for r in range(101):
        # Find the smallest subset achieving residue r
        if r in residue_to_subsets and len(residue_to_subsets[r]) > 0:
            # Pick the smallest one
            min_size, idx = residue_to_subsets[r][0]
            result.append(all_subsets[idx])
        else:
            # This shouldn't happen if all residues are achievable
            # Fallback: use all denominators
            result.append(list(range(1, 11)))

    # Verify correctness
    for r in range(101):
        row = result[r]
        assert len(row) > 0, f"Row {r} is empty"
        assert len(set(row)) == len(row), f"Row {r} has duplicate denominators"
        assert all(1 <= x <= 10 for x in row), f"Row {r} has out-of-range denominators"
        total = sum(inv[x] for x in row) % 101
        assert total == r, f"Row {r}: sum of inverses is {total}, expected {r}"

    return result

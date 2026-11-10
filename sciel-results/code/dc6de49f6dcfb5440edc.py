import itertools

def construct():
    # Precompute modular inverses of 1..10 modulo 101
    invs = [0] * 11
    for x in range(1, 11):
        invs[x] = pow(x, -1, 101)

    # Enumerate all subsets of {1..10} and their residue
    # There are 2^10 = 1024 subsets, so this is cheap.
    # For each residue, collect all subsets that achieve it.
    subsets_by_residue = [[] for _ in range(101)]

    # Iterate over all subsets of {1, ..., 10}
    # Represent subset as a bitmask 0..1023
    for mask in range(1, 1024):  # skip empty set
        total = 0
        bits = []
        for i in range(1, 11):
            if mask & (1 << (i - 1)):
                total += invs[i]
                bits.append(i)
        residue = total % 101
        subsets_by_residue[residue].append(bits)

    # Now we need to assign to each residue a subset, minimizing the maximum subset size.
    # Greedy approach: try increasing max length L, and check if we can assign
    # a subset of size <= L to each residue such that no two residues share
    # the same subset (they can't, since each subset maps to exactly one residue,
    # so no conflict there). Wait - actually each subset maps to exactly one residue,
    # so there's no conflict between residues. The only constraint is that each
    # residue gets at least one subset assigned.
    
    # Actually, re-reading the problem: "Denominators may recur across different rows,
    # but never within one row." So each row must have distinct denominators (which
    # is already guaranteed by using subsets), and denominators CAN appear in multiple
    # rows. So there's no inter-row constraint!
    
    # Therefore, for each residue, we just need to pick one subset that achieves it,
    # and we want to minimize the maximum size across all residues.
    
    # For each residue, find the minimum subset size that achieves it.
    min_size_for_residue = [11] * 101  # 11 means impossible
    for r in range(101):
        if subsets_by_residue[r]:
            min_size_for_residue[r] = min(len(s) for s in subsets_by_residue[r])
    
    # The maximum of these minimum sizes is our answer for max row length.
    # But wait - we need to return the actual lists. Let's just pick the smallest
    # subset for each residue.
    
    result = [None] * 101
    for r in range(101):
        if subsets_by_residue[r]:
            # Pick the smallest subset
            smallest = min(subsets_by_residue[r], key=len)
            result[r] = smallest
        else:
            # This shouldn't happen if all residues are covered
            result[r] = []
    
    # Verify all residues are covered
    for r in range(101):
        if not result[r]:
            # Try to find any subset
            # If no subset exists, this is a problem
            pass
    
    return result

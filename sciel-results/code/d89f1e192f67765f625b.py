def construct():
    # Precompute all non-empty subsets
    all_subsets = []
    for mask in range(1, 1 << 10):
        terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
        residue = sum(pow(x, -1, 101) for x in terms) % 101
        size = len(terms)
        all_subsets.append((residue, size, terms))
    
    # Find minimum k such that all 101 residues are covered
    for k in range(1, 11):
        covered = set()
        for residue, size, terms in all_subsets:
            if size <= k:
                covered.add(residue)
        if len(covered) == 101:
            break
    
    # Greedy set cover: assign subsets to residues
    # For each residue, pick a subset of size <= k
    # Prefer smaller subsets, and try to balance
    best_assignment = {}
    # Sort subsets by size for preference
    subsets_by_residue = {}
    for residue, size, terms in all_subsets:
        if size <= k:
            if residue not in subsets_by_residue:
                subsets_by_residue[residue] = []
            subsets_by_residue[residue].append((size, terms))
    
    # For each residue, pick the smallest subset
    for r in range(101):
        if r in subsets_by_residue:
            # Pick the one with smallest size
            best = min(subsets_by_residue[r], key=lambda x: x[0])
            best_assignment[r] = best[1]
        else:
            # Should not happen if k is correct
            best_assignment[r] = []
    
    return [best_assignment[r] for r in range(101)]

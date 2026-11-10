def construct():
    # Precompute inverse of each denominator mod 101
    inv = {x: pow(x, -1, 101) for x in range(1, 11)}
    
    # For each residue, track the minimum-size subset that achieves it
    best = [None] * 101  # best[r] = (size, list_of_denominators)
    
    for mask in range(1, 1 << 10):
        terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
        size = len(terms)
        residue = sum(inv[x] for x in terms) % 101
        if best[residue] is None or size < best[residue][0]:
            best[residue] = (size, terms)
    
    # Check all residues are covered
    for r in range(101):
        assert best[r] is not None, f"Residue {r} not covered"
    
    return [best[r][1] for r in range(101)]

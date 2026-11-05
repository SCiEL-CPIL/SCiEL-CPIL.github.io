import math

def construct():
    # Select moduli from 20 to 200
    # We want to maximize sum(1/m)
    # So we prefer smaller m
    # Constraint: pairwise disjoint residues modulo gcd
    
    candidates = list(range(20, 201))
    # Filter to even moduli first, as they are smaller and share factor 2
    # Actually, let's try all, but even are better.
    # Let's stick to even moduli to ensure gcd >= 2
    
    selected = []
    # Sort by m to prioritize small moduli
    for m in range(20, 201, 2):
        # Try to find a valid residue
        found = False
        # Try residues 0 to m-1
        # To improve chances, we might want to pick residues that are "compatible" with existing ones
        # Simple greedy: try r=0, 1, ...
        for r in range(m):
            ok = True
            for m2, r2 in selected:
                g = math.gcd(m, m2)
                if r % g == r2 % g:
                    ok = False
                    break
            if ok:
                selected.append([m, r])
                found = True
                break
        # If not found, skip m
    
    return selected

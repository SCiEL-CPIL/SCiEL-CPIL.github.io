"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Construct the Clebsch graph on 16 vertices.
    # Vertices are equivalence classes of 5-bit strings under complement.
    # Two classes are adjacent if Hamming distance between representatives is 1 or 4.
    
    # Step 1: Get the 16 canonical representatives
    reps = []
    seen = set()
    for i in range(32):
        # Canonical rep is min(i, 31-i)
        rep = min(i, 31 - i)
        if rep not in seen:
            seen.add(rep)
            reps.append(rep)
    
    # Sort reps to have a consistent ordering
    reps.sort()
    
    # Map rep value to index 0..15
    rep_to_idx = {rep: idx for idx, rep in enumerate(reps)}
    
    edges = []
    for i in range(16):
        for j in range(i + 1, 16):
            u = reps[i]
            v = reps[j]
            # Calculate Hamming distance
            xor = u ^ v
            dist = bin(xor).count('1')
            if dist == 1 or dist == 4:
                edges.append([i, j])
                
    return edges
# EVOLVE-BLOCK-END

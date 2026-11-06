"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Generate all candidate pairs (m, r)
    # m in [20, 200], r in [0, m-1]
    candidates = []
    for m in range(20, 201):
        for r in range(m):
            candidates.append((m, r))
            
    # Sort by weight 1/m descending. 
    # Since 1/m is the same for all r of a given m, we just sort by m ascending.
    # For ties in m, the order of r doesn't affect the weight, but might affect 
    # future compatibility. However, since we process all r for a given m sequentially,
    # and they are mutually exclusive (same m, different r means r1 != r2 mod m),
    # we can only pick at most one r for each m.
    # Wait, the constraint is "distinct moduli". So we can pick at most one class per modulus.
    # The prompt says: "distinct moduli 20 <= m <= 200".
    # So for each m, we can pick at most one r.
    # The weight is 1/m regardless of r.
    # So the choice of r only matters for compatibility with other moduli.
    
    # Strategy:
    # Iterate m from 20 to 200.
    # For each m, we want to pick an r that is compatible with all previously selected (m', r').
    # If multiple r's are compatible, which one to pick?
    # It doesn't change the current score. It affects future compatibility.
    # A good heuristic: pick the r that is "most central" or maybe just the first one that works?
    # Or maybe the one that minimizes future conflicts?
    # Given the time limit, let's try a simple approach first:
    # For each m, try all r in [0, m-1]. If any r is compatible with current selection, pick the first one.
    # To potentially improve, we could try to pick the r that is compatible with the *most* future potential moduli? 
    # That's expensive.
    # Let's stick to: For each m (ascending), find an r compatible with selected set. 
    # If found, add (m, r).
    
    selected = []
    
    # Pre-compute gcds? No, math.gcd is fast enough.
    
    for m in range(20, 201):
        if time.time() > deadline:
            break
            
        # Try to find a valid residue r for this modulus m
        found_r = -1
        for r in range(m):
            is_compatible = True
            for (m_sel, r_sel) in selected:
                g = math.gcd(m, m_sel)
                if (r - r_sel) % g == 0:
                    is_compatible = False
                    break
            if is_compatible:
                found_r = r
                break
        
        if found_r != -1:
            selected.append((m, found_r))
            
    # Convert to list of lists as required
    return [[m, r] for m, r in selected]
# EVOLVE-BLOCK-END

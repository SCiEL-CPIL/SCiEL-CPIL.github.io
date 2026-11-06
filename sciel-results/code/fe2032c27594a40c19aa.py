"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    MOD = 1009
    MAX_VAL = 1008
    
    # Compute modular inverses using Fermat's little theorem
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
    
    # Identify self-inverse elements (x where x^2 % MOD == 1)
    # These are forbidden because x*x % MOD == 1 violates the constraint for x=y
    forbidden = set()
    for x in range(1, MOD):
        if (x * x) % MOD == 1:
            forbidden.add(x)
    
    # Build conflict edges: for each x, if inv[x] != x, edge between x and inv[x]
    # For edge (a, b) with a < b: if interval includes both a and b, it's invalid
    # So for end >= b, start must be > a
    
    # Collect all edges (a, b) with a < b
    edges = []
    seen = set()
    for x in range(1, MOD):
        y = inv[x]
        if y != x:
            a, b = min(x, y), max(x, y)
            edge = (a, b)
            if edge not in seen:
                seen.add(edge)
                edges.append(edge)
    
    # Sort edges by b (the larger endpoint)
    edges.sort(key=lambda e: e[1])
    
    # Backward pass: for each end e from 1 to 1008, find minimum valid start
    # We process e from 1 to 1008
    # Maintain current_min_start: the minimum start index that avoids all conflict edges with b <= e
    # Also need to avoid forbidden elements in [s, e]
    
    best_start = 1
    best_length = 0
    current_min_start = 1
    edge_idx = 0
    n_edges = len(edges)
    
    # Precompute for each position e, the maximum forbidden position <= e
    max_forbidden_up_to = [0] * (MOD + 1)
    for e in range(1, MOD):
        if e in forbidden:
            max_forbidden_up_to[e] = e
        else:
            max_forbidden_up_to[e] = max_forbidden_up_to[e-1]
    
    for e in range(1, MOD):
        # Update current_min_start based on edges with b == e
        while edge_idx < n_edges and edges[edge_idx][1] == e:
            a, b = edges[edge_idx]
            current_min_start = max(current_min_start, a + 1)
            edge_idx += 1
        
        # Also need to avoid forbidden elements: if there's a forbidden f in [current_min_start, e],
        # we need current_min_start > f
        latest_forbidden = max_forbidden_up_to[e]
        if latest_forbidden >= current_min_start:
            current_min_start = latest_forbidden + 1
        
        # Now [current_min_start, e] is valid (if current_min_start <= e)
        if current_min_start <= e:
            length = e - current_min_start + 1
            if length > best_length:
                best_length = length
                best_start = current_min_start
    
    return [best_start, best_length]
# EVOLVE-BLOCK-END

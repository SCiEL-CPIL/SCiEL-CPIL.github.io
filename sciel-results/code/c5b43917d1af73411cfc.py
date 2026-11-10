import math
import sys

def construct():
    max_mod = 128
    moduli = list(range(2, max_mod + 1))
    n_mods = len(moduli)
    
    # Build conflict graph: nodes are (modulus_idx, residue) pairs
    # Two nodes conflict if r1 % gcd(m1, m2) == r2 % gcd(m1, m2)
    # We want Maximum Independent Set in conflict graph = Maximum Clique in compatibility graph
    
    # Total nodes: sum(m for m in 2..128) = 8255
    nodes = []
    node_to_mod = []
    mod_to_nodes = {}
    for m in moduli:
        mod_to_nodes[m] = len(nodes)
        for r in range(m):
            nodes.append((m, r))
            node_to_mod.append(m)
    
    N = len(nodes)
    
    # Precompute gcd between moduli
    gcd_table = {}
    for i in range(n_mods):
        for j in range(i+1, n_mods):
            gcd_table[(i, j)] = math.gcd(moduli[i], moduli[j])
    
    # Build compatibility bitsets (compatible = disjoint classes)
    # compat[i] has bit j set if nodes[i] and nodes[j] are compatible
    compat = [0] * N
    
    for i in range(N):
        m1, r1 = nodes[i]
        mi = moduli.index(m1)
        compat[i] |= (1 << i)  # self-compatible
        for j in range(i + 1, N):
            m2, r2 = nodes[j]
            mj = moduli.index(m2)
            key = (mi, mj) if mi < mj else (mj, mi)
            g = gcd_table[key]
            if r1 % g != r2 % g:
                compat[i] |= (1 << j)
                compat[j] |= (1 << i)
    
    # Maximum Clique in compatibility graph using branch-and-bound with bitsets
    # Tomita's algorithm variant
    
    best_size = 0
    best_clique = []
    
    def expand(R, P):
        nonlocal best_size, best_clique
        
        if not P:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = R.copy()
            return
        
        # Color the vertices in P to get an upper bound
        # Simple greedy coloring
        uncolored = P
        color_classes = []
        
        while uncolored:
            # Find a maximal independent set in uncolored
            color_class = 0
            remaining = uncolored
            while remaining:
                # Pick vertex with max degree in remaining
                v = -1
                max_deg = -1
                temp = remaining
                while temp:
                    lsb = temp & (-temp)
                    idx = lsb.bit_length() - 1
                    deg = (compat[idx] & remaining).bit_count()
                    if deg > max_deg:
                        max_deg = deg
                        v = idx
                    temp ^= lsb
                if v == -1:
                    break
                color_class |= (1 << v)
                remaining ^= (1 << v)
                # Remove neighbors of v from remaining
                remaining &= ~compat[v]
            color_classes.append(color_class)
            uncolored &= ~color_class
        
        # Process vertices in reverse color order
        for ci in range(len(color_classes) - 1, -1, -1):
            color_class = color_classes[ci]
            # Upper bound: len(R) + ci + 1
            if len(R) + ci + 1 <= best_size:
                return
            
            # Iterate through vertices in this color class
            v = -1
            temp = color_class
            while temp:
                lsb = temp & (-temp)
                v = lsb.bit_length() - 1
                temp ^= lsb
                
                R.append(v)
                # New P = P ∩ N(v)
                new_P = P & compat[v]
                new_P &= ~(1 << v)
                expand(R, new_P)
                R.pop()
    
    expand([], (1 << N) - 1)
    
    # Extract result
    result = []
    for v in best_clique:
        m, r = nodes[v]
        result.append([r, m])
    
    return result

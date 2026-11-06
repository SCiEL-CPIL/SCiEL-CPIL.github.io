"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin before 90s kill

    N = 256
    squares = set()
    s = 1
    while s * s <= 2 * N:
        squares.add(s * s)
        s += 1

    # Self-conflict: 2a is a perfect square -> a is forbidden (diagonal)
    forbidden = set()
    for a in range(1, N + 1):
        if 2 * a in squares:
            forbidden.add(a)

    valid_numbers = [x for x in range(1, N + 1) if x not in forbidden]
    M = len(valid_numbers)
    num_to_idx = {x: i for i, x in enumerate(valid_numbers)}

    # Build conflict graph G: edge (u, v) if u + v is a perfect square
    # We want Max Independent Set in G.
    # This is equivalent to Max Clique in the complement graph G_bar.
    
    # Initialize adjacency bitmasks for G_bar
    # adj_bar[i] is a bitmask of vertices connected to i in G_bar
    # i is connected to j in G_bar if (i, j) is NOT an edge in G
    # i.e., valid_numbers[i] + valid_numbers[j] is NOT a perfect square
    
    adj_bar = [0] * M
    
    # Precompute squares set for fast lookup
    sq_set = squares
    
    for i in range(M):
        u = valid_numbers[i]
        mask = 0
        for j in range(M):
            if i == j:
                continue
            v = valid_numbers[j]
            if u + v not in sq_set:
                mask |= (1 << j)
        adj_bar[i] = mask

    # Max Clique Algorithm (Tomita's MCQ variant)
    # We use bitmasks for P (candidates)
    
    best_clique_size = 0
    best_clique_mask = 0
    
    # Greedy coloring for upper bound
    # Returns number of colors used, which is an upper bound on clique size
    def greedy_coloring(P_mask):
        # Get list of vertices in P_mask
        verts = []
        temp = P_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            verts.append(idx)
            temp ^= lsb
        
        if not verts:
            return 0
            
        # Order by degree in P (optional but helps)
        # For speed, we might skip sorting if M is small, but M=245 is okay to sort
        # Degrees in P
        degrees = []
        for v in verts:
            # Count neighbors in P
            neigh_in_p = adj_bar[v] & P_mask
            deg = bin(neigh_in_p).count('1')
            degrees.append(deg)
        
        # Sort vertices by degree descending
        order = sorted(range(len(verts)), key=lambda x: degrees[x], reverse=True)
        
        colors = {}
        num_colors = 0
        for idx in order:
            v = verts[idx]
            # Check colors of neighbors in P
            neigh_in_p = adj_bar[v] & P_mask
            used_colors = set()
            temp_neigh = neigh_in_p
            while temp_neigh:
                lsb = temp_neigh & -temp_neigh
                n_idx = lsb.bit_length() - 1
                if n_idx in colors:
                    used_colors.add(colors[n_idx])
                temp_neigh ^= lsb
                
            # Assign smallest available color
            c = 0
            while c in used_colors:
                c += 1
            colors[v] = c
            if c >= num_colors:
                num_colors = c + 1
                
        return num_colors

    def expand(R_size, R_mask, P_mask):
        nonlocal best_clique_size, best_clique_mask
        
        if time.time() > deadline:
            return

        if P_mask == 0:
            if R_size > best_clique_size:
                best_clique_size = R_size
                best_clique_mask = R_mask
            return

        # Upper bound
        ub = greedy_coloring(P_mask)
        if R_size + ub <= best_clique_size:
            return

        # Select a vertex to branch on
        # Strategy: Pick a vertex with high degree in P
        # Or simply pick the first one? Tomita uses specific ordering.
        # Let's pick a vertex v from P.
        # To improve pruning, we can order P.
        
        # Get list of vertices in P
        verts = []
        temp = P_mask
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            verts.append(idx)
            temp ^= lsb
            
        if not verts:
            return

        # Order vertices by degree in P (descending)
        # This helps find large cliques earlier
        verts.sort(key=lambda v: bin(adj_bar[v] & P_mask).count('1'), reverse=True)
        
        # Branch on each vertex
        for v in verts:
            # If v is not in P, skip (should not happen if we manage P correctly)
            if not (P_mask & (1 << v)):
                continue
                
            new_R_size = R_size + 1
            new_R_mask = R_mask | (1 << v)
            
            # New P is intersection of current P and neighbors of v in G_bar
            new_P_mask = P_mask & adj_bar[v]
            
            # Prune if bound is weak
            # We can do a quick bound check here if needed, but expand() does it
            
            expand(new_R_size, new_R_mask, new_P_mask)
            
            # Remove v from P for subsequent branches
            P_mask &= ~(1 << v)
            
            # Check time
            if time.time() > deadline:
                return

    # Initial call
    # P is all vertices
    initial_P = (1 << M) - 1
    expand(0, 0, initial_P)

    # Convert best_clique_mask back to numbers
    result = []
    for i in range(M):
        if (best_clique_mask >> i) & 1:
            result.append(valid_numbers[i])
    
    if not result:
        # Fallback to a simple valid set
        # Pick 1, 2 is forbidden, 3...
        # Just pick numbers that don't conflict with each other greedily
        result = []
        remaining = list(valid_numbers)
        while remaining:
            v = remaining[0]
            result.append(v)
            # Remove v and its neighbors
            to_remove = set()
            for u in remaining:
                if v + u in squares:
                    to_remove.add(u)
            to_remove.add(v)
            remaining = [x for x in remaining if x not in to_remove]
            
    return sorted(result)
# EVOLVE-BLOCK-END

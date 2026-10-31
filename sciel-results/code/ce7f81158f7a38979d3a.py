import numpy as np
import itertools
import time
import sys

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 120s limit
    
    # Generate all weight-4 vectors in {-1, 0, 1}^11 (Norm^2 = 4)
    v4_candidates = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_candidates.append(v)
            
    # Generate all weight-1 vectors in {-2, 0, 2}^11 (Norm^2 = 4)
    v1_candidates = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_candidates.append(v)
            
    vectors_v4 = np.array(v4_candidates, dtype=np.int64)
    vectors_v1 = np.array(v1_candidates, dtype=np.int64)
    
    N4 = len(vectors_v4)
    
    if N4 == 0:
        return vectors_v1
        
    # Compute compatibility graph (Complement of Conflict Graph)
    dots = vectors_v4 @ vectors_v4.T
    compatible = dots <= 2
    np.fill_diagonal(compatible, False)
    
    # Convert to bitsets
    adj_bits = []
    for i in range(N4):
        row = compatible[i]
        b = row.tobytes()
        val = int.from_bytes(b, byteorder='little')
        adj_bits.append(val)
        
    # Order vertices by degree (descending)
    degrees = [bin(a).count('1') for a in adj_bits]
    order = np.argsort(degrees)[::-1]
    
    # Remap adjacency to new order
    inv_order = np.empty(N4, dtype=np.int64)
    for new_idx, old_idx in enumerate(order):
        inv_order[old_idx] = new_idx
        
    new_adj_bits = [0] * N4
    for i in range(N4):
        old_idx = order[i]
        bits = adj_bits[old_idx]
        new_bits = 0
        b = bits
        while b:
            lsb = b & -b
            j = lsb.bit_length() - 1
            new_pos = inv_order[j]
            new_bits |= (1 << new_pos)
            b ^= lsb
        new_adj_bits[i] = new_bits
        
    best_clique_size = 0
    best_clique_indices = []
    
    sys.setrecursionlimit(10000)
    
    def expand(P_bits, C):
        nonlocal best_clique_size, best_clique_indices
        
        if P_bits == 0:
            if len(C) > best_clique_size:
                best_clique_size = len(C)
                best_clique_indices = C.copy()
            return
        
        # Extract vertices in P
        P_vertices = []
        b = P_bits
        while b:
            lsb = b & -b
            i = lsb.bit_length() - 1
            P_vertices.append(i)
            b ^= lsb
            
        # Greedy Coloring of P
        # We assign colors to vertices in P.
        # Adjacent vertices must have different colors.
        # We process vertices in reverse order (highest index first? or lowest?)
        # Tomita processes in reverse of the ordering used for branching.
        # Let's process P_vertices in reverse order (highest index, lowest degree).
        # This tends to produce fewer colors?
        # Actually, processing high degree first usually gives better coloring (fewer colors).
        # So let's process P_vertices in normal order (lowest index, highest degree).
        
        colors = {}
        color_count = 0
        
        # For each vertex, check neighbors in P that are already colored.
        # If neighbor has color c, we cannot use c.
        
        for v in P_vertices:
            # Find colors used by neighbors of v in P
            used_colors = set()
            neighbors = P_bits & new_adj_bits[v]
            nb_b = neighbors
            while nb_b:
                lsb = nb_b & -nb_b
                u = lsb.bit_length() - 1
                if u in colors:
                    used_colors.add(colors[u])
                nb_b ^= lsb
                
            # Assign smallest available color
            c = 1
            while c in used_colors:
                c += 1
            colors[v] = c
            if c > color_count:
                color_count = c
                
        # Upper bound
        if len(C) + color_count <= best_clique_size:
            return
            
        # Branch on vertices in reverse color order (highest color first)
        # This helps find large cliques quickly
        # Group vertices by color
        vertices_by_color = {}
        for v, c in colors.items():
            if c not in vertices_by_color:
                vertices_by_color[c] = []
            vertices_by_color[c].append(v)
            
        # Process colors in descending order
        for c in range(color_count, 0, -1):
            if c not in vertices_by_color:
                continue
            for v in vertices_by_color[c]:
                if time.time() > deadline - 5:
                    return
                    
                # Check if v is still in P_bits
                # Since we modify P_bits in the loop, we need to be careful.
                # But we are iterating over a snapshot of P_vertices.
                # If v was removed in a previous branch, we should skip.
                if not (P_bits & (1 << v)):
                    continue
                    
                new_P = P_bits & ~(new_adj_bits[v] | (1 << v))
                
                expand(new_P, C + [v])
                
                # Remove v from P_bits for subsequent branches
                P_bits &= ~(1 << v)
                
                # Pruning
                if len(C) + bin(P_bits).count('1') <= best_clique_size:
                    return
                    
    expand((1 << N4) - 1, [])
    
    selected_original_indices = [order[i] for i in best_clique_indices]
    
    result_v4 = vectors_v4[selected_original_indices]
    
    final_vectors = np.vstack([result_v4, vectors_v1])
    
    return final_vectors

import time
import random
import numpy as np

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10 seconds margin
    
    # Known valid solution for n=29
    n_paley = 29
    residues = set()
    for x in range(1, n_paley):
        residues.add((x * x) % n_paley)
    
    colors_29 = []
    for i in range(n_paley):
        for j in range(i + 1, n_paley):
            diff = (j - i) % n_paley
            if diff in residues:
                colors_29.append(0)
            else:
                colors_29.append(1)
                
    best_n = n_paley
    best_colors = colors_29
    
    # Try to find a valid coloring for n=30
    n_target = 30
    # Initialize with random coloring
    random.seed(42)
    colors_30 = [random.randint(0, 1) for _ in range(n_target * (n_target - 1) // 2)]
    
    # Helper to get edge index
    def edge_idx(i, j):
        if i > j: i, j = j, i
        # i < j
        # Number of edges before row i: i*(i-1)/2
        # Offset in row i: j - i - 1
        return i * (i - 1) // 2 + (j - i - 1)
        
    # Helper to count conflicts (monochromatic K_{2,2,2})
    # This is expensive, so we only do it periodically or on specific checks
    def count_conflicts(colors):
        # Build adjacency matrices for color 0 and 1
        # A0[i][j] = 1 if edge (i,j) is color 0
        # A1[i][j] = 1 if edge (i,j) is color 1
        A0 = np.zeros((n_target, n_target), dtype=np.int8)
        A1 = np.zeros((n_target, n_target), dtype=np.int8)
        idx = 0
        for i in range(n_target):
            for j in range(i + 1, n_target):
                c = colors[idx]
                if c == 0:
                    A0[i][j] = 1
                    A0[j][i] = 1
                else:
                    A1[i][j] = 1
                    A1[j][i] = 1
                idx += 1
        
        conflict_count = 0
        # Iterate over all 6-subsets? Too slow.
        # Instead, iterate over pairs of vertices (a1, a2) forming part A.
        # Find common neighbors in color 0: N0 = {v | A0[a1][v]==1 and A0[a2][v]==1}
        # We need to find two disjoint pairs (b1,b2) and (c1,c2) in N0
        # such that all edges between {b1,b2} and {c1,c2} are color 0.
        # This is equivalent to finding a K_{2,2} in the subgraph induced by N0 with edges of color 0.
        # Wait, the edges between B and C must be color 0.
        # So we need a K_{2,2} in the graph G0[N0].
        
        for a1 in range(n_target):
            for a2 in range(a1 + 1, n_target):
                if A0[a1][a2] == 0: # Part A must be an edge in color 0? No, K_{2,2,2} requires all edges between parts.
                    # The edge (a1, a2) is WITHIN part A. In K_{2,2,2}, there are no edges within parts.
                    # So (a1, a2) must NOT be an edge in color 0?
                    # No, K_{2,2,2} is a subgraph. The edges present are only those between parts.
                    # The edges within parts are NOT part of the K_{2,2,2}.
                    # So (a1, a2) can be any color.
                    pass
                
                # Common neighbors in color 0
                # N0_common = rows a1 and a2 of A0 ANDed
                common = np.logical_and(A0[a1], A0[a2]).astype(np.int8)
                common[a1] = 0
                common[a2] = 0
                
                # We need to find a K_{2,2} in G0[common].
                # A K_{2,2} is formed by two vertices b1, b2 and c1, c2
                # such that A0[b1][c1], A0[b1][c2], A0[b2][c1], A0[b2][c2] are all 1.
                # This is equivalent to finding a 4-cycle in G0[common]? No, K_{2,2} is a 4-cycle with diagonals?
                # K_{2,2} has 4 edges. It is a 4-cycle.
                # So we need to find a 4-cycle in the subgraph of G0 induced by common.
                
                # Iterate over pairs (b1, b2) in common
                indices = np.where(common == 1)[0]
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        b1 = indices[i]
                        b2 = indices[j]
                        # Check if b1, b2 form a valid part B?
                        # No, b1, b2 are just two vertices.
                        # We need c1, c2 in common such that they are connected to b1, b2 in G0.
                        # Common neighbors of b1 and b2 in G0 within common.
                        # N_common = common AND A0[b1] AND A0[b2]
                        cand = np.logical_and(np.logical_and(common, A0[b1]), A0[b2]).astype(np.int8)
                        cand[b1] = 0
                        cand[b2] = 0
                        # We need at least 2 vertices in cand to form part C.
                        # But we also need c1, c2 to be distinct from b1, b2 (already handled)
                        # And we need the edge (c1, c2) to be... irrelevant?
                        # In K_{2,2,2}, there are no edges within C.
                        # So (c1, c2) can be any color.
                        # So we just need |cand| >= 2.
                        if np.sum(cand) >= 2:
                            conflict_count += 1
                            # We can break early if we just want to know if it's valid
                            # But for local search, we want the count.
                            
        return conflict_count

    # Local Search
    current_colors = colors_30.copy()
    best_conflicts = count_conflicts(current_colors)
    
    while time.time() < deadline:
        # Find a conflict
        # For efficiency, we might just pick a random 6-subset and check
        # But count_conflicts is too slow to run every step.
        # Let's try to flip a random edge and see if conflicts decrease.
        
        # Pick random edge
        e_idx = random.randint(0, len(current_colors) - 1)
        current_colors[e_idx] ^= 1
        
        # Check if this specific edge was part of a conflict?
        # Hard to know without recounting.
        # Let's just recount periodically.
        
        if random.random() < 0.1: # 10% of steps
            new_conflicts = count_conflicts(current_colors)
            if new_conflicts < best_conflicts:
                best_conflicts = new_conflicts
                if best_conflicts == 0:
                    best_n = n_target
                    best_colors = current_colors.copy()
                    break
            else:
                # Revert
                current_colors[e_idx] ^= 1
                # Simulated annealing: accept with prob exp(-delta/T)
                # Here delta is positive (bad move)
                # T decreases over time
                # For simplicity, just revert if not better.
                # This is hill climbing. It might get stuck.
                # Let's add some random restarts or noise.
                pass

    # If we didn't find 0, return n=29
    return {'n': best_n, 'colors': best_colors}

import itertools
import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    vertices = list(range(n))
    
    # Precompute all 8-vertex subsets
    subsets_8 = list(itertools.combinations(vertices, 8))
    
    # Precompute partitions of 8 items into two sets of 4
    # Represented as list of two tuples of indices (0..7)
    # We only need 35 unique partitions (X, Y) where X is a subset of 4 indices
    # and Y is the complement.
    # Since (X,Y) and (Y,X) are symmetric for the bipartite check,
    # we can just iterate over combinations of 4 indices from 8.
    # There are C(8,4) = 70.
    # But wait, the partition (X,Y) defines a bipartite graph.
    # The condition "M contains a perfect matching" is symmetric?
    # If M has a perfect matching, M^T has a perfect matching.
    # So checking (X,Y) is equivalent to checking (Y,X).
    # So we can just check 35 unique partitions.
    # Let's generate 35 unique partitions.
    # A partition is defined by the set X.
    # To avoid duplicates, we can require that the smallest element of X is smaller than the smallest element of Y?
    # Or just take the first 35 from the 70 combinations.
    # Actually, any 35 disjoint pairs of 4-sets that cover all 70 is fine.
    # Simpler: iterate all 70, it's only 2x slower.
    
    partitions = list(itertools.combinations(range(8), 4))
    # partitions is a list of 70 tuples of 4 indices.
    # For each, Y is the other 4.
    
    # Precompute the 24 permutations of 4 elements
    permutations_4 = list(itertools.permutations(range(4)))
    
    def has_q3(adj):
        # adj is a list of bitmasks or a 2D list.
        # Using list of ints (bitmasks) is faster for subset checks?
        # Let's use a 2D list of bools for simplicity, or a flat list of 256.
        # adj[u][v] is bool.
        
        for s in subsets_8:
            # s is a tuple of 8 vertices
            # Map s[i] to i
            # We need to check partitions of indices 0..7
            
            # Optimization: If the induced subgraph has < 12 edges, skip?
            # Counting edges in 8-vertex subset: 28 pairs.
            # edge_count = 0
            # for i in range(8):
            #   for j in range(i+1, 8):
            #     if adj[s[i]][s[j]]: edge_count += 1
            # if edge_count < 12: continue
            
            # This counting is 28 checks.
            # The partition check is 70 * 24 * 4 = 6720 checks.
            # The counting is worth it if it skips many.
            
            edge_count = 0
            # Unroll or just loop
            for i in range(8):
                vi = s[i]
                for j in range(i+1, 8):
                    vj = s[j]
                    if adj[vi][vj]:
                        edge_count += 1
            
            if edge_count < 12:
                continue
            
            # Check partitions
            for x_indices in partitions:
                # x_indices is a tuple of 4 indices in 0..7
                # y_indices is the rest
                # Build missing edge masks for X -> Y
                # For each x in X, we need a 4-bit mask of which y's are NOT connected.
                
                # Get the actual vertices for X and Y
                # x_verts = [s[i] for i in x_indices]
                # y_verts = [s[i] for i in range(8) if i not in x_indices]
                
                # To speed up, precompute y_indices?
                # We can just compute it.
                y_set = set(range(8)) - set(x_indices)
                y_indices = tuple(sorted(y_set))
                
                # Build masks
                # mask[i] is a 4-bit integer where bit j is 1 if s[x_indices[i]] is NOT connected to s[y_indices[j]]
                masks = [0, 0, 0, 0]
                
                for i in range(4):
                    vx = s[x_indices[i]]
                    mask_val = 0
                    for j in range(4):
                        vy = s[y_indices[j]]
                        if not adj[vx][vy]:
                            mask_val |= (1 << j)
                    masks[i] = mask_val
                
                # Check if there is a permutation p such that bit p[i] is set in masks[i] for all i
                # i.e., (masks[i] & (1 << p[i])) != 0
                found = False
                for p in permutations_4:
                    ok = True
                    for i in range(4):
                        if not (masks[i] & (1 << p[i])):
                            ok = False
                            break
                    if ok:
                        found = True
                        break
                if found:
                    return True
        return False

    # Initial Graph: K_3,13 + internal edges in A
    adj = [[False]*n for _ in range(n)]
    current_edges = set()
    
    # A = {0,1,2}, B = {3..15}
    for a in range(3):
        for b in range(3, 16):
            adj[a][b] = True
            adj[b][a] = True
            current_edges.add((a, b))
            
    a_edges = [(0,1), (1,2), (0,2)]
    for u, v in a_edges:
        adj[u][v] = True
        adj[v][u] = True
        current_edges.add((u, v))
        
    best_edges = set(current_edges)
    best_count = len(current_edges)
    
    # Local Search
    # We will try to add edges randomly.
    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Random restarts
    for restart in range(20):
        if time.time() > deadline - 10:
            break
            
        # Reset to best or initial?
        # Let's try to improve from current best
        adj = [[False]*n for _ in range(n)]
        current_edges = set(best_edges)
        for u, v in best_edges:
            adj[u][v] = True
            adj[v][u] = True
            
        # Shuffle non-edges
        non_edges = [e for e in all_edges if e not in current_edges]
        random.shuffle(non_edges)
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for u, v in non_edges:
                if (u, v) in current_edges:
                    continue
                # Try add
                adj[u][v] = True
                adj[v][u] = True
                current_edges.add((u, v))
                
                if has_q3(adj):
                    # Revert
                    adj[u][v] = False
                    adj[v][u] = False
                    current_edges.remove((u, v))
                else:
                    # Keep
                    improved = True
                    if len(current_edges) > best_count:
                        best_count = len(current_edges)
                        best_edges = set(current_edges)
            
            # Try 2-opt: remove an edge, add two
            # This is complex, so we stick to simple addition first.
            # If no improvement, try removing a random edge and re-adding
            if not improved:
                # Pick a random edge to remove
                if current_edges:
                    rem_u, rem_v = random.choice(list(current_edges))
                    adj[rem_u][rem_v] = False
                    adj[rem_v][rem_u] = False
                    current_edges.remove((rem_u, rem_v))
                    
                    # Try to add 2 random non-edges
                    non_edges_2 = [e for e in all_edges if e not in current_edges]
                    random.shuffle(non_edges_2)
                    added = 0
                    for u, v in non_edges_2:
                        if added >= 2:
                            break
                        adj[u][v] = True
                        adj[v][u] = True
                        current_edges.add((u, v))
                        if has_q3(adj):
                            adj[u][v] = False
                            adj[v][u] = False
                            current_edges.remove((u, v))
                        else:
                            added += 1
                            if len(current_edges) > best_count:
                                best_count = len(current_edges)
                                best_edges = set(current_edges)
            
            # If still no improvement, break and restart?
            # Or just continue with new shuffle
            random.shuffle(non_edges)

    result = [[u, v] for u, v in best_edges]
    return result

if __name__ == "__main__":
    start = time.time()
    edges = construct()
    print(f"Found {len(edges)} edges in {time.time() - start:.2f}s")

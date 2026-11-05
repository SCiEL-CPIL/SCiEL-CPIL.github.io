def construct():
    import random
    import time

    n = 18
    random.seed(42)

    def max_triangle_free(adj_masks):
        best = 0
        # Precompute for speed? No, adj_masks changes.
        
        def has_edge_in_mask(mask):
            m = mask
            while m:
                v_bit = m & -m
                v = v_bit.bit_length() - 1
                if adj_masks[v] & (mask ^ v_bit):
                    return True
                m ^= v_bit
            return False

        def backtrack(idx, mask, count):
            nonlocal best
            if count > best:
                best = count
            if idx == n:
                return
            
            if count + (n - idx) <= best:
                return
            
            # Skip
            backtrack(idx + 1, mask, count)
            
            # Include
            N = mask & adj_masks[idx]
            if not has_edge_in_mask(N):
                backtrack(idx + 1, mask | (1 << idx), count + 1)

        backtrack(0, 0, 0)
        return best

    def create_turan():
        adj = [[False]*n for _ in range(n)]
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        for i in range(3):
            for j in range(i+1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj[a][b] = True
                        adj[b][a] = True
        return adj

    def adj_to_mask(adj):
        masks = [0]*n
        for i in range(n):
            m = 0
            for j in range(n):
                if adj[i][j]:
                    m |= (1 << j)
            masks[i] = m
        return masks

    def mask_to_edges(masks):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if masks[i] & (1 << j):
                    edges.append([i, j])
        return edges

    def is_k4_free(adj):
        for a in range(n):
            for b in range(a+1, n):
                if not adj[a][b]: continue
                for c in range(b+1, n):
                    if not adj[a][c] or not adj[b][c]: continue
                    for d in range(c+1, n):
                        if adj[a][d] and adj[b][d] and adj[c][d]:
                            return False
        return True

    def check_k4_on_add(adj, u, v):
        # Check if adding (u,v) creates a K4
        # K4 must contain u and v.
        # Other two vertices x, y must be neighbors of both u and v.
        # And x, y must be neighbors of each other.
        common = []
        for w in range(n):
            if w != u and w != v and adj[u][w] and adj[v][w]:
                common.append(w)
        for i in range(len(common)):
            for j in range(i+1, len(common)):
                x = common[i]
                y = common[j]
                if adj[x][y]:
                    return False # K4 found
        return True

    def find_large_triangle_free(adj_masks):
        # Returns the mask of the largest triangle-free induced subgraph
        best_mask = 0
        best_count = 0
        
        def has_edge_in_mask(mask):
            m = mask
            while m:
                v_bit = m & -m
                v = v_bit.bit_length() - 1
                if adj_masks[v] & (mask ^ v_bit):
                    return True
                m ^= v_bit
            return False

        def backtrack(idx, mask, count):
            nonlocal best_mask, best_count
            if count > best_count:
                best_count = count
                best_mask = mask
            if idx == n:
                return
            if count + (n - idx) <= best_count:
                return
            
            # Skip
            backtrack(idx + 1, mask, count)
            
            # Include
            N = mask & adj_masks[idx]
            if not has_edge_in_mask(N):
                backtrack(idx + 1, mask | (1 << idx), count + 1)

        backtrack(0, 0, 0)
        return best_mask

    adj = create_turan()
    adj_masks = adj_to_mask(adj)
    current_score = max_triangle_free(adj_masks)
    best_score = current_score
    best_adj = [row[:] for row in adj]
    best_masks = adj_masks[:]

    start_time = time.time()
    max_time = 110 # Leave 10s for safety

    while time.time() - start_time < max_time:
        # Find the "bad" set
        bad_mask = find_large_triangle_free(adj_masks)
        bad_vertices = [i for i in range(n) if bad_mask & (1 << i)]
        
        if len(bad_vertices) <= 8:
            break

        # Try to find a swap: remove (x,y), add (u,v)
        # (u,v) should be within bad_vertices to create a triangle there
        # (x,y) should be an edge whose removal allows (u,v)
        
        moves = []
        
        # Candidates for (u,v): non-adjacent pairs in bad_vertices
        # We want adding (u,v) to create a triangle in bad_vertices
        # So there must be w in bad_vertices such that u-w and v-w are edges.
        for i in range(len(bad_vertices)):
            u = bad_vertices[i]
            for j in range(i+1, len(bad_vertices)):
                v = bad_vertices[j]
                if adj[u][v]:
                    continue # Already an edge
                
                # Check if adding (u,v) creates a triangle in bad_vertices
                creates_triangle = False
                for w in bad_vertices:
                    if w != u and w != v and adj[u][w] and adj[v][w]:
                        creates_triangle = True
                        break
                if not creates_triangle:
                    continue
                
                # Check if adding (u,v) creates K4 in current graph
                if not check_k4_on_add(adj, u, v):
                    # Need to remove an edge (x,y)
                    # The K4 is {u, v, x, y} where x,y are common neighbors
                    # We can remove one of the edges in this K4 (other than u-v which we are adding)
                    # Actually, we just need to ensure no K4 exists after swap.
                    # The K4s formed by adding (u,v) involve common neighbors.
                    # Let's identify the specific K4s that would form.
                    k4s_to_break = []
                    common = [w for w in range(n) if w != u and w != v and adj[u][w] and adj[v][w]]
                    for a_idx in range(len(common)):
                        for b_idx in range(a_idx+1, len(common)):
                            x = common[a_idx]
                            y = common[b_idx]
                            if adj[x][y]:
                                # K4 is {u, v, x, y}
                                # We can remove any edge in this K4 except (u,v)
                                # Removing (u,x), (u,y), (v,x), (v,y), (x,y)
                                # But (u,v) is not in graph yet.
                                # So we can remove (x,y) or any edge connecting u/v to x/y.
                                # Removing (x,y) is often good as it breaks other potential K4s?
                                # Let's try removing (x,y) first.
                                k4s_to_break.append((x, y))
                                # Also consider removing (u,x) etc?
                                # If we remove (u,x), does it help?
                                # Yes, it breaks this K4.
                                # But we want to minimize score.
                                # Let's just try removing (x,y) as a heuristic.
                                # Or try all edges in the K4.
                                k4s_to_break.append((u, x))
                                k4s_to_break.append((u, y))
                                k4s_to_break.append((v, x))
                                k4s_to_break.append((v, y))
                    
                    for x, y in k4s_to_break:
                        if x == y: continue
                        # Check if (x,y) is currently an edge
                        if adj[x][y]:
                            # Simulate swap
                            # Remove (x,y)
                            adj[x][y] = False
                            adj[y][x] = False
                            # Add (u,v)
                            adj[u][v] = True
                            adj[v][u] = True
                            
                            # Check if graph is still K4-free
                            if is_k4_free(adj):
                                # Evaluate score
                                new_masks = adj_to_mask(adj)
                                new_score = max_triangle_free(new_masks)
                                if new_score < current_score:
                                    moves.append((new_score, u, v, x, y))
                            
                            # Revert
                            adj[x][y] = True
                            adj[y][x] = True
                            adj[u][v] = False
                            adj[v][u] = False

        if moves:
            # Pick the best move
            moves.sort(key=lambda x: x[0])
            _, u, v, x, y = moves[0]
            
            # Apply move
            adj[x][y] = False
            adj[y][x] = False
            adj[u][v] = True
            adj[v][u] = True
            
            adj_masks = adj_to_mask(adj)
            current_score = max_triangle_free(adj_masks)
            
            if current_score < best_score:
                best_score = current_score
                best_adj = [row[:] for row in adj]
                best_masks = adj_masks[:]
                
            # If we improved, we can continue.
            # If not, we might be stuck.
        else:
            # No improving move found in this neighborhood.
            # Try a random restart or perturbation?
            # For now, break to avoid wasting time.
            # Or try removing a random edge and adding a random edge?
            # Let's just break for simplicity in this version.
            break

    return mask_to_edges(best_masks)

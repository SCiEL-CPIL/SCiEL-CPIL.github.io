import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def has_k222(colors, n, color):
        # Check if color 'color' contains K2,2,2
        # K2,2,2 vertices: u1, u2, v1, v2, w1, w2
        # Edges between U and V, U and W, V and W must be 'color'.
        # We can iterate over pairs of pairs.
        # Number of vertices n.
        # Choose 2 for U, 2 for V, 2 for W.
        # This is O(n^6) which is too slow for n=29.
        # Optimization:
        # For a fixed pair (u1, u2), find all vertices v such that (u1,v) and (u2,v) are 'color'.
        # Let S_u be the set of such vertices.
        # Then we need to find two disjoint pairs (v1, v2) and (w1, w2) in S_u 
        # such that all edges between {v1,v2} and {w1,w2} are 'color'.
        # Actually, if we pick v1, v2 from S_u, we need w1, w2 from S_u 
        # such that (v1,w1), (v1,w2), (v2,w1), (v2,w2) are 'color'.
        # This is still complex.
        
        # Alternative:
        # A K2,2,2 is a subgraph of K6.
        # We can precompute for each color, the adjacency matrix.
        # Then check for K2,2,2.
        
        # Efficient check:
        # For each pair of vertices (u1, u2):
        #   Let N_u = {v | (u1,v) is color and (u2,v) is color and v != u1, u2}
        #   We need to find v1, v2, w1, w2 in N_u such that:
        #   v1 != v2, w1 != w2, {v1,v2} != {w1,w2} (disjoint sets? No, just distinct vertices for the 6 parts? 
        #   Yes, K2,2,2 has 6 distinct vertices. So u1,u2,v1,v2,w1,w2 must be distinct.
        #   And edges between {v1,v2} and {w1,w2} must be color.
        
        # Let's build the graph G_color.
        # For each pair (u1, u2), compute common neighbors N(u1, u2).
        # Within N(u1, u2), we need to find a K2,2 (which is a 4-cycle) involving 4 distinct vertices v1,v2,w1,w2?
        # No, the condition is that all edges between {v1,v2} and {w1,w2} exist.
        # This means {v1,v2} and {w1,w2} form a complete bipartite graph K2,2 in G_color.
        # So we need to find a K2,2 in the subgraph induced by N(u1, u2) \ {u1, u2}.
        # Note: u1, u2 are not in N(u1, u2) usually (no self loops).
        
        # So:
        # 1. Build adjacency matrix A for color.
        # 2. For each pair (u1, u2):
        #    Let S = {v | A[u1][v] and A[u2][v]}
        #    If |S| < 4, continue.
        #    Check if the subgraph induced by S contains a K2,2.
        #    A K2,2 exists if there are two distinct vertices v1, v2 in S such that 
        #    they have at least 2 common neighbors in S.
        #    (Because if v1 and v2 share neighbors w1 and w2, then {v1,v2} and {w1,w2} form K2,2).
        
        # This is O(n^2 * n^2) = O(n^4) per color. For n=29, 29^4 ~ 700k. Very fast.
        
        adj = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                if c == color:
                    adj[i][j] = 1
                    adj[j][i] = 1
                idx += 1
        
        for u1 in range(n):
            for u2 in range(u1+1, n):
                # Find common neighbors
                S = []
                for v in range(n):
                    if v == u1 or v == u2: continue
                    if adj[u1][v] and adj[u2][v]:
                        S.append(v)
                if len(S) < 4: continue
                # Check for K2,2 in S
                # Map S to indices 0..k-1 for easier checking? Or just use list.
                # For each pair (v1, v2) in S:
                #   Count common neighbors in S.
                #   If count >= 2, return True.
                k = len(S)
                # Precompute common neighbor counts within S?
                # Just iterate pairs in S.
                for a_idx in range(k):
                    v1 = S[a_idx]
                    for b_idx in range(a_idx+1, k):
                        v2 = S[b_idx]
                        common = 0
                        for c_idx in range(k):
                            w = S[c_idx]
                            if w == v1 or w == v2: continue
                            if adj[v1][w] and adj[v2][w]:
                                common += 1
                                if common >= 2:
                                    return True
        return False

    def count_k222(colors, n, color):
        # Similar to has_k222 but counts
        adj = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[idx]
                if c == color:
                    adj[i][j] = 1
                    adj[j][i] = 1
                idx += 1
        
        count = 0
        for u1 in range(n):
            for u2 in range(u1+1, n):
                S = []
                for v in range(n):
                    if v == u1 or v == u2: continue
                    if adj[u1][v] and adj[u2][v]:
                        S.append(v)
                if len(S) < 4: continue
                k = len(S)
                for a_idx in range(k):
                    v1 = S[a_idx]
                    for b_idx in range(a_idx+1, k):
                        v2 = S[b_idx]
                        common = 0
                        for c_idx in range(k):
                            w = S[c_idx]
                            if w == v1 or w == v2: continue
                            if adj[v1][w] and adj[v2][w]:
                                common += 1
                        if common >= 2:
                            count += 1
        return count

    def is_valid(colors, n):
        return not has_k222(colors, n, 0) and not has_k222(colors, n, 1)

    def random_colors(n):
        return [random.randint(0, 1) for _ in range(n*(n-1)//2)]

    def local_search(colors, n, max_iter=10000):
        current_violations = count_k222(colors, n, 0) + count_k222(colors, n, 1)
        if current_violations == 0:
            return colors, 0
        
        best_colors = colors[:]
        best_violations = current_violations
        temp = 1.0
        cooling = 0.999
        
        for i in range(max_iter):
            if time.time() > deadline:
                break
            # Pick a random edge to flip
            edge_idx = random.randint(0, n*(n-1)//2 - 1)
            old_val = colors[edge_idx]
            new_val = 1 - old_val
            colors[edge_idx] = new_val
            
            new_violations = count_k222(colors, n, 0) + count_k222(colors, n, 1)
            
            delta = new_violations - current_violations
            if delta < 0 or random.random() < pow(2.71828, -delta/temp):
                current_violations = new_violations
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_colors = colors[:]
                    if best_violations == 0:
                        return best_colors, 0
            else:
                colors[edge_idx] = old_val
            
            temp *= cooling
        
        return best_colors, best_violations

    # Try to find valid coloring for increasing n
    best_n = 17
    best_colors = None
    
    # Use Paley for 17 as baseline
    squares = {x * x % 17 for x in range(1, 17)}
    p17 = [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]
    if is_valid(p17, 17):
        best_n = 17
        best_colors = p17

    # Try n = 20, 25, 29
    for target_n in [20, 25, 29]:
        if time.time() > deadline - 20:
            break
            
        # Initialize with random or extended Paley?
        # Random is often better for local search
        colors = random_colors(target_n)
        
        # Run local search
        colors, viol = local_search(colors, target_n, max_iter=5000)
        
        if viol == 0:
            best_n = target_n
            best_colors = colors
            break # Found a valid one, try larger?
            # If we found 20, we should try 25. 
            # If we found 25, try 29.
            # The loop handles this.
        
        # If not valid, keep best_n so far.

    # If we didn't find a valid one for >17, return 17
    if best_colors is None:
        return {'n': 17, 'colors': p17}
        
    return {'n': best_n, 'colors': best_colors}

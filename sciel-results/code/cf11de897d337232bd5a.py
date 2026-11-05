import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for 120s limit

    def has_k222(colors, n, color):
        # Build adjacency matrix for the given color
        # adj[i][j] is 1 if edge (i,j) has the given color
        adj = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if colors[idx] == color:
                    adj[i][j] = 1
                    adj[j][i] = 1
                idx += 1
        
        # Check for K2,2,2 in this color
        # For each pair (u1, u2), find common neighbors S
        # Check if S contains a K2,2
        for u1 in range(n):
            for u2 in range(u1+1, n):
                # Find S: vertices connected to both u1 and u2 in 'color'
                S = []
                for v in range(n):
                    if v == u1 or v == u2:
                        continue
                    if adj[u1][v] == 1 and adj[u2][v] == 1:
                        S.append(v)
                
                if len(S) < 4:
                    continue
                
                # Check if subgraph induced by S contains K2,2
                # K2,2 exists if there are two vertices in S with >= 2 common neighbors in S
                k = len(S)
                # Map S to 0..k-1 for faster index? Or just use values.
                # Using values is fine for n=30.
                for i in range(k):
                    v1 = S[i]
                    for j in range(i+1, k):
                        v2 = S[j]
                        common_count = 0
                        for l in range(k):
                            v3 = S[l]
                            if v3 == v1 or v3 == v2:
                                continue
                            if adj[v1][v3] == 1 and adj[v2][v3] == 1:
                                common_count += 1
                                if common_count >= 2:
                                    return True
        return False

    def count_k222(colors, n, color):
        # Similar to has_k222 but counts
        adj = [[0]*n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                if colors[idx] == color:
                    adj[i][j] = 1
                    adj[j][i] = 1
                idx += 1
        
        count = 0
        for u1 in range(n):
            for u2 in range(u1+1, n):
                S = []
                for v in range(n):
                    if v == u1 or v == u2:
                        continue
                    if adj[u1][v] == 1 and adj[u2][v] == 1:
                        S.append(v)
                
                if len(S) < 4:
                    continue
                
                k = len(S)
                for i in range(k):
                    v1 = S[i]
                    for j in range(i+1, k):
                        v2 = S[j]
                        common_count = 0
                        for l in range(k):
                            v3 = S[l]
                            if v3 == v1 or v3 == v2:
                                continue
                            if adj[v1][v3] == 1 and adj[v2][v3] == 1:
                                common_count += 1
                        if common_count >= 2:
                            count += 1
        return count

    def is_valid(colors, n):
        return not has_k222(colors, n, 0) and not has_k222(colors, n, 1)

    def get_paley(n):
        # n must be prime q = 4k+1
        squares = {x*x % n for x in range(1, n)}
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                d = (j - i) % n
                if d in squares:
                    colors.append(0)
                else:
                    colors.append(1)
        return colors

    # Step 1: Try Paley 29
    n29 = 29
    colors29 = get_paley(n29)
    if is_valid(colors29, n29):
        best_n = n29
        best_colors = colors29
    else:
        best_n = 17
        best_colors = get_paley(17)

    # Step 2: Try to extend to n=30
    if time.time() < deadline - 20:
        n30 = 30
        # Start with Paley 29 + random edges for new vertex
        base_colors = colors29[:] if best_n == 29 else get_paley(29)
        
        # If base is not valid, we might want to start from scratch or use a different base.
        # But if Paley 29 is valid, it's a good start.
        
        # Number of edges in K30 is 30*29/2 = 435
        # Edges in K29 is 29*28/2 = 406
        # New edges are (29, 0), (29, 1), ..., (29, 28)
        # Indices for new edges:
        # The list is lexicographic: (0,1), (0,2)...(0,28), (1,2)...(1,28), ..., (28,29)
        # The last 29 edges are (0,29), (1,29), ..., (28,29)?
        # No, lexicographic order of pairs (i,j) with i<j.
        # (0,1)...(0,28) -> 28 edges
        # (1,2)...(1,28) -> 27 edges
        # ...
        # (28,29) -> 1 edge
        # Total 406.
        # For n=30:
        # (0,1)...(0,29) -> 29 edges
        # (1,2)...(1,29) -> 28 edges
        # ...
        # (29,30) is not a vertex. Vertices are 0..29.
        # Pairs involving 29: (0,29), (1,29), ..., (28,29).
        # These are scattered in the list.
        # It is easier to generate a full list of 435 colors.
        
        colors30 = [0] * (30*29//2)
        
        # Copy base colors where possible?
        # The indices for K29 are a subset of K30 indices?
        # Index for (i,j) in K_n is sum_{k=0}^{i-1} (n-1-k) + (j-i-1)
        # This mapping changes when n changes.
        # So we can't just copy. We must reconstruct.
        
        # Initialize colors30
        # Use Paley 29 logic for i,j < 29
        squares_29 = {x*x % 29 for x in range(1, 29)}
        idx = 0
        for i in range(30):
            for j in range(i+1, 30):
                if i < 29 and j < 29:
                    d = (j - i) % 29
                    if d in squares_29:
                        colors30[idx] = 0
                    else:
                        colors30[idx] = 1
                else:
                    # Random init for edges involving vertex 29
                    colors30[idx] = random.randint(0, 1)
                idx += 1
        
        # Local search
        current_colors = colors30
        current_violations = count_k222(current_colors, 30, 0) + count_k222(current_colors, 30, 1)
        
        best_colors30 = current_colors[:]
        best_violations = current_violations
        
        temp = 1.0
        cooling = 0.999
        max_iter = 50000
        
        for i in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Pick random edge
            edge_idx = random.randint(0, 434)
            old_val = current_colors[edge_idx]
            new_val = 1 - old_val
            current_colors[edge_idx] = new_val
            
            new_violations = count_k222(current_colors, 30, 0) + count_k222(current_colors, 30, 1)
            
            delta = new_violations - current_violations
            if delta < 0 or random.random() < pow(2.71828, -delta/temp):
                current_violations = new_violations
                if new_violations < best_violations:
                    best_violations = new_violations
                    best_colors30 = current_colors[:]
                    if best_violations == 0:
                        best_n = 30
                        best_colors = best_colors30
                        break
            else:
                current_colors[edge_idx] = old_val
            
            temp *= cooling
        
        if best_violations == 0:
            best_n = 30
            best_colors = best_colors30

    return {'n': best_n, 'colors': best_colors}

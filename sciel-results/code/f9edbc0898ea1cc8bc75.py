import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n = 30
    num_edges = n * (n - 1) // 2
    
    # Edge index map for quick access
    # We store colors in a list of length num_edges
    # We also maintain adj bitmasks for speed
    
    # Initialize random coloring
    colors = [random.randint(0, 1) for _ in range(num_edges)]
    
    # Build edge index mapping (i, j) -> idx
    edge_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx[(i, j)] = idx
            idx += 1

    def rebuild_adj(colors):
        adj = [[0, 0] for _ in range(n)]
        for (i, j), idx in edge_idx.items():
            c = colors[idx]
            adj[i][c] |= (1 << j)
            adj[j][c] |= (1 << i)
        return adj

    adj = rebuild_adj(colors)
    
    def count_bad_pairs(adj):
        """
        Count the number of pairs (u, v) such that the common neighborhood S = N(u) & N(v)
        in the same color contains a K_{2,2}.
        """
        bad_count = 0
        for c in range(2):
            for u in range(n):
                for v in range(u + 1, n):
                    # Check if edge (u, v) has color c
                    if not (adj[u][c] & (1 << v)):
                        continue
                    
                    S = adj[u][c] & adj[v][c]
                    # S contains vertices connected to both u and v in color c
                    # We need to check if the subgraph induced by S in color c contains K_{2,2}
                    # K_{2,2} exists if there are two vertices a, b in S such that
                    # N(a) & N(b) & S has size >= 2.
                    
                    # Get list of vertices in S
                    s_list = []
                    mask = S
                    while mask:
                        lb = mask & (-mask)
                        v_idx = lb.bit_length() - 1
                        s_list.append(v_idx)
                        mask ^= lb
                    
                    # Check for K_{2,2} in G[S]
                    found_k22 = False
                    for i in range(len(s_list)):
                        a = s_list[i]
                        for j in range(i + 1, len(s_list)):
                            b = s_list[j]
                            # Check if edge (a, b) is color c
                            if not (adj[a][c] & (1 << b)):
                                continue
                            T = adj[a][c] & adj[b][c] & S
                            if T.bit_count() >= 2:
                                found_k22 = True
                                break
                        if found_k22:
                            break
                    if found_k22:
                        bad_count += 1
        return bad_count

    # Simulated Annealing
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    current_bad = count_bad_pairs(adj)
    best_bad = current_bad
    best_colors = colors[:]
    
    while time.time() < deadline:
        if current_bad == 0:
            break
            
        # Pick a random edge to flip
        e_idx = random.randint(0, num_edges - 1)
        
        # Determine which edge (u, v) this is
        # We need a reverse map or just iterate. Since n=30, we can precompute a list of edges
        # Let's precompute edge list
        # Actually, we can just use the edge_idx keys reversed? No, that's a dict.
        # Let's create a list of (i, j, idx)
        
        # Optimization: Precompute edge list
        if not hasattr(construct, 'edge_list'):
            construct.edge_list = []
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    construct.edge_list.append((i, j, idx))
                    idx += 1
        
        u, v, idx = construct.edge_list[e_idx]
        
        # Flip color
        old_color = colors[idx]
        new_color = 1 - old_color
        colors[idx] = new_color
        
        # Update adj bitmasks
        adj[u][old_color] &= ~(1 << v)
        adj[u][new_color] |= (1 << v)
        adj[v][old_color] &= ~(1 << u)
        adj[v][new_color] |= (1 << u)
        
        new_bad = count_bad_pairs(adj)
        
        delta = new_bad - current_bad
        
        if delta <= 0 or random.random() < math.exp(-delta / T):
            current_bad = new_bad
            if new_bad < best_bad:
                best_bad = new_bad
                best_colors = colors[:]
        else:
            # Revert
            colors[idx] = old_color
            adj[u][new_color] &= ~(1 << v)
            adj[u][old_color] |= (1 << v)
            adj[v][new_color] &= ~(1 << u)
            adj[v][old_color] |= (1 << u)
            
        T *= cooling
        if T < min_T:
            T = min_T

    # If we found a valid coloring, return it
    if best_bad == 0:
        return {'n': 30, 'colors': best_colors}
    else:
        # Fallback to n=29 Paley graph
        p = 29
        squares = {x * x % p for x in range(1, p)}
        base_colors = []
        for i in range(p):
            for j in range(i + 1, p):
                diff = (j - i) % p
                if diff in squares:
                    base_colors.append(0)
                else:
                    base_colors.append(1)
        return {'n': 29, 'colors': base_colors}

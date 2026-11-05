import time
import random
import numpy as np

def construct():
    deadline = time.time() + 118  # Safety margin
    
    def build_adj(colors, n):
        adj = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if colors[idx]:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                idx += 1
        return adj

    def count_violations(adj, n):
        count = 0
        # Iterate over pairs u1, u2
        for u1 in range(n):
            for u2 in range(u1 + 1, n):
                # S = N(u1) & N(u2)
                s_mask = adj[u1] & adj[u2]
                # Remove u1, u2 from S? 
                # u1 is not in N(u1). u2 is in N(u1) if edge exists.
                # In K2,2,2, vertices are distinct.
                # So u1, u2 must not be in S.
                s_mask &= ~(1 << u1)
                s_mask &= ~(1 << u2)
                
                # Extract vertices in S
                verts = []
                m = s_mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    verts.append(idx)
                    m ^= lsb
                
                l = len(verts)
                if l < 4:
                    continue
                
                # Check for K2,2 in G[S]
                # K2,2 exists if there is v1, v2 in S such that |N(v1) & N(v2) & S| >= 2
                for i in range(l):
                    v1 = verts[i]
                    for j in range(i + 1, l):
                        v2 = verts[j]
                        common = adj[v1] & adj[v2] & s_mask
                        if common & (common - 1): # at least 2 bits set
                            # Count how many pairs of w's
                            # We just need to know if violation exists, or count them for score
                            # For optimization, counting is better for gradient
                            c = bin(common).count('1')
                            count += c * (c - 1) // 2
        return count

    def total_violations(colors, n):
        adj0 = build_adj([1-c for c in colors], n) # Wait, colors are 0/1. 
        # Let's just build two adjacencies or pass color to build_adj
        # Better: build_adj takes colors and target_color
        pass

    # Refined build_adj
    def build_adj_for_color(colors, n, color):
        adj = [0] * n
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if colors[idx] == color:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                idx += 1
        return adj

    def total_violations_fast(colors, n):
        adj0 = build_adj_for_color(colors, n, 0)
        adj1 = build_adj_for_color(colors, n, 1)
        return count_violations(adj0, n) + count_violations(adj1, n)

    # 1. Paley 29
    q = 29
    squares = {x * x % q for x in range(1, q)}
    colors_29 = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in squares:
                colors_29.append(0)
            else:
                colors_29.append(1)
    
    # Verify
    if total_violations_fast(colors_29, 29) == 0:
        best_n = 29
        best_colors = colors_29
    else:
        best_n = 1
        best_colors = [0]

    # 2. Try n=30
    n = 30
    n_edges = n * (n - 1) // 2
    
    # Initialize with Paley 29 + random 29 edges
    init_colors = colors_29.copy()
    for _ in range(29):
        init_colors.append(random.randint(0, 1))
    
    current_colors = init_colors
    current_viol = total_violations_fast(current_colors, n)
    
    if current_viol == 0:
        return {'n': n, 'colors': current_colors}
        
    # Simulated Annealing / Hill Climbing
    # We want to minimize violations.
    T = 1.0
    T_min = 0.01
    alpha = 0.999
    
    while time.time() < deadline - 5 and T > T_min:
        edge_idx = random.randint(0, n_edges - 1)
        old_val = current_colors[edge_idx]
        new_val = 1 - old_val
        current_colors[edge_idx] = new_val
        
        new_viol = total_violations_fast(current_colors, n)
        
        delta = new_viol - current_viol
        if delta < 0 or random.random() < pow(2.718, -delta / T):
            current_viol = new_viol
            if current_viol == 0:
                return {'n': n, 'colors': current_colors}
        else:
            current_colors[edge_idx] = old_val
            
        T *= alpha
        
    return {'n': best_n, 'colors': best_colors}

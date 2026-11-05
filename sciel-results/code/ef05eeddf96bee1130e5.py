import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Generate Paley coloring for n=29
    def generate_paley(n, prime):
        res = set()
        for x in range(1, prime):
            res.add((x * x) % prime)
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                diff = (j - i) % prime
                if diff in res:
                    colors.append(0)
                else:
                    colors.append(1)
        return colors

    # Build adjacency bitmasks from colors
    def build_adj(colors, n):
        adj = [[0] * n for _ in range(2)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        return adj

    # Count K2,2,2 violations using adjacency bitmasks
    def count_violations(adj, n):
        total_viol = 0
        for c in [0, 1]:
            A = adj[c]
            for u1 in range(n):
                for u2 in range(u1 + 1, n):
                    S = A[u1] & A[u2]
                    if S == 0:
                        continue
                    # Extract set bits
                    s_bits = []
                    temp = S
                    while temp:
                        lsb = temp & -temp
                        s_bits.append(lsb.bit_length() - 1)
                        temp ^= lsb
                    k = len(s_bits)
                    if k < 4:
                        continue
                    for a in range(k):
                        v1 = s_bits[a]
                        for b in range(a + 1, k):
                            v2 = s_bits[b]
                            C = A[v1] & A[v2]
                            common_in_S = C & S
                            if common_in_S & (common_in_S - 1):
                                total_viol += 1
        return total_viol

    # Fast validity check
    def is_valid_fast(colors, n):
        adj = build_adj(colors, n)
        return count_violations(adj, n) == 0

    # Local search for n=30 with incremental adjacency
    def local_search_30(colors, n):
        # Build initial adjacency
        adj = build_adj(colors, n)
        current_viol = count_violations(adj, n)
        best_colors = colors[:]
        best_viol = current_viol
        
        # Precompute edge index to (i,j) mapping
        num_edges = n * (n - 1) // 2
        edge_map = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map.append((i, j))
                idx += 1
        
        temp = 2.0
        cooling = 0.9995
        iter_count = 0
        stall_count = 0
        stall_threshold = 5000
        
        while time.time() < deadline - 8:
            iter_count += 1
            
            # Pick a random edge
            edge_idx = random.randint(0, num_edges - 1)
            i, j = edge_map[edge_idx]
            old_val = colors[edge_idx]
            new_val = 1 - old_val
            
            # Update adjacency incrementally
            # Remove old edge from adj[old_val]
            adj[old_val][i] &= ~(1 << j)
            adj[old_val][j] &= ~(1 << i)
            # Add new edge to adj[new_val]
            adj[new_val][i] |= (1 << j)
            adj[new_val][j] |= (1 << i)
            
            new_viol = count_violations(adj, n)
            colors[edge_idx] = new_val
            
            delta = new_viol - current_viol
            if delta < 0 or random.random() < pow(2.71828, -delta / temp):
                current_viol = new_viol
                if new_viol < best_viol:
                    best_viol = new_viol
                    best_colors = colors[:]
                    stall_count = 0
                    if best_viol == 0:
                        return best_colors
            else:
                # Revert: undo adjacency change
                adj[new_val][i] &= ~(1 << j)
                adj[new_val][j] &= ~(1 << i)
                adj[old_val][i] |= (1 << j)
                adj[old_val][j] |= (1 << i)
                colors[edge_idx] = old_val
                stall_count += 1
            
            temp *= cooling
            
            # If stuck for too long, try a multi-flip restart
            if stall_count > stall_threshold:
                # Try flipping 3 random edges together
                flipped_edges = []
                for _ in range(3):
                    e1 = random.randint(0, num_edges - 1)
                    i1, j1 = edge_map[e1]
                    old1 = colors[e1]
                    new1 = 1 - old1
                    adj[old1][i1] &= ~(1 << j1)
                    adj[old1][j1] &= ~(1 << i1)
                    adj[new1][i1] |= (1 << j1)
                    adj[new1][j1] |= (1 << i1)
                    colors[e1] = new1
                    flipped_edges.append((e1, i1, j1, old1, new1))
                
                nv = count_violations(adj, n)
                if nv < current_viol:
                    current_viol = nv
                    if nv < best_viol:
                        best_viol = nv
                        best_colors = colors[:]
                else:
                    # Revert all three
                    for (e1, i1, j1, old1, new1) in flipped_edges:
                        adj[new1][i1] &= ~(1 << j1)
                        adj[new1][j1] &= ~(1 << i1)
                        adj[old1][i1] |= (1 << j1)
                        adj[old1][j1] |= (1 << i1)
                        colors[e1] = old1
                # Reset stall
                stall_count = 0
                temp = 2.0  # Reheat
        
        return best_colors

    # 1. Generate and verify Paley(29)
    n29 = 29
    colors29 = generate_paley(n29, 29)
    
    if is_valid_fast(colors29, n29):
        # We have a valid n=29 solution.
        # Try to extend to n=30
        n30 = 30
        colors30 = colors29[:]
        for i in range(n29):
            colors30.append(random.randint(0, 1))
            
        colors30_optimized = local_search_30(colors30, n30)
        
        if is_valid_fast(colors30_optimized, n30):
            return {'n': 30, 'colors': colors30_optimized}
        else:
            return {'n': 29, 'colors': colors29}
    else:
        return {'n': 17, 'colors': generate_paley(17, 17)}

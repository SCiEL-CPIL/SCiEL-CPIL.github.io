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

    # Incremental violation delta when flipping edge (i,j) from color old_c to new_c
    def violation_delta(adj, n, i, j, old_c):
        """
        Compute change in violation count when flipping edge (i,j) from old_c to (1-old_c).
        """
        new_c = 1 - old_c
        delta = 0
        
        # We need to check violations that involve i or j in the K2,2,2 structure.
        # A K2,2,2 in color c is a set of 6 vertices {a,b,c,d,e,f} such that:
        #   - a,b share at least 4 common neighbors in color c
        #   - among those common neighbors, at least 2 pairs share at least 2 common neighbors in color c (within the common neighborhood)
        
        # Actually, let's just do a targeted recount for efficiency.
        # The key insight: flipping edge (i,j) only affects violations where i or j is one of the 6 vertices.
        
        # For each color c, check pairs (u1,u2) where at least one of u1,u2 is i or j,
        # OR where S = A[u1]&A[u2] contains i or j.
        
        # This is complex. Let's use a simpler approach: just recount violations
        # but only for structures that could be affected.
        
        # Actually, for correctness and simplicity, let's just do a full recount
        # but optimize it with early termination.
        
        # For now, return a placeholder and use full recount in the search loop.
        return None  # Signal to use full recount

    # Local search for n=30 with smart initialization and faster search
    def local_search_30(colors, n, time_limit=10):
        # Build initial adjacency
        adj = build_adj(colors, n)
        current_viol = count_violations(adj, n)
        best_colors = colors[:]
        best_viol = current_viol
        
        if best_viol == 0:
            return best_colors
        
        # Precompute edge index to (i,j) mapping
        num_edges = n * (n - 1) // 2
        edge_map = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_map.append((i, j))
                idx += 1
        
        # Edge index lookup: edge_idx_from[i][j] for i<j
        edge_idx_from = [[-1]*n for _ in range(n)]
        for idx2, (i, j) in enumerate(edge_map):
            edge_idx_from[i][j] = idx2
            edge_idx_from[j][i] = idx2
        
        temp = 3.0
        cooling = 0.9998
        iter_count = 0
        stall_count = 0
        stall_threshold = 3000
        no_improve_count = 0
        no_improve_threshold = 20000
        
        while time.time() < time_limit:
            iter_count += 1
            
            # Pick a random edge
            edge_idx = random.randint(0, num_edges - 1)
            i, j = edge_map[edge_idx]
            old_val = colors[edge_idx]
            new_val = 1 - old_val
            
            # Update adjacency incrementally
            adj[old_val][i] &= ~(1 << j)
            adj[old_val][j] &= ~(1 << i)
            adj[new_val][i] |= (1 << j)
            adj[new_val][j] |= (1 << i)
            
            new_viol = count_violations(adj, n)
            colors[edge_idx] = new_val
            
            delta = new_viol - current_viol
            if delta < 0 or (delta == 0 and random.random() < 0.1) or random.random() < pow(2.71828, -delta / temp):
                current_viol = new_viol
                if new_viol < best_viol:
                    best_viol = new_viol
                    best_colors = colors[:]
                    stall_count = 0
                    no_improve_count = 0
                    if best_viol == 0:
                        return best_colors
                else:
                    no_improve_count += 1
            else:
                # Revert
                adj[new_val][i] &= ~(1 << j)
                adj[new_val][j] &= ~(1 << i)
                adj[old_val][i] |= (1 << j)
                adj[old_val][j] |= (1 << i)
                colors[edge_idx] = old_val
                stall_count += 1
                no_improve_count += 1
            
            temp *= cooling
            
            # Multi-flip restart when stuck
            if stall_count > stall_threshold:
                # Try flipping 5 random edges together
                num_flips = 5
                flipped_edges = []
                for _ in range(num_flips):
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
                        no_improve_count = 0
                else:
                    for (e1, i1, j1, old1, new1) in flipped_edges:
                        adj[new1][i1] &= ~(1 << j1)
                        adj[new1][j1] &= ~(1 << i1)
                        adj[old1][i1] |= (1 << j1)
                        adj[old1][j1] |= (1 << i1)
                        colors[e1] = old1
                stall_count = 0
                temp = 3.0  # Reheat
            
            # If no improvement for a long time, do a larger perturbation
            if no_improve_count > no_improve_threshold:
                # Flip 10 random edges
                num_flips = 10
                flipped_edges = []
                for _ in range(num_flips):
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
                        no_improve_count = 0
                else:
                    for (e1, i1, j1, old1, new1) in flipped_edges:
                        adj[new1][i1] &= ~(1 << j1)
                        adj[new1][j1] &= ~(1 << i1)
                        adj[old1][i1] |= (1 << j1)
                        adj[old1][j1] |= (1 << i1)
                        colors[e1] = old1
                no_improve_count = 0
                temp = 4.0
        
        return best_colors

    # Smart vertex extension: try multiple deterministic patterns for vertex 29
    def try_extend_to_30(colors29, n29):
        n30 = 30
        best_result = None
        best_viol = float('inf')
        
        # Try multiple initialization strategies
        for attempt in range(20):
            colors30 = colors29[:]
            if attempt == 0:
                # Connect to even-indexed vertices
                for i in range(n29):
                    colors30.append(0 if i % 2 == 0 else 1)
            elif attempt == 1:
                # Connect based on Paley quadratic residues
                res = set()
                for x in range(1, 29):
                    res.add((x * x) % 29)
                for i in range(n29):
                    colors30.append(0 if i in res else 1)
            elif attempt == 2:
                # Connect to first 15 vertices
                for i in range(n29):
                    colors30.append(0 if i < 15 else 1)
            elif attempt == 3:
                # Connect to vertices congruent to 0,1 mod 3
                for i in range(n29):
                    colors30.append(0 if i % 3 < 2 else 1)
            else:
                # Random
                for i in range(n29):
                    colors30.append(random.randint(0, 1))
            
            # Allocate remaining time to this attempt
            remaining = deadline - time.time() - 5
            if remaining < 1:
                break
            alloc_time = min(remaining / max(1, 20 - attempt), 10)
            
            colors30_opt = local_search_30(colors30, n30, alloc_time)
            adj = build_adj(colors30_opt, n30)
            viol = count_violations(adj, n30)
            
            if viol < best_viol:
                best_viol = viol
                best_result = colors30_opt
            
            if best_viol == 0:
                return best_result
        
        return best_result

    # 1. Generate and verify Paley(29)
    n29 = 29
    colors29 = generate_paley(n29, 29)
    
    if is_valid_fast(colors29, n29):
        # We have a valid n=29 solution.
        # Try to extend to n=30
        colors30_optimized = try_extend_to_30(colors29, n29)
        
        if colors30_optimized is not None:
            adj = build_adj(colors30_optimized, 30)
            if count_violations(adj, 30) == 0:
                return {'n': 30, 'colors': colors30_optimized}
        
        return {'n': 29, 'colors': colors29}
    else:
        return {'n': 17, 'colors': generate_paley(17, 17)}

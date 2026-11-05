import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Precompute edge list for n up to 20
    # We will try n=17, 16, 15...
    
    def _has_path_len_5(adj, u, v, n):
        """
        Check if there is a path of length 5 between u and v in the graph defined by adj.
        adj is a list of lists (adjacency lists) or bitmasks.
        For speed, we use bitmasks.
        adj[i] is an int bitmask.
        Path: u - a - b - c - d - v
        We need to find a, b, c, d such that:
        a in N(u), b in N(a), c in N(b), d in N(c), v in N(d)
        And all vertices u,a,b,c,d,v are distinct.
        
        Optimization:
        Iterate a in N(u).
        Iterate b in N(a) \ {u}.
        Iterate c in N(b) \ {u, a}.
        Iterate d in N(c) \ {u, a, b}.
        Check if v in N(d) and v not in {u, a, b, c, d}.
        """
        # N(u)
        mask_a = adj[u]
        # We don't need to exclude v here yet, but we should ensure a != v for simple path logic?
        # A C6 is a simple cycle. So u != v, a != u, a != v? 
        # If a == v, path is u-v-...-v? No, path is u-a-b-c-d-v.
        # If a == v, then edge (u,v) exists in this color. But we are checking if adding (u,v) creates a C6.
        # The existing graph does NOT contain (u,v) yet. So a cannot be v because (u,v) is not in adj yet.
        # Wait, adj represents the graph BEFORE adding (u,v).
        # So v is not in N(u) in adj.
        
        while mask_a:
            a = (mask_a & -mask_a).bit_length() - 1
            mask_a &= mask_a - 1
            
            # b in N(a) \ {u}
            mask_b = adj[a] & ~(1 << u)
            while mask_b:
                b = (mask_b & -mask_b).bit_length() - 1
                mask_b &= mask_b - 1
                
                # c in N(b) \ {u, a}
                mask_c = adj[b] & ~(1 << u) & ~(1 << a)
                while mask_c:
                    c = (mask_c & -mask_c).bit_length() - 1
                    mask_c &= mask_c - 1
                    
                    # d in N(c) \ {u, a, b}
                    mask_d = adj[c] & ~(1 << u) & ~(1 << a) & ~(1 << b)
                    while mask_d:
                        d = (mask_d & -mask_d).bit_length() - 1
                        mask_d &= mask_d - 1
                        
                        # Check if v in N(d) and v not in {u, a, b, c, d}
                        if adj[d] & (1 << v):
                            # Ensure v is distinct from u, a, b, c, d
                            # v != u is given (edge u-v)
                            # v != a: if v==a, then a is in N(u) and v is in N(d). 
                            # But a is in N(u) means edge (u,a) exists. 
                            # If a==v, then edge (u,v) exists in adj? No, we said adj doesn't have (u,v).
                            # So a cannot be v.
                            # v != b: b is in N(a). If b==v, then edge (a,v) exists.
                            # v != c: c is in N(b). If c==v, edge (b,v) exists.
                            # v != d: d is in N(c). If d==v, edge (c,v) exists.
                            # We just need to ensure v is not u, a, b, c, d.
                            # Since u,a,b,c,d are distinct in the loops (by exclusion), 
                            # and v is a specific vertex:
                            # We just need to check if v is one of u,a,b,c,d.
                            if v != u and v != a and v != b and v != c and v != d:
                                return True
        return False

    def _try_assign(adj, u, v, c):
        """
        Attempt to add edge (u,v) with color c.
        Returns True if valid (no C6 created), False otherwise.
        Does not modify adj.
        """
        # Check if (u,v) already exists in color c? 
        # In a complete graph coloring, each edge has exactly one color.
        # So (u,v) is not in any adj yet.
        
        # Check if adding (u,v) to color c creates a C6.
        # This happens if there is a path of length 5 between u and v in the current graph of color c.
        if _has_path_len_5(adj[c], u, v, 20): # n is small, use max 20 for mask size
            return False
        return True

    def _backtrack_csp(n, time_limit):
        """
        Solve 4-coloring of K_n with no mono C6 using Backtracking with Forward Checking.
        """
        deadline = time.time() + time_limit
        E = n * (n - 1) // 2
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Randomize edge order to help escape symmetry and find solutions faster
        random.shuffle(edges)
        
        # adj[c][i] is bitmask of neighbors of i in color c
        adj = [[0] * n for _ in range(4)]
        colors = [-1] * E # colors[idx] corresponds to edges[idx]
        
        # To speed up, we can order edges by vertex degree or just random.
        # Random is fine.
        
        def backtrack(idx):
            if time.time() > deadline:
                return False, "timeout"
            
            if idx == len(edges):
                return True, "success"
            
            u, v = edges[idx]
            
            # Try colors 0-3 in random order
            color_order = [0, 1, 2, 3]
            random.shuffle(color_order)
            
            for c in color_order:
                if _try_assign(adj, u, v, c):
                    # Assign
                    adj[c][u] |= (1 << v)
                    adj[c][v] |= (1 << u)
                    colors[idx] = c
                    
                    success, status = backtrack(idx + 1)
                    if success:
                        return True, "success"
                    
                    # Unassign
                    adj[c][u] &= ~(1 << v)
                    adj[c][v] &= ~(1 << u)
                    colors[idx] = -1
                    
                    if status == "timeout":
                        return False, "timeout"
            
            return False, "backtrack"

        success, status = backtrack(0)
        if success:
            # Reconstruct colors in lex order
            res_colors = [0] * E
            # We need to map the shuffled edge index back to lex order
            # edges[idx] = (u, v). Lex index is determined by u, v.
            # We can just iterate through res_colors and find the matching edge in colors?
            # No, colors is indexed by the position in the 'edges' list.
            # We need to output colors in lex order of (i,j).
            
            # Create a map from (u,v) to the index in 'edges'
            edge_to_idx = {e: i for i, e in enumerate(edges)}
            
            for i in range(n):
                for j in range(i + 1, n):
                    idx_in_edges = edge_to_idx[(i, j)]
                    res_colors[i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)] = colors[idx_in_edges]
            
            # Wait, the lex index formula:
            # Index of (i,j) in list [(0,1), (0,2), ..., (0,n-1), (1,2), ...]
            # Count edges before i: sum_{k=0}^{i-1} (n - 1 - k)
            # Count edges for i before j: j - i - 1
            # Let's just build it simply to avoid errors.
            
            res_colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    idx_in_edges = edge_to_idx[(i, j)]
                    res_colors.append(colors[idx_in_edges])
            
            return res_colors
        return None

    # Strategy:
    # 1. Try n=17 with backtracking.
    # 2. If fails or times out, try n=16.
    # 3. Fallback to local search if backtracking fails for n=17? 
    #    Backtracking with FC is usually much stronger for CSPs like this.
    
    best_n = 0
    best_colors = None
    
    # We know n=17 is possible.
    # Let's try n=17 first.
    # Allocate 60s for n=17.
    # If it fails, try n=16 with 30s.
    
    for n in [17, 16, 15]:
        if time.time() > deadline - 10:
            break
            
        time_alloc = 50.0 if n == 17 else 30.0
        
        # Run backtracking
        sol = _backtrack_csp(n, time_limit=time_alloc)
        
        if sol is not None:
            # Verify solution
            # Quick check: no C6 in any color
            adj = [[0] * n for _ in range(4)]
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = sol[idx]
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
                    idx += 1
            
            valid = True
            for c in range(4):
                # Check for C6
                for v0 in range(n):
                    mask_v1 = adj[c][v0]
                    while mask_v1:
                        v1 = (mask_v1 & -mask_v1).bit_length() - 1
                        mask_v1 &= mask_v1 - 1
                        mask_v2 = adj[c][v1] & ~(1 << v0)
                        while mask_v2:
                            v2 = (mask_v2 & -mask_v2).bit_length() - 1
                            mask_v2 &= mask_v2 - 1
                            mask_v3 = adj[c][v2] & ~(1 << v0) & ~(1 << v1)
                            while mask_v3:
                                v3 = (mask_v3 & -mask_v3).bit_length() - 1
                                mask_v3 &= mask_v3 - 1
                                mask_v4 = adj[c][v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                                while mask_v4:
                                    v4 = (mask_v4 & -mask_v4).bit_length() - 1
                                    mask_v4 &= mask_v4 - 1
                                    mask_v5 = adj[c][v4] & adj[c][v0]
                                    used_mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                                    mask_v5 &= ~used_mask
                                    if mask_v5:
                                        valid = False
                                        break
                                if not valid: break
                            if not valid: break
                        if not valid: break
                    if not valid: break
                if not valid: break
            
            if valid:
                best_n = n
                best_colors = sol
                break # Found a solution for this n, which is the max we tried. 
                # Since we try 17, 16, 15, if 17 works, we return 17.
        
    # If backtracking failed for all, fallback to a simple local search for n=15 or lower
    # Or just return n=4 if nothing works (should not happen)
    if best_n == 0:
        # Fallback: Local search for n=15
        def _local_search(n, time_limit=10.0):
            deadline = time.time() + time_limit
            E = n * (n - 1) // 2
            colors = [random.randint(0, 3) for _ in range(E)]
            
            def count_c6s(cols):
                adj = [[0] * n for _ in range(4)]
                idx = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        c = cols[idx]
                        adj[c][i] |= (1 << j)
                        adj[c][j] |= (1 << i)
                        idx += 1
                total = 0
                for c in range(4):
                    for v0 in range(n):
                        mask_v1 = adj[c][v0]
                        while mask_v1:
                            v1 = (mask_v1 & -mask_v1).bit_length() - 1
                            mask_v1 &= mask_v1 - 1
                            mask_v2 = adj[c][v1] & ~(1 << v0)
                            while mask_v2:
                                v2 = (mask_v2 & -mask_v2).bit_length() - 1
                                mask_v2 &= mask_v2 - 1
                                mask_v3 = adj[c][v2] & ~(1 << v0) & ~(1 << v1)
                                while mask_v3:
                                    v3 = (mask_v3 & -mask_v3).bit_length() - 1
                                    mask_v3 &= mask_v3 - 1
                                    mask_v4 = adj[c][v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                                    while mask_v4:
                                        v4 = (mask_v4 & -mask_v4).bit_length() - 1
                                        mask_v4 &= mask_v4 - 1
                                        mask_v5 = adj[c][v4] & adj[c][v0]
                                        used_mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                                        mask_v5 &= ~used_mask
                                        if mask_v5:
                                            total += 1
                return total

            current_c6s = count_c6s(colors)
            best_colors_ls = colors[:]
            best_c6s = current_c6s
            
            for _ in range(10000):
                if time.time() > deadline:
                    break
                if best_c6s == 0:
                    return best_colors_ls
                    
                edge_idx = random.randint(0, E - 1)
                old_color = colors[edge_idx]
                new_color = (old_color + 1 + random.randint(0, 2)) % 4
                colors[edge_idx] = new_color
                new_c6s = count_c6s(colors)
                
                if new_c6s <= best_c6s:
                    best_c6s = new_c6s
                    best_colors_ls = colors[:]
                else:
                    colors[edge_idx] = old_color
            
            if best_c6s == 0:
                return best_colors_ls
            return None

        for n in [15, 14, 13]:
            if time.time() > deadline - 5:
                break
            sol = _local_search(n, 5.0)
            if sol:
                best_n = n
                best_colors = sol
                break

    if best_n == 0:
        # Absolute fallback
        best_n = 4
        best_colors = [0, 1, 2, 3, 0, 1]

    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"n={res['n']}")

import time
import random
import math

def construct():
    # Deadline guard: 140 seconds to be safe within 150s limit
    deadline = time.time() + 140.0
    random.seed(42)

    # Seed solution for n=11
    def get_seed(n):
        # K11 split into five Hamiltonian cycles by circular distance 1..5
        return [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]

    best_n = 11
    best_colors = get_seed(11)
    
    # Data structures for current best coloring
    # adj[c][u] = set of neighbors of u in color c
    # common_counts[u][v][c] = number of common neighbors of u and v in color c
    # We only need common_counts for u < v
    
    def build_structures(colors, n):
        adj = [set() for _ in range(n)] # adj[u] will be a list of sets? No, adj[c][u]
        # Let's use adj[c][u] = set()
        adj = [[set() for _ in range(n)] for _ in range(5)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i].add(j)
                adj[c][j].add(i)
                idx += 1
        
        # Precompute common counts
        # common[u][v][c] for u < v
        # To save memory/time, we can compute on the fly or store in a dict/list
        # Since n is small (< 30), we can store a 3D array or dict
        # common_counts[u][v] is a list of 5 ints
        common_counts = [[None] * n for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                cc = [0] * 5
                for c in range(5):
                    nu = adj[c][u]
                    nv = adj[c][v]
                    cc[c] = len(nu & nv)
                common_counts[u][v] = cc
                common_counts[v][u] = cc
        return adj, common_counts

    adj, common_counts = build_structures(best_colors, best_n)

    def try_extend(n_curr, adj, common_counts, colors, deadline):
        """
        Try to extend coloring from n_curr to n_curr + 1.
        Returns (success, new_colors, new_adj, new_common_counts)
        """
        n_new = n_curr + 1
        # Edges to add: (n_new, i) for i in 0..n_curr-1
        # There are n_curr edges.
        
        # Initialize new edge colors
        new_edge_colors = [random.randint(0, 4) for _ in range(n_curr)]
        
        # Helper to calculate cost of a specific assignment of new_edge_colors
        def calc_cost(new_edge_colors):
            # Cost components:
            # 1. Violations between existing vertices u, v caused by new vertex n_new
            #    If color(n_new, u) == color(n_new, v) == c, and common_counts[u][v][c] == 1,
            #    then adding n_new makes common count 2 -> 1 violation.
            # 2. Violations between n_new and existing vertex u
            #    Common neighbors of n_new and u in color c:
            #    Set S_u_c = { w < n_curr | color(n_new, w) == c and w in adj[c][u] }
            #    Violations += binom(|S_u_c|, 2)
            
            cost = 0
            
            # Precompute for each color c, the set of vertices connected to n_new with color c
            # Sets: N_new[c] = { i | new_edge_colors[i] == c }
            N_new = [set() for _ in range(5)]
            for i in range(n_curr):
                N_new[new_edge_colors[i]].add(i)
                
            # Part 1: Pairs u < v in existing graph
            for u in range(n_curr):
                for v in range(u + 1, n_curr):
                    cu = new_edge_colors[u]
                    cv = new_edge_colors[v]
                    if cu == cv:
                        c = cu
                        if common_counts[u][v][c] == 1:
                            cost += 1
                            
            # Part 2: Pairs (n_new, u)
            for u in range(n_curr):
                # For each color c, count common neighbors
                # Common neighbors are w in N_new[c] such that w is in adj[c][u]
                # Note: w != u is guaranteed because w is neighbor of n_new, 
                # and we are counting neighbors of u. 
                # Wait, does w=u count? 
                # color(n_new, u) is some color c_u.
                # For a color c, we look at w such that color(n_new, w)=c.
                # If w=u, then color(n_new, u)=c.
                # Is u a neighbor of u? No.
                # So we just check intersection of N_new[c] and adj[c][u].
                for c in range(5):
                    # Intersection size
                    # N_new[c] is set of indices i where new_edge_colors[i] == c
                    # adj[c][u] is set of neighbors of u in color c
                    inter = N_new[c] & adj[c][u]
                    k = len(inter)
                    if k >= 2:
                        cost += math.comb(k, 2)
                        
            return cost

        current_cost = calc_cost(new_edge_colors)
        
        if current_cost == 0:
            # Success
            new_colors = colors + new_edge_colors
            # Update structures
            new_adj = [ [set(s) for s in adj[c]] for c in range(5) ] # Deep copy
            # Add new vertex n_new
            for c in range(5):
                new_adj[c].append(set()) # Placeholder for n_new in each color list? 
                # Actually, new_adj is list of lists of sets.
                # new_adj[c] is list of sets for vertices 0..n_new-1.
                # We need to append a set for vertex n_new.
            # Let's rebuild structures properly to be safe, or update incrementally.
            # Rebuilding is O(n^3) which is fine for n<30.
            new_adj, new_common_counts = build_structures(new_colors, n_new)
            return True, new_colors, new_adj, new_common_counts

        # Local Search
        best_cost = current_cost
        best_colors_local = new_edge_colors.copy()
        
        start_time = time.time()
        no_improve = 0
        max_no_improve = 1000
        
        while time.time() < deadline and time.time() - start_time < 5.0: # 5s per extension attempt
            if no_improve > max_no_improve:
                break
                
            # Pick random edge index
            i = random.randint(0, n_curr - 1)
            old_c = new_edge_colors[i]
            new_c = random.randint(0, 4)
            if new_c == old_c:
                continue
                
            new_edge_colors[i] = new_c
            new_cost = calc_cost(new_edge_colors)
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_colors_local = new_edge_colors.copy()
                no_improve = 0
                if best_cost == 0:
                    new_colors = colors + best_colors_local
                    new_adj, new_common_counts = build_structures(new_colors, n_new)
                    return True, new_colors, new_adj, new_common_counts
            else:
                # Revert
                new_edge_colors[i] = old_c
                no_improve += 1
                
        # If we didn't find 0, fail
        return False, None, None, None

    # Try to extend n
    for n_target in range(12, 29):
        if time.time() > deadline - 10:
            break
        
        success, new_colors, new_adj, new_common_counts = try_extend(
            best_n, adj, common_counts, best_colors, deadline - 5
        )
        
        if success:
            best_n = n_target
            best_colors = new_colors
            adj = new_adj
            common_counts = new_common_counts
        else:
            # If we fail at n, we might try again with different random seed?
            # Or just stop?
            # Given the difficulty, if we fail at 12, 13 is likely to fail too.
            # But let's try a few restarts for the current n?
            # The current loop structure tries each n once.
            # Let's add a small restart mechanism inside try_extend if it fails?
            # Or just break?
            # Let's break to save time, as the search space grows exponentially.
            break

    return {'n': best_n, 'colors': best_colors}

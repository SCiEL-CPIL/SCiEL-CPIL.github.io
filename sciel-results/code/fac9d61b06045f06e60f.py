import time
import sys
from random import Random
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0  # Safety margin

    def gen_instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        # Generate graph similar to seed
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_bipartite(edges):
        m = len(edges)
        if m == 0:
            return 0, [], []
        
        n = max(max(u, v) for u, v in edges) + 1
        adj = [set() for _ in range(n)]
        inc = [set() for _ in range(n)]
        for i, (u, v) in enumerate(edges):
            adj[u].add(v)
            adj[v].add(u)
            inc[u].add(i)
            inc[v].add(i)
        
        # conflict[i] is the set of edges j such that i and j cannot share a color
        # i and j conflict if they share a vertex OR are joined by an edge
        # Let's precompute conflict as bitsets for speed later, 
        # but for MWIS formulation on (edge, color) pairs:
        # Node (i, c) conflicts with (j, c) if i in conflict[i]
        # Node (i, c) conflicts with (i, c') if c != c' (same edge diff color)
        
        # We solve for a fixed k (number of colors).
        # We want to know if there is an independent set of size m in the 
        # bipartite graph G' where left nodes are edges 0..m-1 and right nodes 
        # are colors 0..k-1.
        # Edge (i, c) exists in G' if we CAN assign color c to edge i.
        # But wait, the constraint is:
        # 1. Each edge gets exactly one color.
        # 2. If edge i has color c, then no edge j in conflict[i] can have color c.
        
        # This is equivalent to finding a matching in a hypergraph? No.
        # It's a standard graph coloring problem where the "vertices" are edges of G,
        # and the "colors" are 0..k-1.
        # The conflict graph H has vertices 0..m-1. Edge (i,j) in H if conflict.
        # We need to check if H is k-colorable.
        
        # The MWIS approach mentioned in the prompt:
        # "Formulate ... as MWIS ... on the conflict graph by introducing a node for each edge-colour pair"
        # This creates a graph G'' with m*k nodes.
        # Node (i, c) represents "edge i has color c".
        # Two nodes (i, c) and (j, c') conflict if:
        #   - i == j and c != c' (an edge can't have two colors)
        #   - c == c' and i in conflict[i] (two conflicting edges can't have same color)
        #   - (i != j and c != c') -> No conflict? 
        #     Wait. If i has color c and j has color c' (c!=c'), they don't conflict 
        #     unless they are the same edge or have same color.
        #     So yes, the conflict graph G'' is the "conflict graph of the coloring assignment".
        #     We want to select exactly m nodes such that:
        #     1. For each edge i, exactly one (i, c) is selected.
        #     2. No two selected nodes conflict.
        #     3. The selected nodes form an independent set in G''.
        #     4. We want to minimize k.
        
        # Actually, simpler view:
        # We are looking for a k-coloring of the conflict graph H (size m).
        # This is NP-hard.
        # The prompt suggests MWIS on the "edge-color" graph.
        # Let's stick to the standard B&B for graph coloring (k-colorability) which is 
        # effectively searching for an independent set of size m in the k-partite 
        # complement? No.
        
        # Let's use a B&B for k-colorability of H.
        # Nodes in B&B tree are partial colorings.
        # State: current color of each vertex in H (0..k-1 or -1).
        # Constraints:
        #   - Adjacent vertices in H must have different colors.
        #   - We must assign all m vertices.
        
        # To make it fast, we use bitsets for available colors.
        # H has m vertices.
        # Adjacency list in H: H_adj[i] = set of j such that i, j conflict.
        
        # Let's build H_adj.
        H_adj = [0] * m  # bitset of neighbors in H
        for i in range(m):
            u, v = edges[i]
            # Neighbors in H are edges sharing a vertex with (u,v) or adjacent to (u,v)
            # i.e., edges incident to u or v, plus edges connecting (u, w) or (v, w) where w is neighbor?
            # Definition of strong edge coloring:
            # Two edges e1, e2 are adjacent in line graph if they share a vertex.
            # They are "joined by an edge" if there is an edge connecting an endpoint of e1 to an endpoint of e2.
            # So e1=(u,v), e2=(x,y).
            # Conflict if:
            # 1. {u,v} & {x,y} != empty (share vertex)
            # 2. (u,x) in E, (u,y) in E, (v,x) in E, or (v,y) in E
            
            # Let's compute this efficiently.
            # For edge i=(u,v):
            # Candidates are all edges incident to u or v.
            # Plus all edges (x,y) such that x in N(u) or x in N(v) or y in N(u) or y in N(v).
            # Actually, if e2=(x,y) and (u,x) in E, then e1 and e2 are joined by edge (u,x).
            
            # Let's just iterate all edges for small m (m < 300).
            # m is roughly n*d/2. For n=80, d=6, m ~ 240.
            # 240^2 = 57600 checks. Very fast.
            
            mask = 0
            for j in range(m):
                if i == j: continue
                x, y = edges[j]
                # Check share vertex
                if (u == x or u == y or v == x or v == y):
                    mask |= (1 << j)
                    continue
                # Check joined by edge
                # Need to check if any of (u,x), (u,y), (v,x), (v,y) is in edges
                # We can use a set of edges for O(1) lookup
                # But we are inside a loop. Let's prepare a set of edges.
            # Optimization: Precompute edge_set
            pass
        
        edge_set = set(edges)
        H_adj = [0] * m
        for i in range(m):
            u, v = edges[i]
            mask = 0
            for j in range(m):
                if i == j: continue
                x, y = edges[j]
                if (u == x or u == y or v == x or v == y):
                    mask |= (1 << j)
                else:
                    if (u, x) in edge_set or (u, y) in edge_set or (v, x) in edge_set or (v, y) in edge_set:
                        mask |= (1 << j)
            H_adj[i] = mask
        return m, H_adj

    def solve_k_colorability(m, H_adj, k, deadline):
        # Returns True if k-colorable, False otherwise.
        # Uses B&B with bitsets.
        # Order vertices by degree in H to reduce branching early.
        deg = [bin(H_adj[i]).count('1') for i in range(m)]
        order = sorted(range(m), key=lambda x: -deg[x])
        # Reverse order for assignment? No, assign highest degree first.
        
        # available_colors[v] is a bitmask of colors still available for vertex v
        # Initially (1<<k) - 1 for all v.
        # When we assign color c to vertex v, we remove c from available_colors[u] for all u in H_adj[v].
        
        # We need to track the state to backtrack.
        # Since m is small, we can copy the array.
        
        # Heuristic: Choose the uncolored vertex with the fewest available colors (MRV).
        
        # To speed up, we can use a recursive function with bitsets.
        # However, Python recursion and bit operations might be slow if k is large.
        # m is ~240. k is ~20-45.
        
        # Let's implement a simple B&B.
        # State: 
        #   uncolored: bitmask of vertices not yet colored
        #   colors: list of int, color assigned to each vertex (or -1)
        #   avail: list of int, bitmask of available colors for each vertex
        
        # Optimization: 
        # If the graph is sparse, we can use DSATUR heuristic.
        
        # Let's try a different approach: 
        # Since k is small (<= 45), we can try to find a coloring by backtracking.
        
        # To make it faster, we will use a "forward checking" approach.
        
        # Global best k found so far? No, we are checking a specific k.
        
        # Let's use a time limit inside the recursion.
        # If time.time() > deadline, raise an exception or return False (abort).
        
        # We'll use a global flag for timeout.
        timed_out = [False]
        
        # Precompute neighbor masks in the order of assignment?
        # No, H_adj is fixed.
        
        # MRV selection:
        # Find v in uncolored with min popcount(avail[v]).
        
        # We need to be careful with the number of states.
        # Worst case is k^m. But pruning is strong.
        
        # Let's implement.
        
        import sys
        sys.setrecursionlimit(10000)
        
        def backtrack(uncolored_mask, avail, colors):
            if timed_out[0]:
                return False
            
            if time.time() > deadline:
                timed_out[0] = True
                return False
                
            if uncolored_mask == 0:
                return True
            
            # Select vertex with MRV
            min_count = 1000
            best_v = -1
            # Iterate through uncolored vertices
            # We can iterate through bits of uncolored_mask
            mm = uncolored_mask
            while mm:
                # Get lowest set bit
                lsb = mm & -mm
                v = lsb.bit_length() - 1
                cnt = bin(avail[v]).count('1')
                if cnt == 0:
                    return False # Dead end
                if cnt < min_count:
                    min_count = cnt
                    best_v = v
                    if cnt == 1:
                        break # Can't get better than 1
                mm ^= lsb
            
            if best_v == -1:
                return False
            
            # Try all available colors for best_v
            v = best_v
            avail_colors = avail[v]
            
            # Order colors? Maybe try most constrained first? 
            # For now, just iterate.
            
            # New uncolored mask
            new_uncolored = uncolored_mask & ~(1 << v)
            
            # Iterate colors
            ac = avail_colors
            while ac:
                lsb = ac & -ac
                c = lsb.bit_length() - 1
                
                # Assign color c to v
                colors[v] = c
                
                # Update avail for neighbors
                # Neighbors of v are in H_adj[v]
                # We need to remove c from their avail sets
                # And check if any become 0
                
                neighbors = H_adj[v]
                # Only consider neighbors that are still uncolored
                active_neighbors = neighbors & new_uncolored
                
                valid = True
                # Save old values for backtracking
                # Since we modify avail in place, we need to restore.
                # To avoid copying the whole list, we can store the changes.
                # Or just copy the list if m is small? m=240. List copy is fast.
                # Let's copy.
                new_avail = avail[:]
                
                # Apply changes
                nn = active_neighbors
                while nn:
                    lsb_n = nn & -nn
                    u = lsb_n.bit_length() - 1
                    new_avail[u] &= ~lsb # Remove color c
                    if new_avail[u] == 0:
                        valid = False
                        break
                    nn ^= lsb_n
                
                if valid:
                    if backtrack(new_uncolored, new_avail, colors):
                        return True
                
                ac ^= lsb
            
            colors[v] = -1
            return False

        # Initial state
        initial_avail = [(1 << k) - 1] * m
        initial_colors = [-1] * m
        initial_uncolored = (1 << m) - 1
        
        # Symmetry breaking? 
        # The first vertex (highest degree) can be assigned color 0 WLOG?
        # Yes, we can fix the color of the first vertex to 0 to reduce branching by factor k.
        # Let's do that.
        
        v0 = order[0]
        initial_colors[v0] = 0
        initial_avail[v0] = 1 # Only color 0 available for v0 in this branch? 
        # Wait, if we fix v0 to 0, we must remove 0 from neighbors.
        
        # Let's just run the general backtracking, it's safer.
        # But for k=20, branching factor 20 is high.
        
        # Let's add a simple symmetry breaking: 
        # The color assigned to the first vertex (in order) must be 0.
        # This is valid because colors are symmetric.
        
        # Modify backtrack to handle this? 
        # Or just pre-assign.
        
        # Pre-assign v0 = 0
        if m > 0:
            v0 = order[0]
            initial_colors[v0] = 0
            # Remove 0 from neighbors
            for u in range(m):
                if u != v0 and (H_adj[v0] >> u) & 1:
                    initial_avail[u] &= ~1
                    if initial_avail[u] == 0:
                        return False
            initial_uncolored = (1 << m) - 1
            initial_uncolored &= ~(1 << v0)
            initial_avail[v0] = 0 # No colors left for v0 (already assigned)
            
            return backtrack(initial_uncolored, initial_avail, initial_colors)
        else:
            return True

    def find_min_colors(m, H_adj, deadline):
        # Binary search or linear increase?
        # Lower bound: max degree + 1? Or clique number?
        # Upper bound: greedy coloring result.
        
        # Greedy coloring to get UB
        col = [-1] * m
        # Order by degree
        deg = [bin(H_adj[i]).count('1') for i in range(m)]
        order = sorted(range(m), key=lambda x: -deg[x])
        
        for v in order:
            used = 0
            # Check neighbors
            nb = H_adj[v]
            # Only consider neighbors already colored? 
            # In greedy, we color in order. Neighbors not yet colored don't matter.
            # But H_adj[v] contains all neighbors.
            # We need to know which neighbors are already colored.
            # Since we iterate in order, we can check col[neighbor] != -1.
            
            # Optimization: Precompute neighbor list?
            # For now, iterate bits.
            mm = nb
            while mm:
                lsb = mm & -mm
                u = lsb.bit_length() - 1
                if col[u] != -1:
                    used |= (1 << col[u])
                mm ^= lsb
            
            c = 0
            while (used >> c) & 1:
                c += 1
            col[v] = c
            
        ub = max(col) + 1
        if m == 0:
            return 0, [0]*m # Dummy
            
        # Lower bound
        # Max degree + 1
        lb = max(deg) + 1
        
        # Try k from lb to ub
        # We want the smallest k.
        # We can binary search, but k is small. Linear is fine.
        
        best_k = ub
        best_col = col[:]
        
        # Try to improve
        # We check k-colorability for k in [lb, ub-1]
        # If k is not colorable, try k+1.
        
        # Actually, we want the MINIMUM k.
        # So we check k = lb, lb+1, ...
        # If k is colorable, that's our answer (since we start from LB).
        
        # But wait, is it possible that a smaller k is colorable but we missed it?
        # No, we start from LB. If LB is colorable, it's the min.
        # If not, we go up.
        
        # However, B&B for k-colorability can be slow for large k.
        # But we stop as soon as we find a valid k.
        
        # Time management:
        # We have deadline.
        # We should allocate time to the most promising k.
        # Usually, the true chromatic number is close to LB.
        
        # Let's try k = lb, lb+1, ...
        # But if lb is far from the answer, we waste time.
        
        # Strategy:
        # 1. Try k = lb.
        # 2. If fails, try k = lb + 1.
        # ...
        
        # To avoid wasting time on impossible k's, we can use a heuristic:
        # If k is too small, B&B will fail quickly (many dead ends).
        # If k is close, it might take longer to prove non-existence or find solution.
        
        # Let's just iterate.
        
        for k in range(lb, ub + 1):
            if time.time() > deadline - 1.0:
                break
                
            # Allocate slice of time
            # If we have many k's to try, give less time to early ones?
            # No, early ones are more likely to be the answer if LB is tight.
            
            slice_dl = min(deadline - 0.5, time.time() + 5.0) # 5s per k?
            
            # Reset timeout flag
            # solve_k_colorability uses global timed_out? 
            # Let's pass deadline and handle timeout inside.
            
            is_colorable = solve_k_colorability(m, H_adj, k, slice_dl)
            
            if is_colorable:
                # We found a coloring with k colors.
                # But solve_k_colorability only returns True/False.
                # We need the actual coloring.
                # Let's modify solve_k_colorability to return the coloring.
                # Or re-run to get it?
                # Better to modify.
                
                # For now, let's assume we can get the coloring.
                # I'll refactor solve_k_colorability to return col list or None.
                
                # Let's re-implement the check to return the coloring.
                
                # Actually, let's just return the best coloring found by greedy if B&B is too slow?
                # No, the goal is to minimize.
                
                # Let's refine solve_k_colorability to return the assignment.
                
                # I will rewrite the inner function in the final code to return the coloring.
                
                # For this draft, let's assume we get the coloring.
                # I need to capture the `colors` array when backtrack returns True.
                
                # Let's just return the greedy coloring for now to ensure validity, 
                # and use B&B only to verify if a smaller k is possible?
                # No, that doesn't help minimize.
                
                # Okay, I will implement the full version in the final code.
                
                # Break if found
                break
        
        # If we didn't find a better k, use greedy result
        # We need to return the coloring for the best k found.
        
        # This draft logic is a bit messy. Let's clean up in the final code.
        
        return ub, best_col

    # Refactored solve_k_colorability to return coloring
    def solve_k_colorability_v2(m, H_adj, k, deadline):
        timed_out = [False]
        
        import sys
        sys.setrecursionlimit(10000)
        
        # Order vertices
        deg = [bin(H_adj[i]).count('1') for i in range(m)]
        order = sorted(range(m), key=lambda x: -deg[x])
        
        def backtrack(uncolored_mask, avail, colors):
            if timed_out[0]:
                return None
            
            if time.time() > deadline:
                timed_out[0] = True
                return None
                
            if uncolored_mask == 0:
                return colors[:]
            
            # MRV
            min_count = 1000
            best_v = -1
            mm = uncolored_mask
            while mm:
                lsb = mm & -mm
                v = lsb.bit_length() - 1
                cnt = bin(avail[v]).count('1')
                if cnt == 0:
                    return None
                if cnt < min_count:
                    min_count = cnt
                    best_v = v
                    if cnt == 1:
                        break
                mm ^= lsb
            
            if best_v == -1:
                return None
            
            v = best_v
            avail_colors = avail[v]
            new_uncolored = uncolored_mask & ~(1 << v)
            
            ac = avail_colors
            while ac:
                lsb = ac & -ac
                c = lsb.bit_length() - 1
                
                colors[v] = c
                
                neighbors = H_adj[v]
                active_neighbors = neighbors & new_uncolored
                
                valid = True
                new_avail = avail[:]
                
                nn = active_neighbors
                while nn:
                    lsb_n = nn & -nn
                    u = lsb_n.bit_length() - 1
                    new_avail[u] &= ~lsb
                    if new_avail[u] == 0:
                        valid = False
                        break
                    nn ^= lsb_n
                
                if valid:
                    res = backtrack(new_uncolored, new_avail, colors)
                    if res is not None:
                        return res
                
                ac ^= lsb
            
            colors[v] = -1
            return None

        if m == 0:
            return []
            
        # Symmetry breaking: fix first vertex to color 0
        v0 = order[0]
        initial_colors = [-1] * m
        initial_colors[v0] = 0
        initial_avail = [(1 << k) - 1] * m
        initial_avail[v0] = 0
        
        # Remove 0 from neighbors
        for u in range(m):
            if u != v0 and (H_adj[v0] >> u) & 1:
                initial_avail[u] &= ~1
                if initial_avail[u] == 0:
                    return None
        
        initial_uncolored = (1 << m) - 1
        initial_uncolored &= ~(1 << v0)
        
        return backtrack(initial_uncolored, initial_avail, initial_colors)

    def solve_instance(edges, deadline):
        m = len(edges)
        if m == 0:
            return []
        
        m, H_adj = build_bipartite(edges)
        
        # Greedy for UB
        deg = [bin(H_adj[i]).count('1') for i in range(m)]
        order = sorted(range(m), key=lambda x: -deg[x])
        col_greedy = [-1] * m
        for v in order:
            used = 0
            nb = H_adj[v]
            mm = nb
            while mm:
                lsb = mm & -mm
                u = lsb.bit_length() - 1
                if col_greedy[u] != -1:
                    used |= (1 << col_greedy[u])
                mm ^= lsb
            c = 0
            while (used >> c) & 1:
                c += 1
            col_greedy[v] = c
            
        ub = max(col_greedy) + 1
        lb = max(deg) + 1
        
        best_col = col_greedy
        best_k = ub
        
        # Try to find smaller k
        # We try k from lb to ub-1
        # If we find a valid coloring, update best and break (since we go up from LB)
        
        # Time allocation:
        # We have deadline.
        # We try each k with a time slice.
        
        # If lb == ub, no need to search.
        
        for k in range(lb, ub):
            if time.time() > deadline - 1.0:
                break
            
            # Allocate time
            # If we have many k's, give less time to each?
            # Let's give 2 seconds per k.
            slice_dl = min(deadline - 0.5, time.time() + 2.0)
            
            res = solve_k_colorability_v2(m, H_adj, k, slice_dl)
            
            if res is not None:
                # Found valid coloring with k
                best_k = k
                best_col = res
                break # Since k is increasing, this is the minimum
        
        # Remap colors to 0..best_k-1
        present = sorted(set(best_col))
        remap = {old: new for new, old in enumerate(present)}
        return [remap[c] for c in best_col]

    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    out = []
    for idx, spec in enumerate(specs):
        edges = gen_instance(*spec)
        m = len(edges)
        left_after = len(specs) - 1 - idx
        inst_dl = min(DEADLINE - 3.0 - 2.0 * left_after, time.time() + 45.0)
        if inst_dl < time.time() + 5.0:
            inst_dl = min(DEADLINE - 1.0, time.time() + 5.0)
        try:
            out.append(solve_instance(edges, inst_dl))
        except Exception:
            # Fallback: Greedy
            m, H_adj = build_bipartite(edges)
            deg = [bin(H_adj[i]).count('1') for i in range(m)]
            order = sorted(range(m), key=lambda x: -deg[x])
            col = [-1] * m
            for v in order:
                used = 0
                nb = H_adj[v]
                mm = nb
                while mm:
                    lsb = mm & -mm
                    u = lsb.bit_length() - 1
                    if col[u] != -1:
                        used |= (1 << col[u])
                    mm ^= lsb
                c = 0
                while (used >> c) & 1:
                    c += 1
                col[v] = c
            present = sorted(set(col))
            remap = {old: new for new, old in enumerate(present)}
            out.append([remap[c] for c in col])
    return out


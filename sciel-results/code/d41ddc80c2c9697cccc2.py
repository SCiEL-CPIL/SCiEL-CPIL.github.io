import time
import sys
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    def instance(n, d, seed):
        rng = Random(seed)
        degree = [0] * n
        edges = set()
        # Generate graph
        # We need to generate a graph with max degree d.
        # The prompt implies a specific generator "bounded_degree_graph(n, d, seed)".
        # Standard approach: random pairing or random edges until degrees filled.
        # Let's try to match the likely generator: random edges, reject if degree exceeded.
        # To ensure connectivity or specific properties, usually random edges are fine for this task type.
        # However, "bounded_degree_graph" might imply a specific construction.
        # Given the previous code used `rng.randrange(n)` twice, I will stick to that but ensure we try enough times to fill degrees if possible, or just stop when no more valid edges can be added.
        # Actually, the previous code stopped after `4 * n * d` attempts. 
        # Let's use a more robust generation: try to add edges until we have a maximal set of edges with max degree d.
        # Or, simpler: just use the previous generation logic which seems to be the intended one based on the "failed program" context.
        
        # Re-implementing the generation from the failed program to ensure consistency with the verifier's expectation of "bounded_degree_graph".
        # The failed program used:
        # for _ in range(4 * n * d):
        #    u, v = rng.randrange(n), rng.randrange(n)
        #    if u != v and degree[u] < d and degree[v] < d and e not in edges:
        #       add
        # This produces a random graph with max degree <= d. It might not be connected, but that's fine.
        
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_strong_conflict(edges, adj):
        n_edges = len(edges)
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        # For each vertex, list of incident edge indices
        vert_to_edges = [[] for _ in range(len(adj))]
        for i, (u, v) in enumerate(edges):
            vert_to_edges[u].append(i)
            vert_to_edges[v].append(i)
            
        strong_adj = [0] * n_edges
        
        for i, (u, v) in enumerate(edges):
            mask = 0
            # Type 1: Share vertex u or v
            for j in vert_to_edges[u]:
                mask |= (1 << j)
            for j in vert_to_edges[v]:
                mask |= (1 << j)
                
            # Type 2: Connected by an edge in G
            # An edge (x, y) conflicts with (u, v) if there is an edge in G connecting {u,v} to {x,y}.
            # This happens if:
            # (u, x) in E, (u, y) in E, (v, x) in E, or (v, y) in E.
            # Note: If (u,x) is in E, then x is a neighbor of u.
            # So for each neighbor x of u, any edge incident to x conflicts with (u,v).
            # This includes (u,x) itself (already covered in Type 1) and others.
            
            for x in adj[u]:
                for j in vert_to_edges[x]:
                    mask |= (1 << j)
                    
            for x in adj[v]:
                for j in vert_to_edges[x]:
                    mask |= (1 << j)
                    
            strong_adj[i] = mask
        return strong_adj

    def solve_coloring(n_edges, strong_adj, max_colors, time_limit):
        # Order vertices by degree descending
        degrees = [bin(strong_adj[i]).count('1') for i in range(n_edges)]
        order = sorted(range(n_edges), key=lambda i: degrees[i], reverse=True)
        
        color = [-1] * n_edges
        # available_colors[i] = bitmask of colors available for vertex i
        available = [(1 << max_colors) - 1] * n_edges
        
        def backtrack(idx):
            if time.time() > time_limit:
                return False
            if idx == n_edges:
                return True
            
            i = order[idx]
            
            avail = available[i]
            if avail == 0:
                return False
            
            # Try each color
            while avail:
                c_bit = avail & (-avail)
                c = c_bit.bit_length() - 1
                avail -= c_bit
                
                color[i] = c
                # Update neighbors
                neighbors_mask = strong_adj[i]
                
                valid = True
                affected = []
                n_mask = neighbors_mask
                while n_mask:
                    nb_bit = n_mask & (-n_mask)
                    nb = nb_bit.bit_length() - 1
                    n_mask -= nb_bit
                    
                    if color[nb] == -1: # Uncolored
                        available[nb] &= ~c_bit
                        if available[nb] == 0:
                            valid = False
                            break
                        affected.append(nb)
                
                if valid:
                    if backtrack(idx + 1):
                        return True
                
                # Undo
                for nb in affected:
                    available[nb] |= c_bit
                color[i] = -1
                
            return False
        
        success = backtrack(0)
        if success:
            return color
        return None

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        n_edges = len(edges)
        if n_edges == 0:
            out.append([])
            continue
            
        strong_adj = build_strong_conflict(edges, adj)
        
        # Conjectured bound
        max_k = int(5 * d * d / 4)
        
        # We want to minimize colors.
        # Start from a reasonable lower bound or just try from max_k down?
        # No, we want the *minimum*. So we should try to find a small k.
        # But finding a valid coloring with k colors is hard if k is too small.
        # Strategy: Start with a greedy coloring to get an upper bound K_max.
        # Then binary search or linear search down for the minimum K.
        
        # Greedy Coloring (Largest First)
        def greedy_coloring(n_edges, strong_adj, adj_order):
            color = [-1] * n_edges
            for i in adj_order:
                used = set()
                mask = strong_adj[i]
                while mask:
                    nb_bit = mask & (-mask)
                    nb = nb_bit.bit_length() - 1
                    mask -= nb_bit
                    if color[nb] != -1:
                        used.add(color[nb])
                c = 0
                while c in used:
                    c += 1
                color[i] = c
            return color

        # Order by degree
        degrees = [bin(strong_adj[i]).count('1') for i in range(n_edges)]
        order = sorted(range(n_edges), key=lambda i: degrees[i], reverse=True)
        
        # Get an initial upper bound
        greedy_colors = greedy_coloring(n_edges, strong_adj, order)
        current_max = max(greedy_colors) + 1
        
        # Try to improve
        best_colors = greedy_colors
        best_k = current_max
        
        # Search for smaller k
        # We know k must be at least max_degree_strong + 1? No, that's a lower bound for proper coloring, 
        # but strong coloring is stricter. 
        # A lower bound is the size of the largest clique in the strong conflict graph.
        # Let's just iterate down from best_k - 1 to 1.
        
        found_better = False
        for k in range(best_k - 1, 0, -1):
            if time.time() > deadline - 2:
                break
            res = solve_coloring(n_edges, strong_adj, k, deadline - 1)
            if res is not None:
                best_colors = res
                best_k = k
                found_better = True
                # We could continue to find even smaller, but let's check time.
                # If we found a solution with k, we can try k-1.
                # The loop does this naturally.
        
        # If we didn't find anything better, best_colors is the greedy one.
        # If we found something, best_colors is the backtracking one.
        
        out.append(best_colors)

    return out

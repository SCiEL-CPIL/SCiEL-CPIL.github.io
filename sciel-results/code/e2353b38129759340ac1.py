import time
from random import Random

def construct():
    t0 = time.time()
    DEADLINE = t0 + 140.0  # Safety margin

    def gen_instance(n, d, seed):
        rng = Random(seed)
        degree = [0] * n
        edges = set()
        # Use a loop that attempts to add edges until we try enough times or reach a limit
        # The seed program uses 4*n*d attempts.
        attempts = 0
        max_attempts = 4 * n * d * 10 # Increase attempts to ensure connectivity/density if needed, but stick to seed logic first
        # Actually, the seed logic is:
        # for _ in range(4 * n * d): ...
        # I will stick to the seed's instance generation to ensure we are solving the *same* graphs.
        
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            if u == v: continue
            e = (min(u, v), max(u, v))
            if e in edges: continue
            if degree[u] < d and degree[v] < d:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict(edges):
        m = len(edges)
        # Map edge index to endpoints
        # conflict[i] is a set of edge indices j such that edge i and edge j cannot have the same color
        conflict = [set() for _ in range(m)]
        
        # Adjacency list of the original graph
        adj = [set() for _ in range(max(len(edges)*2, 1))] # Vertices are 0..n-1, n is small
        # Actually n is known from generation, but let's infer max vertex id
        max_v = 0
        for u, v in edges:
            if u > max_v: max_v = u
            if v > max_v: max_v = v
        adj = [set() for _ in range(max_v + 1)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Precompute neighbors for each edge
        # Edge i = (u, v)
        # Conflicts with:
        # 1. Edges sharing u or v
        # 2. Edges (x, y) where x in N(u) or N(v) and y in N(u) or N(v)?
        #    Specifically, edge (x,y) conflicts with (u,v) if:
        #    - {x,y} intersects {u,v} (covered by 1)
        #    - OR there is an edge in G connecting {u,v} to {x,y}.
        #      This means (u,x) in E or (u,y) in E or (v,x) in E or (v,y) in E.
        
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        for i, (u, v) in enumerate(edges):
            # 1. Share vertex
            for w in adj[u]:
                e2 = (min(u, w), max(u, w))
                if e2 in edge_to_idx:
                    j = edge_to_idx[e2]
                    if j != i:
                        conflict[i].add(j)
                e3 = (min(v, w), max(v, w)) # Wait, w is neighbor of u. Edge is (u,w).
                # The edges incident to u are (u, w) for w in adj[u].
                # So for each w in adj[u], the edge (u,w) conflicts with (u,v).
            for w in adj[v]:
                e2 = (min(v, w), max(v, w))
                if e2 in edge_to_idx:
                    j = edge_to_idx[e2]
                    if j != i:
                        conflict[i].add(j)
            
            # 2. Joined by an edge
            # For each neighbor x of u:
            #   For each neighbor y of x (excluding u):
            #      Edge (x, y) conflicts with (u, v) IF y is not u and (x,y) is not (u,v) or (v,u).
            #      Actually, if (x,y) shares a vertex with (u,v), it's already handled.
            #      We need to check if (x,y) is "joined" to (u,v) via the edge (u,x).
            #      This implies the path u-x-y has length 2.
            #      So edge (x,y) and edge (u,v) are at distance 2 in the line graph?
            #      In the line graph, L(G), edges are vertices.
            #      (u,v) is connected to (u,x) in L(G).
            #      (u,x) is connected to (x,y) in L(G).
            #      So (u,v) and (x,y) are at distance 2 in L(G).
            #      Strong coloring requires distance > 1 in L(G). So distance 2 is a conflict.
            
            for x in adj[u]:
                for y in adj[x]:
                    if y == u: continue
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_to_idx:
                        j = edge_to_idx[e2]
                        if j != i:
                            # Check if they share a vertex (already handled, but adding again is fine)
                            # Or if they are the same edge (handled by j!=i)
                            conflict[i].add(j)
                            
            for x in adj[v]:
                for y in adj[x]:
                    if y == v: continue
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_to_idx:
                        j = edge_to_idx[e2]
                        if j != i:
                            conflict[i].add(j)
                            
        # Symmetrize
        for i in range(m):
            for j in conflict[i]:
                conflict[j].add(i)
                
        return conflict

    def solve_instance(n, d, seed):
        edges = gen_instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        
        conflict = build_conflict(edges)
        
        # Greedy Coloring
        # Order edges by degree in conflict graph (descending) to make greedy tighter?
        # Or just random restarts.
        
        best_coloring = None
        best_num_colors = float('inf')
        
        rng = Random(seed + 1)
        
        # Try multiple random orderings
        num_restarts = 20
        for r in range(num_restarts):
            if time.time() > DEADLINE:
                break
                
            order = list(range(m))
            rng.shuffle(order)
            
            color = [-1] * m
            used_colors = set()
            
            for i in order:
                # Find available colors
                # A color c is available if no neighbor j in conflict[i] has color c
                forbidden = set()
                for j in conflict[i]:
                    if color[j] != -1:
                        forbidden.add(color[j])
                
                # Find smallest non-negative integer not in forbidden
                c = 0
                while c in forbidden:
                    c += 1
                color[i] = c
                used_colors.add(c)
            
            num_colors = len(used_colors)
            
            # Local Search: Try to reduce number of colors
            # We try to remove the highest color class
            # Iterate from max_color down to 0
            # If we can recolor all edges of color c using colors < c, we remove c.
            
            improved = True
            while improved and time.time() < DEADLINE:
                improved = False
                current_max = max(color) if color else -1
                for c in range(current_max, -1, -1):
                    # Get edges with color c
                    edges_in_c = [i for i in range(m) if color[i] == c]
                    if not edges_in_c:
                        continue
                    
                    # Try to recolor these edges to colors < c
                    # This is a small CSP.
                    # Variables: edges_in_c
                    # Domains: {0, ..., c-1}
                    # Constraints: For each pair (u, v) in edges_in_c, if u and v are in conflict, they must have different colors.
                    # Also, for each u in edges_in_c and v NOT in edges_in_c, if u and v are in conflict, color[u] != color[v].
                    
                    # Backtracking search
                    # Sort edges_in_c by number of conflicts with other edges_in_c (most constrained first)
                    # Or just fixed order.
                    
                    def backtrack(idx, assignment):
                        if idx == len(edges_in_c):
                            return True
                        e = edges_in_c[idx]
                        # Try colors 0 to c-1
                        for new_c in range(c):
                            # Check against other edges_in_c assigned so far
                            valid = True
                            for k in range(idx):
                                e2 = edges_in_c[k]
                                if e in conflict[e2]:
                                    if assignment[k] == new_c:
                                        valid = False
                                        break
                            if not valid:
                                continue
                            
                            # Check against edges NOT in edges_in_c
                            for j in conflict[e]:
                                if color[j] == new_c:
                                    # If j is in edges_in_c, it would have been checked above if assigned
                                    # If j is not in edges_in_c, its color is fixed (old color)
                                    # Wait, if j is in edges_in_c and not assigned yet, we don't know its color yet.
                                    # But we are only checking against *fixed* colors (old colors) and *assigned* new colors.
                                    # The old color of j is c.
                                    # If new_c == c, that's not allowed since we are trying to use < c.
                                    # So if color[j] == new_c, it must be that j is not in edges_in_c (since all in edges_in_c have old color c, and new_c < c).
                                    # So this check is valid.
                                    valid = False
                                    break
                            
                            if valid:
                                assignment[idx] = new_c
                                if backtrack(idx + 1, assignment):
                                    return True
                                assignment[idx] = -1
                        return False

                    assignment = [-1] * len(edges_in_c)
                    if backtrack(0, assignment):
                        # Apply new colors
                        for i, new_c in zip(edges_in_c, assignment):
                            color[i] = new_c
                        improved = True
                        # Continue checking lower colors or restart the loop?
                        # Since we removed a color, the max color might decrease.
                        # Break out of for loop and restart while loop to recalculate max?
                        # Or just continue to c-1.
                        break 
                        
            if num_colors < best_num_colors:
                best_num_colors = num_colors
                best_coloring = color[:]
                
        # Final check and return
        if best_coloring is None:
             # Fallback: simple greedy
             color = [-1] * m
             for i in range(m):
                 forbidden = set(color[j] for j in conflict[i] if color[j] != -1)
                 c = 0
                 while c in forbidden: c += 1
                 color[i] = c
             return color
             
        return best_coloring

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > DEADLINE - 10:
             # Not enough time, use simple greedy for remaining
             edges = gen_instance(*spec)
             m = len(edges)
             if m == 0:
                 results.append([])
                 continue
             conflict = build_conflict(edges)
             color = [-1] * m
             for i in range(m):
                 forbidden = set(color[j] for j in conflict[i] if color[j] != -1)
                 c = 0
                 while c in forbidden: c += 1
                 color[i] = c
             results.append(color)
        else:
             results.append(solve_instance(*spec))
             
    return results

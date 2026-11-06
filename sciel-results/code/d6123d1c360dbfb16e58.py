import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    out = []
    
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > deadline - 5:
            # Fallback if we run out of time
            edges = instance(*spec)
            out.append([0]*len(edges))
            continue
            
        edges = instance(*spec)
        n_edges = len(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Precompute conflict graph
        # Two edges conflict if they share a vertex or are joined by an edge.
        # Edge i = (u, v). Edge j = (x, y).
        # Conflict if:
        # 1. {u, v} & {x, y} != empty (share vertex)
        # 2. u-x in E, u-y in E, v-x in E, or v-y in E (joined by an edge)
        
        # For efficiency, we can represent edges as indices 0..n_edges-1
        # conflict[i] is a set of j such that i and j conflict.
        
        # Optimization: 
        # For a given edge (u, v), the conflicting edges are:
        # A. Edges incident to u or v.
        # B. Edges (x, y) such that x in N(u) and y in N(v) OR x in N(v) and y in N(u).
        # Note: If x in N(u) and y in N(v), then edge (x,y) conflicts with (u,v).
        # Also note that edges in A are included in B if we are careful, but it's safer to union them.
        
        conflict = [set() for _ in range(n_edges)]
        
        # Map vertex to incident edges
        vert_to_edges = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            vert_to_edges[u].append(i)
            vert_to_edges[v].append(i)
            
        for i, (u, v) in enumerate(edges):
            # 1. Shared vertices
            for e_idx in vert_to_edges[u]:
                if e_idx != i:
                    conflict[i].add(e_idx)
            for e_idx in vert_to_edges[v]:
                if e_idx != i:
                    conflict[i].add(e_idx)
                    
            # 2. Joined by an edge
            # For each neighbor x of u, and each neighbor y of v, if (x,y) is an edge, it conflicts.
            # We can iterate through neighbors of u and neighbors of v.
            # Since degree is bounded, this is O(d^2) per edge.
            for x in adj[u]:
                for y in adj[v]:
                    # Check if (x,y) is an edge
                    # We need a fast lookup for edges.
                    # Let's create a set of frozensets or sorted tuples for O(1) lookup.
                    pass
            
        # Let's rebuild conflict with a lookup table for edges
        edge_set = set(edges)
        
        for i, (u, v) in enumerate(edges):
            # Clear previous if we were reusing, but here we are building fresh
            # Actually, the loop above was incomplete. Let's do it properly.
            
            # 1. Shared vertices
            for e_idx in vert_to_edges[u]:
                if e_idx != i:
                    conflict[i].add(e_idx)
            for e_idx in vert_to_edges[v]:
                if e_idx != i:
                    conflict[i].add(e_idx)
            
            # 2. Joined by an edge
            for x in adj[u]:
                for y in adj[v]:
                    if x == y: continue
                    e_xy = (min(x, y), max(x, y))
                    if e_xy in edge_set:
                        # Find index of e_xy
                        # This is O(1) if we have a dict, but we don't have index lookup yet.
                        # Let's create edge_to_idx
                        pass
        
        # Create edge_to_idx
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        for i, (u, v) in enumerate(edges):
            # 1. Shared vertices
            for e_idx in vert_to_edges[u]:
                if e_idx != i:
                    conflict[i].add(e_idx)
            for e_idx in vert_to_edges[v]:
                if e_idx != i:
                    conflict[i].add(e_idx)
            
            # 2. Joined by an edge
            for x in adj[u]:
                for y in adj[v]:
                    if x == y: continue
                    e_xy = (min(x, y), max(x, y))
                    if e_xy in edge_to_idx:
                        j = edge_to_idx[e_xy]
                        if j != i:
                            conflict[i].add(j)
                            
        # Now we have the conflict graph.
        # We want to cover all vertices (edges of original graph) with minimum number of cliques.
        # A clique in the conflict graph corresponds to a set of edges that are pairwise conflicting.
        # Wait. The task says: "Each colour class must be an induced matching".
        # An induced matching is a set of edges such that no two edges share a vertex and no two edges are joined by an edge.
        # This means that for any two edges in the same color class, they do NOT conflict.
        # So, a color class is a set of vertices in the conflict graph that form an INDEPENDENT SET.
        # Therefore, we are looking for a minimum number of independent sets to cover all vertices.
        # This is the Graph Coloring problem on the conflict graph!
        # The number of colors is the chromatic number of the conflict graph.
        
        # So, the "Clique Cover" mentioned in the directive is actually "Independent Set Cover" which is equivalent to "Graph Coloring".
        # The directive said: "Model the strong edge colouring as a minimum clique cover problem... by first computing a maximal set of pairwise non-conflicting edges (induced matchings)".
        # Pairwise non-conflicting edges = Independent Set in Conflict Graph.
        # Covering all edges with these sets = Partition into Independent Sets = Coloring.
        
        # So we need to color the conflict graph.
        # The conflict graph has n_edges vertices.
        # We can use a heuristic coloring algorithm.
        
        # Strategy:
        # 1. Compute degree of each vertex in conflict graph.
        # 2. Use DSATUR (Degree of Saturation) or similar greedy heuristic.
        # 3. Since n_edges is small (max ~ 80*6/2 = 240), we can afford some search.
        # 4. Try multiple random orderings and keep the best.
        
        def color_conflict_graph(conflict, n):
            # DSATUR
            # colors[i] is the color assigned to vertex i, -1 if uncolored
            colors = [-1] * n
            # sat[i] is the set of colors used by neighbors of i
            sat = [set() for _ in range(n)]
            
            # Precompute degrees for tie-breaking
            degrees = [len(conflict[i]) for i in range(n)]
            
            uncolored = set(range(n))
            
            # To make it faster, we can use a priority queue or just scan.
            # Scanning O(n) per step, total O(n^2). n=240 -> 57600 ops. Very fast.
            
            while uncolored:
                # Find vertex with max saturation degree, tie-break by max degree
                best_v = -1
                best_sat = -1
                best_deg = -1
                for v in uncolored:
                    s = len(sat[v])
                    d = degrees[v]
                    if s > best_sat or (s == best_sat and d > best_deg):
                        best_sat = s
                        best_deg = d
                        best_v = v
                        
                # Assign lowest available color
                used_colors = sat[best_v]
                c = 0
                while c in used_colors:
                    c += 1
                colors[best_v] = c
                
                # Update neighbors
                uncolored.remove(best_v)
                for u in conflict[best_v]:
                    if u in uncolored:
                        sat[u].add(c)
                        
            return colors
        
        # To improve, we can shuffle the vertices or use a randomized DSATUR.
        # Or simply run DSATUR a few times with different tie-breaking randomizations.
        
        best_colors = None
        best_num_colors = float('inf')
        
        # Run DSATUR with random tie-breaking
        # We can modify the selection step to pick randomly among top candidates.
        
        def color_conflict_graph_random(conflict, n, rng):
            colors = [-1] * n
            sat = [set() for _ in range(n)]
            degrees = [len(conflict[i]) for i in range(n)]
            uncolored = set(range(n))
            
            while uncolored:
                # Find max saturation
                max_sat = max(len(sat[v]) for v in uncolored)
                candidates = [v for v in uncolored if len(sat[v]) == max_sat]
                
                # Tie-break by degree, but add some randomness
                # Sort candidates by degree descending, then pick from top K randomly?
                # Or just pick random from candidates if we want pure random.
                # Better: Sort by degree, then pick randomly from the top 10% or something.
                # Let's just sort by degree and pick the first one, but shuffle the input order initially?
                # No, DSATUR is deterministic given the graph.
                # To introduce randomness, we can shuffle the list of candidates before picking the one with max degree?
                # Actually, standard DSATUR is quite strong.
                
                # Let's try a variation: Randomized Greedy
                # Pick a random uncolored vertex? No, that's bad.
                
                # Let's stick to DSATUR but maybe try different initial orderings?
                # DSATUR doesn't depend on initial ordering much, it's adaptive.
                
                # Let's just run standard DSATUR. It's usually very good.
                # If we want to improve, we can do local search afterwards.
                
                # For now, let's implement a simple local search to reduce colors.
                # 1. Get a coloring.
                # 2. Try to recolor a vertex to an existing color if possible.
                # 3. Try to merge two color classes if possible? No, that's hard.
                # 4. Kempe chains?
                
                # Let's just return the DSATUR result for now.
                # We can run it multiple times with slight perturbations.
                
                # Perturbation: Randomly shuffle the tie-breaking.
                candidates.sort(key=lambda v: degrees[v], reverse=True)
                # Pick from the top few
                k = min(5, len(candidates))
                best_v = candidates[rng.randrange(k)]
                
                used_colors = sat[best_v]
                c = 0
                while c in used_colors:
                    c += 1
                colors[best_v] = c
                
                uncolored.remove(best_v)
                for u in conflict[best_v]:
                    if u in uncolored:
                        sat[u].add(c)
                        
            return colors

        # Run multiple times
        rng = random.Random(42)
        for _ in range(100): # 100 restarts
            if time.time() > deadline - 5:
                break
            cols = color_conflict_graph_random(conflict, n_edges, rng)
            num_colors = max(cols) + 1 if cols else 0
            if num_colors < best_num_colors:
                best_num_colors = num_colors
                best_colors = cols
                
        if best_colors is None:
            # Fallback
            best_colors = [0]*n_edges
            
        out.append(best_colors)
        
    return out


import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (evaluator kills at 150s)
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(all_edges)}
    num_edges = len(all_edges)
    
    # 1. Pre-compute all Q3 embeddings (clauses)
    # A Q3 is defined by 8 vertices. 
    # We identify a Q3 by its structure: 8 vertices, 12 edges.
    # To find all subgraphs isomorphic to Q3:
    # Iterate over all subsets of 8 vertices.
    # Check if the induced subgraph is exactly Q3.
    # This is expensive (C(16,8) = 12870 subsets).
    # For each subset, check if it's a Q3.
    # A graph on 8 vertices is Q3 if:
    # - It has 12 edges.
    # - It is 3-regular.
    # - It is bipartite with parts of size 4.
    # - (Actually, any 3-regular bipartite graph on 8 vertices is Q3? No, K4,4 is 4-regular. 
    #   The only 3-regular bipartite graph on 8 vertices is Q3? 
    #   Let's check. Q3 is the unique 3-regular bipartite graph on 8 vertices? 
    #   Yes, up to isomorphism, there is only one: the cube.
    #   Wait, is that true? 
    #   Number of 3-regular bipartite graphs on 8 vertices.
    #   Let's just check if it is isomorphic to Q3 via adjacency matrix or simpler invariants.
    #   Simpler: Q3 has 4 disjoint 4-cycles? No.
    #   Q3 has 6 square faces.
    #   Let's use a canonical check: 
    #   1. 12 edges.
    #   2. Degrees all 3.
    #   3. Bipartite.
    #   4. Diameter 3.
    #   Actually, let's just hardcode the Q3 structure check using bitmasks or simple graph properties.
    #   Given the small size, we can check if the induced subgraph is isomorphic to Q3.
    #   Since we only have 12870 subsets, we can afford a slightly heavier check.
    
    q3_embeddings = []
    
    # Helper to check if a set of 8 vertices induces a Q3
    def is_q3(vertices):
        # vertices is a tuple of 8 ints
        v_set = set(vertices)
        # Build local adjacency
        # Map vertices to 0..7
        verts = list(vertices)
        # Count edges
        edge_count = 0
        adj_local = {v: set() for v in verts}
        for u in verts:
            for w in verts:
                if u < w:
                    if (u, w) in edge_index: # This is wrong, edge_index is for global edges
                        # We need to check if the edge exists in the COMPLETE graph?
                        # No, we are checking if the *subset* forms a Q3 in the *complete graph*?
                        # No, the "clause" is: NOT (all 12 edges of this specific Q3 embedding are present).
                        # So we need to identify all *potential* Q3s in the complete graph K16.
                        # Yes, so we check if the induced subgraph on these 8 vertices in K16 is Q3.
                        # Since K16 has all edges, the induced subgraph is always K8?
                        # NO. A "subgraph isomorphic to Q3" means there exists a mapping of Q3 vertices to these 8 vertices 
                        # such that all Q3 edges are present in the graph.
                        # It does NOT require the induced subgraph to be exactly Q3.
                        # It requires that the 12 edges of Q3 are a subset of the graph's edges.
                        # So, for a fixed set of 8 vertices, how many ways can we map Q3 to them?
                        # This is the number of subgraphs isomorphic to Q3 contained in these 8 vertices.
                        # But a "clause" in Max-SAT usually forbids a specific assignment.
                        # Here, a "clause" is: "Not (Edge 1 AND Edge 2 AND ... AND Edge 12)".
                        # Each specific set of 12 edges forming a Q3 is a forbidden pattern.
                        # So we need to list all sets of 12 edges in K16 that form a Q3.
                        
                        pass
        
        # Correct approach:
        # Iterate over all 8-vertex subsets.
        # For each subset, check if it contains a Q3 subgraph.
        # Actually, it's easier to iterate over all possible embeddings of Q3 into 16 vertices.
        # Number of embeddings: 16 * 15 * 14 * ... / automorphisms?
        # Simpler: 
        # A Q3 is determined by a bijection from {0,1}^3 to 8 distinct vertices.
        # Total bijections: 16P8 = 16! / 8! = 518,918,400.
        # This is too many to iterate directly.
        
        # Better approach:
        # Identify Q3s by their edge sets.
        # Two embeddings are the same clause if they use the same 12 edges.
        # How many distinct edge sets of size 12 form a Q3 in K16?
        # This is the number of subgraphs of K16 isomorphic to Q3.
        # We can find them by:
        # 1. Pick 8 vertices.
        # 2. Check if the induced subgraph (which is K8) contains a Q3? 
        #    Wait, the graph we are building is a subgraph of K16.
        #    The constraint is: For ANY set of 12 edges that forms a Q3, we cannot have ALL of them present.
        #    So we need the list of all edge sets E_Q3 such that (V(E_Q3), E_Q3) is isomorphic to Q3.
        
        # Let's generate them by iterating over vertex subsets and checking isomorphism to Q3.
        # For a fixed set of 8 vertices S, the number of subgraphs of K[S] isomorphic to Q3 is the number of ways to map Q3 to S.
        # If S induces K8, it contains many Q3s.
        # We need to list each distinct set of 12 edges.
        
        # Algorithm:
        # 1. Iterate over all 8-vertex subsets S.
        # 2. For each S, find all subgraphs of K[S] isomorphic to Q3.
        #    Since K[S] is complete, any set of 12 edges forming a Q3 is valid.
        #    How many such sets are there in K8?
        #    The number of labeled Q3s in 8 vertices is 8! / 48 (automorphisms of Q3) = 40320 / 48 = 840.
        #    So for each S, there are 840 distinct edge sets (Q3s) contained in S.
        #    Total Q3s in K16: C(16,8) * 840 = 12870 * 840 = 10,810,800.
        #    This is ~10 million clauses. Storing them might be heavy but manageable?
        #    10M clauses * 12 edges each. 
        #    We can store them as bitmasks or sorted tuples of edge indices.
        #    10M * 12 * 4 bytes (if int) = 480 MB. Too much?
        #    Maybe we can be smarter.
        
        # Alternative: 
        # We don't need to store ALL clauses if we can check violations on the fly.
        # But the "Max-SAT" strategy relies on knowing which clauses are violated.
        # A clause is violated if all 12 edges are present.
        # We can iterate over all 8-vertex subsets S (12870 of them).
        # For a current graph G, S "contains a Q3" if there exists a mapping of Q3 to S such that all edges are in G.
        # This is equivalent to checking if G[S] (induced subgraph) contains a Q3.
        # Since G is sparse (we are maximizing edges but avoiding Q3), G[S] is a subgraph of K8.
        # Checking if G[S] contains a Q3 is fast if we precompute the 840 edge sets for each S?
        # No, we can just check: Does G[S] have a subgraph isomorphic to Q3?
        # Since G[S] has at most 12 edges (if it's Q3-free, it has <12? No, it can have more but no Q3).
        # Actually, if G[S] has 12 or more edges, it might contain a Q3.
        # The fastest way to check "does G contain a Q3 on vertices S" is to check if any of the 840 Q3-edge-sets is a subset of G's edges restricted to S.
        
        # Let's precompute the 840 edge masks for a canonical Q3 on vertices 0..7.
        # Then for any S, we can map 0..7 to S and check.
        
        # However, 12870 subsets * 840 checks * (check 12 edges) might be too slow per step in local search.
        # 12870 * 840 = 10.8M checks per "find violation" step.
        # If we do this 100 times, that's 1B operations. Too slow in Python.
        
        # We need a faster violation check.
        # Observation: A Q3 is a 3-regular bipartite graph on 8 vertices.
        # If G contains a Q3, it must contain a 3-regular subgraph on 8 vertices?
        # Not necessarily, the Q3 vertices in G might have higher degree in G.
        # But the 12 edges of the Q3 must be present.
        
        # Let's try a different search strategy that avoids precomputing 10M clauses.
        # Current program uses "find one Q3, remove edge".
        # My "Max-SAT" idea: "Remove edge in MOST Q3s".
        # To do this, we need to count how many Q3s contain each edge.
        # This is the "edge frequency" in the set of violated Q3s.
        # If we can't list all violated Q3s, we can't count frequencies.
        
        # Maybe we can approximate?
        # Or, we can just use the current "find one Q3" but choose the edge to remove more intelligently.
        # Instead of random edge from the found Q3, remove the edge that is in the most *potential* Q3s?
        # Or, remove the edge that minimizes the number of new Q3s created? (Not applicable, we are removing).
        
        # Let's stick to the current design family but improve the "remove" heuristic.
        # Current: Random edge from found Q3.
        # Improvement: From the found Q3, remove the edge with the highest "degree" in the current graph?
        # Or, remove the edge that is part of the most triangles? (Q3s contain triangles? No, Q3 is bipartite, no triangles).
        # Remove the edge that is in the most 4-cycles? Q3 contains 4-cycles.
        
        # Actually, let's look at the "Max-SAT" directive again.
        # "Model as Max-SAT... greedy local search... removing edges that violate the most clauses".
        # If we can't afford 10M clauses, maybe we can sample clauses?
        # Or, we can define "violating clauses" more locally.
        
        # Let's try a different approach within the "search" family:
        # Start with a dense graph.
        # While Q3 exists:
        #   Find a Q3.
        #   Instead of removing a random edge, remove the edge in the Q3 that has the highest degree in the graph.
        #   Why? High-degree edges are more "expensive" to keep if they contribute to many structures.
        #   Wait, we want to KEEP as many edges as possible.
        #   Removing a high-degree edge loses more "potential" edges?
        #   Actually, removing a high-degree edge is "more disruptive".
        #   We want to remove an edge that is "least useful".
        #   An edge is less useful if it is in few Q3s.
        #   But we only found ONE Q3.
        
        # Let's try: Remove the edge in the found Q3 that has the LOWEST degree in the graph.
        # Rationale: Low-degree edges are "sparser" connections. Removing them might break fewer other potential structures (though we don't know which ones).
        # Actually, if an edge is in a Q3, and we remove it, we break that Q3.
        # If the edge has low degree, it's likely not in many other Q3s? (Heuristic).
        
        # Let's implement this refinement.
        
    # 2. Implementation of the refined heuristic
    
    all_edges_set = set(all_edges)
    
    def build_adj(edge_set):
        adj = {i: set() for i in range(n)}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def find_q3(graph_adj):
        # Same as before, but we return the Q3 edges
        for v in range(n):
            neighbors = list(graph_adj.get(v, set()))
            if len(neighbors) < 3:
                continue
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    for k in range(j + 1, len(neighbors)):
                        a, b, c = neighbors[i], neighbors[j], neighbors[k]
                        # Check pairwise non-adjacency of a,b,c
                        if (a, b) in graph_adj or (b, a) in graph_adj: continue
                        if (a, c) in graph_adj or (c, a) in graph_adj: continue
                        if (b, c) in graph_adj or (c, b) in graph_adj: continue
                        
                        na = graph_adj.get(a, set())
                        nb = graph_adj.get(b, set())
                        nc = graph_adj.get(c, set())
                        
                        forbidden = {v, a, b, c}
                        xs = (na & nb) - forbidden
                        ys = (na & nc) - forbidden
                        zs = (nb & nc) - forbidden
                        
                        for x in xs:
                            for y in ys:
                                for z in zs:
                                    if x == y or y == z or x == z: continue
                                    if (x, y) in graph_adj or (y, x) in graph_adj: continue
                                    if (x, z) in graph_adj or (z, x) in graph_adj: continue
                                    if (y, z) in graph_adj or (z, y) in graph_adj: continue
                                    
                                    nx = graph_adj.get(x, set())
                                    ny = graph_adj.get(y, set())
                                    nz = graph_adj.get(z, set())
                                    ws = (nx & ny & nz) - forbidden - {x, y, z}
                                    
                                    for w in ws:
                                        if (w, a) in graph_adj or (a, w) in graph_adj: continue
                                        if (w, b) in graph_adj or (b, w) in graph_adj: continue
                                        if (w, c) in graph_adj or (c, w) in graph_adj: continue
                                        
                                        q3_edges = [
                                            (v, a), (v, b), (v, c),
                                            (a, x), (b, x),
                                            (a, y), (c, y),
                                            (b, z), (c, z),
                                            (x, w), (y, w), (z, w)
                                        ]
                                        return q3_edges
        return None

    def remove_edge_heuristic(graph_adj, q3_edges):
        # Heuristic: Remove the edge with the lowest sum of degrees of its endpoints
        # Or just the lowest degree of the endpoints?
        # Let's use: min(deg(u) + deg(v))
        best_edge = None
        min_score = float('inf')
        for u, v in q3_edges:
            score = len(graph_adj[u]) + len(graph_adj[v])
            if score < min_score:
                min_score = score
                best_edge = (u, v)
        
        # Normalize
        if best_edge[0] > best_edge[1]:
            best_edge = (best_edge[1], best_edge[0])
        return best_edge

    best_edges = []
    best_count = 0
    random.seed(42)
    
    for restart in range(100): # Increase restarts if fast
        if time.time() > deadline - 5:
            break
            
        # Start with a slightly denser random graph?
        # Current: p=0.4 (~48 edges).
        # Let's try p=0.5 (~60 edges) and remove.
        current_edges = set()
        for e in all_edges:
            if random.random() < 0.5:
                current_edges.add(e)
                
        adj = build_adj(current_edges)
        
        # Phase 1: Remove Q3s
        steps = 0
        while steps < 2000:
            if time.time() > deadline - 5:
                break
            q3_edges = find_q3(adj)
            if q3_edges is None:
                break
            e_to_remove = remove_edge_heuristic(adj, q3_edges)
            if e_to_remove in current_edges:
                current_edges.remove(e_to_remove)
                u, v = e_to_remove
                adj[u].remove(v)
                adj[v].remove(u)
            else:
                # Bug safety: if not found, break to avoid infinite loop
                break
            steps += 1
            
        # Phase 2: Hill Climbing (Add edges)
        missing = [e for e in all_edges if e not in current_edges]
        random.shuffle(missing)
        
        for e in missing:
            if time.time() > deadline - 5:
                break
            u, v = e
            current_edges.add(e)
            adj[u].add(v)
            adj[v].add(u)
            
            q3_edges = find_q3(adj)
            
            if q3_edges is not None:
                # Check if e is in the Q3
                e_norm = (u, v) if u < v else (v, u)
                contains_e = False
                for qe in q3_edges:
                    if (qe[0] == u and qe[1] == v) or (qe[0] == v and qe[1] == u):
                        contains_e = True
                        break
                
                if contains_e:
                    current_edges.remove(e)
                    adj[u].remove(v)
                    adj[v].remove(u)
                else:
                    # Should not happen, but remove for safety
                    current_edges.remove(e)
                    adj[u].remove(v)
                    adj[v].remove(u)
            
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = list(current_edges)
            
    return [[u, v] for u, v in best_edges]

# EVOLVE-BLOCK-END

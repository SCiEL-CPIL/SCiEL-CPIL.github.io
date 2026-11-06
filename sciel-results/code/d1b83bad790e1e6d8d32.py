import itertools
import time
import random

def construct():
    n = 18
    # Paley graph of order 17
    residues = {1, 2, 4, 8, 9, 13, 15, 16}
    
    # Build adjacency for Paley(17)
    adj = [[False] * 17 for _ in range(17)]
    edges_paley = []
    for u in range(17):
        for v in range(u + 1, 17):
            diff = (v - u) % 17
            if diff in residues:
                adj[u][v] = True
                adj[v][u] = True
                edges_paley.append([u, v])
                
    # Precompute triangles in Paley(17)
    triangles = []
    for u in range(17):
        for v in range(u + 1, 17):
            if adj[u][v]:
                for w in range(v + 1, 17):
                    if adj[u][w] and adj[v][w]:
                        triangles.append((u, v, w))
                        
    # Find all triangle-free subsets of size 8
    tf_sets_8 = []
    for combo in itertools.combinations(range(17), 8):
        # Check if combo is triangle-free
        is_tf = True
        for t in triangles:
            if t[0] in combo and t[1] in combo and t[2] in combo:
                is_tf = False
                break
        if is_tf:
            tf_sets_8.append(combo)
            
    # We want to find a set N of vertices such that:
    # 1. N is triangle-free (in Paley)
    # 2. For every T in tf_sets_8, T \cap N contains at least one edge of Paley.
    #    This ensures that T is not triangle-free in the new graph (since the edge in T \cap N makes it non-triangle-free? No.)
    #    Wait. If we add edges from 17 to N, then for T to be triangle-free in the new graph, T must not contain a triangle.
    #    The new triangles can only involve vertex 17.
    #    A triangle involving 17 is (17, u, v) where u, v are in N and adj[u][v] is True.
    #    So if T contains 17, and T contains u, v in N such that adj[u][v], then (17, u, v) is a triangle in T.
    #    So T is NOT triangle-free.
    #    We want to ensure that NO T in tf_sets_8 is triangle-free in the new graph.
    #    So for every T in tf_sets_8, we need (17, u, v) to be a triangle in T for some u, v in N.
    #    This requires 17 in T. But 17 is not in any T in tf_sets_8 (T is subset of 0..16).
    #    So adding edges from 17 to N does NOT create any triangle in T (since 17 is not in T).
    #    Therefore, T remains triangle-free.
    #    So this approach is WRONG.
    
    # Correct approach:
    # We want to add edges to the Paley graph (on 17 vertices) such that the largest triangle-free induced subgraph is minimized.
    # The current largest is 8. We want to reduce it to 7 or less.
    # To reduce it, we need to add edges such that every 8-vertex subset that was triangle-free now contains a triangle.
    # Let E be the set of edges we add.
    # For every T in tf_sets_8, T must contain a triangle in (Paley + E).
    # Since T was triangle-free in Paley, the new triangle must use at least one edge from E.
    # So for every T in tf_sets_8, there exists a triangle (u,v,w) in T such that at least one of (u,v), (v,w), (u,w) is in E.
    # We want to find a set E of edges to add (that does not create K4) such that this holds.
    # Also, adding edges must not create K4.
    
    # Let's search for E.
    # E is a subset of non-edges of Paley.
    # Non-edges of Paley(17):
    non_edges = []
    for u in range(17):
        for v in range(u + 1, 17):
            if not adj[u][v]:
                non_edges.append((u, v))
                
    # We want to add edges from non_edges.
    # Constraint: No K4 in the new graph.
    # A K4 is a set of 4 vertices where all 6 pairs are edges.
    # Since Paley is K4-free, any new K4 must use at least one new edge.
    
    # Let's try to find E using greedy or backtracking.
    # For each T in tf_sets_8, we need to "hit" it with a new triangle.
    # A new triangle in T is a triple (u,v,w) in T where at least one edge is new.
    # Since T is triangle-free in Paley, no triple in T has 2 or more edges in Paley.
    # So a new triangle in T must have exactly 1 edge in Paley and 2 new edges? No.
    # If (u,v) is in Paley, then (u,w) and (v,w) must be new.
    # Or (u,v) is new, and (u,w), (v,w) are in Paley? But then (u,v,w) would be a triangle in Paley (since (u,w), (v,w) are in Paley and (u,v) is in Paley? No, (u,v) is new).
    # Wait. If (u,w) and (v,w) are in Paley, and (u,v) is new, then (u,v,w) is a triangle in the new graph.
    # So a new triangle in T can be:
    # 1. (u,v) in Paley, (u,w) new, (v,w) new.
    # 2. (u,v) new, (u,w) in Paley, (v,w) in Paley.
    # 3. (u,v) new, (u,w) new, (v,w) in Paley.
    # 4. All three new.
    
    # But T is triangle-free in Paley, so no triple has 2 or more Paley edges.
    # So cases 2 and 3 are impossible (they require 2 Paley edges).
    # So only cases 1 and 4 are possible.
    # Case 1: 1 Paley edge + 2 new edges.
    # Case 4: 3 new edges.
    
    # So for each T, we need to add edges such that there is a triple (u,v,w) in T with:
    # - (u,v) in Paley, and (u,w), (v,w) in E.
    # OR
    # - (u,v), (u,w), (v,w) all in E.
    
    # Let's search for E.
    # We can try to add edges one by one, and check if all T are hit.
    # We also need to check K4-freeness.
    
    start_time = time.time()
    deadline = 140.0
    
    best_E = None
    found = False
    
    # Candidate edges: non_edges
    # We want to find a small E? Or just any E?
    # Let's try to find E with minimal size? Or just any valid E.
    # Greedy: pick an edge that hits the most T's.
    
    # For each non-edge e, count how many T it can help hit.
    # This is complex. Let's just do random search.
    
    random.seed(42)
    
    for _ in range(500):
        if time.time() > deadline:
            break
            
        E = set()
        
        # Randomly add edges from non_edges
        shuffled_non_edges = non_edges[:]
        random.shuffle(shuffled_non_edges)
        
        for e in shuffled_non_edges:
            # Check if adding e creates a K4
            creates_k4 = False
            u, v = e
            # A K4 containing u,v would be {u,v,x,y} where x,y are common neighbors of u and v in the new graph.
            # Common neighbors in Paley:
            common = []
            for w in range(17):
                if w != u and w != v and adj[u][w] and adj[v][w]:
                    common.append(w)
            # Also consider new edges in E.
            # This is complex. Let's just check if adding e creates a K4 with existing E.
            # Actually, let's just check all K4's that include e.
            # A K4 including e=(u,v) is {u,v,x,y} where:
            # - (u,x), (v,x), (u,y), (v,y), (x,y) are all edges in the new graph.
            # Since Paley is K4-free, at least one of these edges is in E.
            # We are adding e, so e is in E.
            # So we need to check if there exist x,y such that all other 5 edges are in Paley or E.
            
            # This is expensive. Let's just skip the K4 check for now and verify at the end.
            # Or do a simple check: if u and v have 2 common neighbors in Paley, and those two are connected in Paley, then we have a K4.
            # But that's a K4 in Paley, which doesn't exist.
            # So the only way to create a K4 is to use new edges.
            
            # Let's just add e and verify K4-freeness at the end.
            E.add(e)
            
            # Check if all T are hit
            all_hit = True
            for T in tf_sets_8:
                T_set = set(T)
                hit = False
                # Check if there is a new triangle in T
                # Case 1: (u,v) in Paley, (u,w), (v,w) in E
                for a in T:
                    for b in T:
                        if a < b and adj[a][b]:
                            for c in T:
                                if c != a and c != b:
                                    if (a,c) in E or (c,a) in E:
                                        if (b,c) in E or (c,b) in E:
                                            hit = True
                                            break
                                if hit: break
                            if hit: break
                        if hit: break
                    if hit: break
                if not hit:
                    # Case 4: all three new
                    for i in range(len(T)):
                        for j in range(i+1, len(T)):
                            for k in range(j+1, len(T)):
                                a, b, c = T[i], T[j], T[k]
                                if (a,b) in E or (b,a) in E:
                                    if (a,c) in E or (c,a) in E:
                                        if (b,c) in E or (c,b) in E:
                                            hit = True
                                            break
                                if hit: break
                            if hit: break
                        if hit: break
                if not hit:
                    all_hit = False
                    break
            
            if all_hit:
                # Verify K4-freeness
                is_k4_free = True
                # Check all 4-subsets
                for quad in itertools.combinations(range(17), 4):
                    a, b, c, d = quad
                    # Check if all 6 edges are present
                    if adj[a][b] or (a,b) in E or (b,a) in E:
                        if adj[a][c] or (a,c) in E or (c,a) in E:
                            if adj[a][d] or (a,d) in E or (d,a) in E:
                                if adj[b][c] or (b,c) in E or (c,b) in E:
                                    if adj[b][d] or (b,d) in E or (d,b) in E:
                                        if adj[c][d] or (c,d) in E or (d,c) in E:
                                            is_k4_free = False
                                            break
                    if not is_k4_free:
                        break
                if is_k4_free:
                    best_E = list(E)
                    found = True
                    break
        
        if found:
            break
            
    if found and best_E is not None:
        # Construct final graph
        edges = edges_paley[:]
        for e in best_E:
            edges.append(list(e))
        # Add vertex 17 as isolated
        return edges
    else:
        # Fallback: Isolated vertex
        edges = edges_paley[:]
        return edges

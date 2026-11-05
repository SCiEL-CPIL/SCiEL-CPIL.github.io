import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    n = 16
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute Q3 structure
    # Q3 vertices 0..7
    # Edges:
    # (0,1), (0,2), (0,3), (1,4), (1,5), (2,4), (2,6), (3,5), (3,6), (4,7), (5,7), (6,7)
    
    # For each edge (u, v) in Q3, determine the "roles" of u and v
    # u=0, v=1: roles 0, 1
    # u=0, v=2: roles 0, 2
    # ...
    
    # We will precompute for each edge in Q3, the constraints on the other 6 vertices.
    # Let's define a function to check if a given embedding exists.
    
    def check_q3_with_edge(adj, u, v):
        # Check if there is a Q3 containing edge (u, v)
        # We try all possible embeddings of (u, v) into Q3 edges.
        # Q3 edges:
        q3_e = [
            (0, 1), (0, 2), (0, 3), 
            (1, 4), (1, 5), 
            (2, 4), (2, 6), 
            (3, 5), (3, 6), 
            (4, 7), (5, 7), (6, 7)
        ]
        
        for eu, ev in q3_e:
            # Try mapping u->eu, v->ev
            # Determine the other 6 vertices in Q3: {0..7} \ {eu, ev}
            others = [x for x in range(8) if x != eu and x != ev]
            
            # For each permutation of 'others' assigned to 'others' in Q3?
            # No, we need to map vertices from graph to Q3 vertices.
            # Let's fix the mapping of u, v.
            # The remaining 6 Q3 vertices must be mapped to 6 distinct graph vertices.
            
            # We can iterate over subsets of size 6 from V \ {u, v}.
            # But 14 choose 6 is 3003.
            # For each subset, we check if it can be mapped to 'others' to satisfy edges.
            
            # Optimization:
            # In Q3, eu has neighbors: [x for x in range(8) if any(a==eu and b==x for a,b in q3_e)]
            # Let's precompute Q3 adjacency.
            q3_adj = [[False]*8 for _ in range(8)]
            for a, b in q3_e:
                q3_adj[a][b] = True
                q3_adj[b][a] = True
            
            # Neighbors of eu in Q3 (excluding ev)
            neu_q3 = [x for x in others if q3_adj[eu][x]]
            # Neighbors of ev in Q3 (excluding eu)
            nev_q3 = [x for x in others if q3_adj[ev][x]]
            
            # In the graph, u's neighbors (excluding v)
            neu_g = [x for x in range(n) if x != u and x != v and adj[u][x]]
            # v's neighbors (excluding u)
            nev_g = [x for x in range(n) if x != u and x != v and adj[v][x]]
            
            # The graph vertices mapped to neu_q3 must be in neu_g.
            # The graph vertices mapped to nev_q3 must be in nev_g.
            
            # Let's iterate over choices for neu_q3 vertices from neu_g.
            # And choices for nev_q3 vertices from nev_g.
            # The remaining others_q3 vertices can be mapped to any remaining graph vertices.
            
            # This is still complex. Let's stick to the previous logic but optimized.
            # Previous logic:
            # For edge (u, v) in graph:
            # 1. Identify candidates for roles 2,3 (neighbors of u) and 4,5 (neighbors of v) in Q3.
            #    This depends on which Q3 edge (u, v) maps to.
            
            # Let's hardcode the check for each Q3 edge type.
            # Type 1: (0, 1) - u=0, v=1
            #   u-neighbors in Q3 (excl v): 2, 3
            #   v-neighbors in Q3 (excl u): 4, 5
            #   Others: 6, 7
            
            # Type 2: (0, 2) - u=0, v=2
            #   u-neighbors in Q3 (excl v): 1, 3
            #   v-neighbors in Q3 (excl u): 4, 6
            #   Others: 5, 7
            
            # It's easier to just iterate over all permutations of the 6 other Q3 vertices?
            # 6! = 720.
            # For each permutation, check if the mapping is valid.
            # But we need to select 6 vertices from 14.
            
            # Let's use the previous method but cleaner.
            # Iterate over all pairs (r2, r3) from neu_g and (r4, r5) from nev_g?
            # No, the roles depend on the specific Q3 edge.
            
            # Let's just use the general check:
            # Is there a bijection f: {0..7} -> V such that f(eu)=u, f(ev)=v and edges are preserved?
            
            # We can try all assignments of the 6 other Q3 vertices to 6 distinct graph vertices.
            # To speed up, we restrict candidates.
            
            # Let's collect candidates for each Q3 vertex in 'others'.
            # A graph vertex x can map to a Q3 vertex q if:
            # 1. x is not u or v.
            # 2. If q is adjacent to eu in Q3, x must be adjacent to u in G.
            # 3. If q is adjacent to ev in Q3, x must be adjacent to v in G.
            # 4. If q is adjacent to another Q3 vertex q', x must be adjacent to f(q'). (Circular)
            
            # Let's just brute force the 6 vertices.
            # Select 6 vertices from V \ {u, v}.
            # This is too slow if done naively.
            
            # Let's go back to the specific structure check.
            # For a fixed Q3 edge (eu, ev):
            # Let A = {q in others | q3_adj[eu][q]}
            # Let B = {q in others | q3_adj[ev][q]}
            # Let C = others \ A \ B
            
            # We need to map A to 2 distinct neighbors of u (in G).
            # We need to map B to 2 distinct neighbors of v (in G).
            # We need to map C to 2 distinct other vertices.
            
            # Iterate over assignments of A to neu_g (2 distinct).
            # Iterate over assignments of B to nev_g (2 distinct, disjoint from A's images).
            # Iterate over assignments of C to remaining vertices.
            
            # This is feasible.
            
            pass # Placeholder, will implement below

    # Let's implement the fast check properly.
    def has_q3_with_edge(adj, u, v):
        q3_e = [
            (0, 1), (0, 2), (0, 3), 
            (1, 4), (1, 5), 
            (2, 4), (2, 6), 
            (3, 5), (3, 6), 
            (4, 7), (5, 7), (6, 7)
        ]
        
        q3_adj = [[False]*8 for _ in range(8)]
        for a, b in q3_e:
            q3_adj[a][b] = True
            q3_adj[b][a] = True
            
        for eu, ev in q3_e:
            others = [x for x in range(8) if x != eu and x != ev]
            
            # Classify others
            # A: neighbors of eu
            # B: neighbors of ev
            # C: others
            A = [q for q in others if q3_adj[eu][q]]
            B = [q for q in others if q3_adj[ev][q]]
            C = [q for q in others if q not in A and q not in B]
            
            # Get graph neighbors
            neu = [x for x in range(n) if x != u and x != v and adj[u][x]]
            nev = [x for x in range(n) if x != u and x != v and adj[v][x]]
            
            # We need to assign A to distinct neighbors of u.
            # Assign B to distinct neighbors of v.
            # Assign C to distinct remaining vertices.
            
            # Iterate over permutations of A into neu?
            # |A| is usually 2. |B| is usually 2. |C| is usually 2.
            
            # Let's iterate over choices for A
            for perm_A in itertools.permutations(neu, len(A)):
                # Check if perm_A is valid for A
                # A vertices must be distinct (permutation ensures this)
                # Now choose B from nev, disjoint from perm_A
                neu_set = set(perm_A)
                nev_avail = [x for x in nev if x not in neu_set]
                
                if len(nev_avail) < len(B):
                    continue
                    
                for perm_B in itertools.permutations(nev_avail, len(B)):
                    # Now choose C from remaining vertices
                    used = set(perm_A) | set(perm_B)
                    remaining = [x for x in range(n) if x not in used and x != u and x != v]
                    
                    if len(remaining) < len(C):
                        continue
                        
                    for perm_C in itertools.permutations(remaining, len(C)):
                        # We have a full mapping for this Q3 edge embedding
                        # Check if all Q3 edges are satisfied
                        valid = True
                        # Mapping:
                        # eu -> u
                        # ev -> v
                        # A[i] -> perm_A[i]
                        # B[i] -> perm_B[i]
                        # C[i] -> perm_C[i]
                        
                        map_q_to_g = {eu: u, ev: v}
                        for i in range(len(A)):
                            map_q_to_g[A[i]] = perm_A[i]
                        for i in range(len(B)):
                            map_q_to_g[B[i]] = perm_B[i]
                        for i in range(len(C)):
                            map_q_to_g[C[i]] = perm_C[i]
                            
                        for a, b in q3_e:
                            ga = map_q_to_g[a]
                            gb = map_q_to_g[b]
                            if not adj[ga][gb]:
                                valid = False
                                break
                                
                        if valid:
                            return True
                            
        return False

    best_adj = None
    best_count = 0
    
    # Run multiple trials with different random orders
    # Or deterministic orders?
    # Random is better for escaping local optima.
    
    num_trials = 20
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
            
        adj = [[False] * n for _ in range(n)]
        
        # Start empty? Or with some seed?
        # Starting empty allows more flexibility.
        
        # Random order of edges
        edges_order = all_edges[:]
        random.shuffle(edges_order)
        
        current_count = 0
        
        for u, v in edges_order:
            if time.time() > deadline:
                break
                
            # Tentatively add
            adj[u][v] = True
            adj[v][u] = True
            
            # Check Q3
            if has_q3_with_edge(adj, u, v):
                # Revert
                adj[u][v] = False
                adj[v][u] = False
            else:
                current_count += 1
                
        if current_count > best_count:
            best_count = current_count
            best_adj = [row[:] for row in adj]
            
    if best_adj is None:
        # Fallback: return empty or some valid graph
        # Empty graph is valid
        return []
        
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i][j]:
                edges.append([i, j])
                
    return edges

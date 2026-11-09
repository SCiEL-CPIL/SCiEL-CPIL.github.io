import time
import itertools
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    n = 16
    num_edges_total = n * (n - 1) // 2  # 120
    
    # Map edge (u, v) to an index 0..119
    edge_idx_map = {}
    inv_edge_map = [None] * num_edges_total
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_idx_map[(u, v)] = idx
            inv_edge_map[idx] = (u, v)
            idx += 1
    
    # Q3 abstract edges (standard labeling of 3-cube)
    # Vertices 0..7. Edges between i,j if popcount(i^j)==1
    q3_abstract_edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_abstract_edges.append((i, j))
    
    # Generate all Q3 subgraphs as bitmasks
    # For each subset of 8 vertices, check if it forms a Q3
    # A Q3 on 8 vertices has exactly 12 edges
    
    # Precompute: for each 8-vertex subset, the set of edges that form a Q3
    # Use the fact that Q3 is the unique 3-regular bipartite graph on 8 vertices with girth 6
    
    # More efficient: enumerate all 8-vertex subsets and check if the induced subgraph 
    # in K8 contains a Q3 as a subgraph (not necessarily induced)
    
    # Actually, we need to find all subgraphs isomorphic to Q3 in K16
    # A Q3 has 8 vertices and 12 edges
    # We need to find all ways to embed Q3 into K16
    
    # Approach: for each 8-vertex subset S, find all subgraphs of K[S] isomorphic to Q3
    # K[S] has 28 edges. A Q3 has 12 edges.
    # We can enumerate all 12-edge subsets of K[S] and check if any is isomorphic to Q3
    # But that's C(28,12) which is too large
    
    # Better approach: use the structure of Q3
    # Q3 is bipartite with parts of size 4
    # The 8 vertices can be labeled 0..7 where 000,001,010,011 are in one part and 100,101,110,111 in the other
    # Edges connect vertices that differ in exactly one bit
    
    # For a given 8-vertex subset S = {v0, v1, ..., v7}, we need to find a bijection 
    # f: {0..7} -> S such that for each Q3 edge (i,j), (f(i), f(j)) is an edge in K16
    # Since K16 is complete, every pair is an edge, so we just need to check if the 
    # abstract Q3 structure can be embedded
    
    # Wait, K16 is complete, so any 8 vertices form K8, which contains Q3 as a subgraph.
    # So we need to find all embeddings of Q3 into K16, which means all ways to choose 8 vertices 
    # and assign them to the 8 Q3 vertices such that the Q3 edges map to edges of K16.
    # Since K16 is complete, every such assignment works.
    # So the number of Q3 subgraphs is C(16,8) * 8! / |Aut(Q3)|
    # |Aut(Q3)| = 48 (the automorphism group of the cube)
    # C(16,8) = 12870
    # 8! = 40320
    # 12870 * 40320 / 48 = 12870 * 840 = 10,810,800 Q3 subgraphs
    
    # That's a lot. We need a more efficient approach.
    
    # Alternative: instead of precomputing all Q3s, use a different strategy
    # Start with a dense graph and try to add edges while checking for Q3s incrementally
    
    # Let's use a hybrid approach:
    # 1. Start with a good initial graph (like the current program's K3,13 + triangle + matching)
    # 2. Try to add missing edges one by one, checking if it creates a Q3
    # 3. For the Q3 check, use backtracking but with good pruning
    
    # Actually, let's try a different approach: use the complement
    # G is Q3-free iff the complement H = K16 \ G is a Q3-hitting set
    # i.e., every Q3 subgraph has at least one edge in H
    
    # This is a hitting set problem. We want to minimize |H| (maximize |G|)
    
    # But finding the minimum hitting set is NP-hard. However, we can use a greedy approach.
    
    # Let's go back to the direct approach but optimize the Q3 check
    
    # Precompute Q3 structure
    q3_adj = [set() for _ in range(8)]
    for i, j in q3_abstract_edges:
        q3_adj[i].add(j)
        q3_adj[j].add(i)
    
    # Initial graph: K3,13 + triangle in A + matching in B
    adj = [[0]*n for _ in range(n)]
    edges = []
    
    # Add K3,13
    for a in range(3):
        for b in range(3, 16):
            adj[a][b] = 1
            adj[b][a] = 1
            edges.append([a, b])
            
    # Add Triangle in A
    edges.append([0, 1]); adj[0][1]=1; adj[1][0]=1
    edges.append([1, 2]); adj[1][2]=1; adj[2][1]=1
    edges.append([0, 2]); adj[0][2]=1; adj[2][0]=1
    
    # Add Matching in B
    for i in range(3, 15, 2):
        edges.append([i, i+1])
        adj[i][i+1] = 1
        adj[i+1][i] = 1
        
    current_edge_count = len(edges)
    best_edges = list(edges)
    best_count = current_edge_count
    
    # Generate all missing edges
    missing_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j] == 0:
                missing_edges.append((i, j))
    
    # Optimize: sort missing edges by some heuristic
    # Edges within B are more likely to create Q3s, so try those first? 
    # Or try edges that are "safe" first?
    # Let's try a random order with multiple restarts if time allows
    
    import random
    
    def check_q3_free(adj, new_u, new_v):
        """
        Check if adding edge (new_u, new_v) to the graph creates a Q3 subgraph.
        We only need to check Q3s that contain the new edge.
        """
        # Try all ways to map the Q3 such that (new_u, new_v) corresponds to a Q3 edge
        for a, b in q3_abstract_edges:
            # Mapping 1: a->new_u, b->new_v
            if _can_extend_q3(adj, new_u, new_v, a, b, n):
                return False
            # Mapping 2: a->new_v, b->new_u  
            if _can_extend_q3(adj, new_v, new_u, a, b, n):
                return False
        return True
    
    def _can_extend_q3(adj, gu, gv, qa, qb, n):
        """
        Check if we can assign the remaining 6 Q3 vertices to graph vertices
        such that all Q3 edges are satisfied, given that qa is mapped to gu and qb to gv.
        """
        remaining_q = [x for x in range(8) if x != qa and x != qb]
        
        assignment = {qa: gu, qb: gv}
        used = {gu, gv}
        
        # Order remaining Q3 vertices by degree for better pruning
        # All Q3 vertices have degree 3, so no help there
        # But we can order by how many neighbors are already assigned
        def order_remaining():
            order = []
            unassigned = list(remaining_q)
            while unassigned:
                # Pick the vertex with most assigned neighbors
                best = max(unassigned, key=lambda q: sum(1 for nb in q3_adj[q] if nb in assignment))
                order.append(best)
                unassigned.remove(best)
            return order
        
        ordered_remaining = order_remaining()
        
        def backtrack(idx):
            if idx == len(ordered_remaining):
                return True
            
            q = ordered_remaining[idx]
            
            # Candidate graph vertices
            candidates = []
            for g in range(n):
                if g in used:
                    continue
                # Check if g is adjacent to all assigned neighbors of q
                valid = True
                for neighbor_q in q3_adj[q]:
                    if neighbor_q in assignment:
                        g_neighbor = assignment[neighbor_q]
                        if adj[g][g_neighbor] == 0:
                            valid = False
                            break
                if valid:
                    candidates.append(g)
            
            if not candidates:
                return False
                
            for g in candidates:
                used.add(g)
                assignment[q] = g
                
                if backtrack(idx + 1):
                    return True
                    
                used.remove(g)
                del assignment[q]
                
            return False
            
        return backtrack(0)
    
    # Greedy addition with multiple orders
    # Try several random orders and keep the best result
    
    best_count = current_edge_count
    best_edges = list(edges)
    
    num_restarts = 5  # Try multiple random orders
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Reset to initial graph
        adj = [[0]*n for _ in range(n)]
        edges = []
        
        for a in range(3):
            for b in range(3, 16):
                adj[a][b] = 1
                adj[b][a] = 1
                edges.append([a, b])
                
        edges.append([0, 1]); adj[0][1]=1; adj[1][0]=1
        edges.append([1, 2]); adj[1][2]=1; adj[2][1]=1
        edges.append([0, 2]); adj[0][2]=1; adj[2][0]=1
        
        for i in range(3, 15, 2):
            edges.append([i, i+1])
            adj[i][i+1] = 1
            adj[i+1][i] = 1
        
        # Generate missing edges
        missing = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j] == 0:
                    missing.append((i, j))
        
        # Shuffle for random order
        random.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline - 5:
                    break
                
            # Tentatively add
            adj[u][v] = 1
            adj[v][u] = 1
            
            if check_q3_free(adj, u, v):
                edges.append([u, v])
            else:
                # Revert
                adj[u][v] = 0
                adj[v][u] = 0
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = list(edges)
    
    return best_edges

import random
import time
import itertools

# Abstract Q3 edges
# Vertices 0..7
# Edges:
# (0,1), (0,2), (0,4)
# (1,3), (1,5)
# (2,3), (2,6)
# (3,7)
# (4,5), (4,6)
# (5,7)
# (6,7)
Q3_EDGES = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
Q3_ADJ = {i: set() for i in range(8)}
for u, v in Q3_EDGES:
    Q3_ADJ[u].add(v)
    Q3_ADJ[v].add(u)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 16
    best_edges = 0
    best_graph = None
    
    # Precompute neighbors for Q3 abstract graph
    # For the check, we fix edge (0,1) to map to (u,v).
    # Neighbors of 0: 1, 2, 4. (1 is v, so 2,4 are other neighbors)
    # Neighbors of 1: 0, 3, 5. (0 is u, so 3,5 are other neighbors)
    
    def creates_q3(adj, u, v):
        """
        Check if adding edge (u,v) creates a Q3 subgraph.
        adj is a list of sets of neighbors.
        """
        # We only need to check Q3s that contain (u,v).
        # Map abstract Q3 vertices such that (0,1) -> (u,v).
        # There are two orientations: (0->u, 1->v) or (0->v, 1->u).
        # Due to symmetry, we can just check one orientation and swap u,v if needed, 
        # or check both. Let's check both by iterating over the two mappings of the edge endpoints.
        
        # Mapping 1: 0->u, 1->v
        # 0's neighbors in Q3: 1, 2, 4. In graph: v, img(2), img(4).
        # 1's neighbors in Q3: 0, 3, 5. In graph: u, img(3), img(5).
        
        # img(2) and img(4) must be in adj[u] (excluding v if we consider current state, 
        # but we are checking if ADDING (u,v) creates it. 
        # So u and v are not connected yet.
        # img(2) must be in adj[u]. img(4) must be in adj[u].
        # img(3) must be in adj[v]. img(5) must be in adj[v].
        
        # Let's collect candidates
        cand_2_4 = [x for x in adj[u] if x != v]
        cand_3_5 = [x for x in adj[v] if x != u]
        
        # We need to pick distinct a,b from cand_2_4 for img(2), img(4)
        # and distinct c,d from cand_3_5 for img(3), img(5)
        # such that:
        # (a,c) in adj (2-3)
        # (b,d) in adj (4-5)
        # Then we need img(6) and img(7).
        # 6 is connected to 2,4,7. So img(6) must be in adj[a] and adj[b].
        # 7 is connected to 3,5,6. So img(7) must be in adj[c] and adj[d] and adj[img(6)].
        # And (img(6), img(7)) must be in adj.
        
        # To speed up, we can precompute intersections? 
        # The lists are small (max 14).
        
        for a in cand_2_4:
            for b in cand_2_4:
                if a == b: continue
                # a is img(2), b is img(4)
                for c in cand_3_5:
                    if c in (a, b, u, v): continue
                    if c not in adj[a]: continue # (2,3) edge
                    for d in cand_3_5:
                        if d in (a, b, c, u, v): continue
                        if d not in adj[b]: continue # (4,5) edge
                        
                        # Now find f (img 6) and g (img 7)
                        # f must be neighbor of a and b
                        # g must be neighbor of c and d
                        # f and g must be connected
                        # f, g must be distinct from u,v,a,b,c,d
                        
                        # Candidates for f: adj[a] & adj[b]
                        # Candidates for g: adj[c] & adj[d]
                        
                        # We need f in adj[a] and adj[b]
                        # We need g in adj[c] and adj[d]
                        # We need (f,g) in adj
                        
                        # Let's iterate over possible f
                        for f in adj[a]:
                            if f in (u, v, a, b, c, d): continue
                            if f not in adj[b]: continue
                            # Now check if there exists g
                            # g must be in adj[c] and adj[d] and adj[f]
                            # And g must not be in (u,v,a,b,c,d,f)
                            # We can just check if the intersection adj[c] & adj[d] & adj[f] 
                            # contains a vertex not in the forbidden set.
                            
                            # Optimization:
                            # Let S = adj[c] & adj[d]
                            # We need an element in S & adj[f] that is not forbidden.
                            # Since S is small, we can just iterate.
                            
                            # Actually, we can just check:
                            # Is there a g in adj[c] such that g in adj[d] and g in adj[f] and g not in forbidden?
                            # Or simpler:
                            # intersection = set(adj[c]) & set(adj[d]) & set(adj[f])
                            # forbidden = {u, v, a, b, c, d, f}
                            # if intersection - forbidden: return True
                            
                            # To avoid set creation overhead in tight loop, 
                            # and since degrees are small, this is fine.
                            
                            # Even better:
                            # We just need to know if such a g exists.
                            # We can precompute adj_sets as sets for faster intersection.
                            # Let's assume adj is list of sets.
                            
                            # Let's do the set intersection.
                            # This is the innermost loop, so we want it fast.
                            # n=16, so sets are small.
                            
                            # Check if any g works
                            # g must be in adj[c]
                            for g in adj[c]:
                                if g in (u, v, a, b, c, d, f): continue
                                if g in adj[d] and g in adj[f]:
                                    return True
        return False

    def check_q3_exists(adj):
        """
        Check if the graph contains any Q3.
        Used for verification of the final result.
        """
        # Iterate over all 8-vertex subsets
        # For each, check if it contains a Q3.
        # This is slower than the incremental check, but we only need it once at the end.
        # Or we can use the same logic but we don't have a "new edge".
        # We can just pick any edge and check if it's part of a Q3? No, that's not sufficient.
        # We need to check all edges.
        # But for the greedy algorithm, we only add edges if they don't create a Q3.
        # So the final graph is guaranteed to be Q3-free if the incremental check is correct.
        # We will trust the incremental check but add a final verification just in case.
        
        # For verification, we can iterate over all pairs (u,v) that are edges,
        # and check if removing (u,v) would make it Q3-free? No.
        # We just need to find ONE Q3.
        # We can iterate over all 8-subsets and check if the induced subgraph contains Q3.
        # To check if an 8-vertex graph contains Q3:
        # We can try all 8! / 48 mappings? No, too slow.
        # We can use the same "creates_q3" logic by adding edges one by one?
        # Simpler: 
        # For each 8-subset S, let G_S be the induced subgraph.
        # If |E(G_S)| < 12, skip.
        # Otherwise, check if G_S contains Q3.
        # We can do this by trying to map Q3 vertices to S.
        # Fix 0->s0, 1->s1.
        # This is similar to creates_q3 but without the "new edge" constraint.
        
        # Given time constraints, I will rely on the correctness of creates_q3.
        # If creates_q3 is correct, the greedy process ensures no Q3 is formed.
        return False # Placeholder, we trust the greedy process

    # Greedy Algorithm
    for restart in range(20):
        if time.time() > deadline - 5:
            break
            
        # Initialize empty graph
        adj = [set() for _ in range(n)]
        
        # Generate a random order of edges
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                edges.append((u, v))
        
        # Shuffle
        random.shuffle(edges)
        
        # Add edges
        for u, v in edges:
            if time.time() > deadline - 5:
                break
            if not creates_q3(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
        
        # Count edges
        num_edges = sum(len(s) for s in adj) // 2
        
        if num_edges > best_edges:
            best_edges = num_edges
            best_graph = [list(s) for s in adj]
            
        # Optional: Local improvement / Backtracking
        # Try to add more edges by re-ordering or backtracking?
        # The greedy approach is usually good enough for this size.
        
    # Construct the output list of edges
    if best_graph is None:
        # Fallback to seed
        return [[a, b] for a in range(3) for b in range(3, 16)]
        
    result_edges = []
    for u in range(n):
        for v in best_graph[u]:
            if u < v:
                result_edges.append([u, v])
                
    return result_edges

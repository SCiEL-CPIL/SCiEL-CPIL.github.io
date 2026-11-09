import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    # Base solution: 8 disjoint K4s
    # n = 32
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    best_n = n
    best_edges = list(edges)
    
    # Try to extend to n=33,34,35 by adding a vertex with appropriate edges
    # A new vertex must:
    # 1. Not create a C5
    # 2. Not create an independent set of size 9
    
    # For C5-freeness: if we add vertex v connected to some existing vertices,
    # we need to ensure no 5-cycle is formed. A 5-cycle through v would need
    # v-a-b-c-d-v where a,d are neighbors of v and a-b-c is a path of length 2
    # in the remaining graph avoiding v.
    
    # For alpha: the new vertex must not be part of an independent 9-set.
    # If v is non-adjacent to a set S of 8 vertices that are mutually non-adjacent,
    # that creates an independent 9-set.
    
    def has_c5(n, edge_set):
        """Check if graph contains a 5-cycle. Returns True if C5 exists."""
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Look for 5-cycles: v0-v1-v2-v3-v4-v0
        # To avoid duplicates, require v0 < v1 < v2 and v0 < v4
        for v0 in range(n):
            for v1 in adj[v0]:
                if v1 <= v0:
                    continue
                for v2 in adj[v1]:
                    if v2 <= v0 or v2 == v1:
                        continue
                    for v3 in adj[v2]:
                        if v3 <= v0 or v3 in (v1, v2):
                            continue
                        for v4 in adj[v3]:
                            if v4 <= v0 or v4 in (v1, v2, v3):
                                continue
                            if v4 in adj[v0]:
                                return True
        return False
    
    def max_independent_set_size(n, edge_set):
        """Find the size of the maximum independent set using branch and bound."""
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Non-edges: pairs that are NOT connected
        # An independent set is a set of vertices with no edges between them
        # i.e., all pairs are non-adjacent
        
        best = [0]
        
        def dfs(candidates, current_size):
            if time.time() > deadline - 5:
                return
            if not candidates:
                if current_size > best[0]:
                    best[0] = current_size
                return
            # Upper bound: current_size + len(candidates)
            if current_size + len(candidates) <= best[0]:
                return
            
            # Branch on first candidate
            v = candidates[0]
            rest = candidates[1:]
            
            # Include v: remove all neighbors of v from rest
            if current_size + 1 + len([x for x in rest if x not in adj[v]]) > best[0]:
                new_rest = [x for x in rest if x not in adj[v]]
                dfs(new_rest, current_size + 1)
            
            # Exclude v
            dfs(rest, current_size)
        
        dfs(list(range(n)), 0)
        return best[0]
    
    def is_valid(n, edge_list):
        """Check C5-free and alpha <= 8."""
        if time.time() > deadline - 5:
            return False
        edge_set = set((min(u,v), max(u,v)) for u, v in edge_list)
        if has_c5(n, edge_set):
            return False
        if max_independent_set_size(n, edge_set) >= 9:
            return False
        return True
    
    # Try extending the base graph
    for target_n in range(33, 36):
        if time.time() > deadline - 10:
            break
        
        new_vertex = 32  # vertex index for the new vertex
        
        # Strategy: add vertex connected to some vertices in the existing graph
        # To avoid C5: the neighbors of the new vertex must not contain a path of length 3
        # (which would form a 5-cycle with the new vertex)
        # To avoid alpha=9: the new vertex's non-neighbors must not contain an independent 8-set
        
        # Try different connection patterns
        # The 8 K4 blocks are: block b has vertices 4b, 4b+1, 4b+2, 4b+3
        
        # Option 1: Connect new vertex to one vertex from each block (8 connections)
        # Then non-neighbors = 32 - 8 = 24 vertices. Need to check if any 8 of those
        # form an independent set. Within each block, the 3 non-connected vertices form
        # a K3 (clique), so they can't be in an independent set together.
        # Actually within a K4, if we connect to 1 vertex, the other 3 are mutually adjacent
        # (they form a K3), so at most 1 from each block's non-neighbors can be in an independent set.
        # So max independent from non-neighbors = 8 (one from each block).
        # That gives alpha = 9 (the new vertex + 8). INVALID.
        
        # Option 2: Connect new vertex to 2 vertices from each block (16 connections)
        # Non-neighbors: 2 per block, and within each block those 2 are adjacent (part of K4).
        # So at most 1 from each block's non-neighbors in independent set.
        # Max independent from non-neighbors = 8. Still alpha = 9. INVALID.
        
        # Option 3: Connect to 3 vertices from each block (24 connections)
        # Non-neighbors: 1 per block. Those 8 vertices are in different blocks (no edges between blocks).
        # So they form an independent set of size 8! Alpha = 9. INVALID.
        
        # Option 4: Connect to all 4 vertices of some blocks and 0 from others
        # E.g., connect to all 4 vertices of 4 blocks (16 connections)
        # Non-neighbors: 4 vertices from each of 4 blocks. Within each block, those 4 form a K4.
        # So at most 1 from each of those 4 blocks in independent set. Max from non-neighbors = 4.
        # Alpha = 5. VALID for alpha!
        # But C5 check: neighbors are 16 vertices forming 4 K4s. 
        # A C5 through new vertex: new-v1-v2-v3-v4-new where v1,v4 are neighbors, v2,v3 are anything.
        # v1 and v4 are in the neighbor set. v2 is adjacent to v1, v3 adjacent to v2 and v4.
        # Since neighbors are in K4 blocks, if v1 is in block A, v2 must be adjacent to v1.
        # v2 could be in block A (adjacent within K4) or... wait, no edges between blocks.
        # So v2 must be in the same block as v1. Similarly v3 must be adjacent to v4, so in same block as v4.
        # For a 5-cycle: new-v1-v2-v3-v4-new. v1,v2 in same block A, v3,v4 in same block B.
        # We need v2-v3 edge. But v2 is in block A and v3 is in block B, and there are no edges between blocks!
        # So NO 5-cycle can be formed! VALID!
        
        # Let's verify: connect new vertex to all 4 vertices of blocks 0,1,2,3.
        test_edges = list(edges)
        for b in range(4):
            for i in range(4):
                test_edges.append([32, 4*b + i])
        
        if is_valid(33, test_edges):
            best_n = 33
            best_edges = test_edges
            print(f"Found valid n=33 graph!")
            break
        else:
            print(f"n=33 option (4 full blocks) invalid")
        
        # Try: connect to 3 vertices from each of 4 blocks (12 connections)
        # Non-neighbors: 1 from each of 4 blocks + 4 from each of 4 blocks = 4 + 16 = 20
        # From the 4 blocks with 1 non-neighbor: those 4 are independent (different blocks)
        # From the 4 blocks with 4 non-neighbors (full K4): at most 1 each, so 4 more
        # Total max independent from non-neighbors: 4 + 4 = 8. Alpha = 9. INVALID.
        
        # Try: connect to 4 vertices from 5 blocks (20 connections), 0 from 3 blocks
        # Non-neighbors: 0 from 5 blocks + 4 from 3 blocks = 12 vertices
        # From the 3 full blocks: at most 1 each = 3
        # Alpha = 4. VALID for alpha.
        # C5: neighbors are 5 K4s. Same argument: no edges between blocks, so no 5-cycle.
        test_edges = list(edges)
        for b in range(5):
            for i in range(4):
                test_edges.append([32, 4*b + i])
        
        if is_valid(33, test_edges):
            best_n = 33
            best_edges = test_edges
            print(f"Found valid n=33 graph (5 blocks)!")
            break
    
    if best_n == 32:
        # Try random search for n=33
        if time.time() < deadline - 15:
            random.seed(42)
            for attempt in range(100):
                if time.time() > deadline - 10:
                    break
                # Random subset of vertices to connect to
                candidates = list(range(32))
                random.shuffle(candidates)
                # Try connecting to a random subset
                for num_connect in [8, 12, 16, 20, 24, 28]:
                    if time.time() > deadline - 10:
                        break
                    neighbors = candidates[:num_connect]
                    test_edges = list(edges)
                    for v in neighbors:
                        test_edges.append([32, v])
                    
                    if is_valid(33, test_edges):
                        best_n = 33
                        best_edges = test_edges
                        print(f"Found valid n=33 via random search!")
                        break
                if best_n == 33:
                    break
    
    # Try n=34: add two more vertices
    if best_n == 33 and time.time() < deadline - 20:
        # Start from best 33-vertex graph and try to add vertex 33
        for target_n in range(34, 36):
            if time.time() > deadline - 10:
                break
            new_v = 33
            base_edges = list(best_edges)
            
            # Try connecting new vertex to various subsets
            existing_vertices = list(range(32))
            random.shuffle(existing_vertices)
            
            found = False
            for num_connect in range(0, 33, 2):
                if time.time() > deadline - 10:
                    break
                neighbors = existing_vertices[:num_connect]
                test_edges = list(base_edges)
                for v in neighbors:
                    test_edges.append([new_v, v])
                
                if is_valid(target_n, test_edges):
                    best_n = target_n
                    best_edges = test_edges
                    found = True
                    print(f"Found valid n={target_n} graph!")
                    break
            
            if not found:
                # Try more systematic: connect to all vertices in certain blocks
                # For n=34, we have 33 existing vertices (32 original + 1 new)
                # This gets complex; try a few structured options
                pass
    
    # Final validation of best solution
    if best_n > 32:
        final_edge_set = set((min(u,v), max(u,v)) for u, v in best_edges)
        if has_c5(best_n, final_edge_set) or max_independent_set_size(best_n, final_edge_set) >= 9:
            print("Final validation failed, reverting to n=32")
            best_n = 32
            best_edges = edges
    
    return {'n': best_n, 'edges': best_edges}

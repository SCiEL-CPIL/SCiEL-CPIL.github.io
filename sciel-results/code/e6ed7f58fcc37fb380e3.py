import random
import time

def construct():
    n = 16
    # Start with K_{3,13}
    # Vertices 0,1,2 are in part A. 3..15 in part B.
    edges = set()
    for a in range(3):
        for b in range(3, 16):
            edges.add((a, b))
            
    # Helper to check if a set of 8 vertices contains Q3
    # Q3 adjacency: 000, 001, 010, 011, 100, 101, 110, 111
    # Edges: (000,001), (000,010), (000,100), (001,011), (001,101), (010,011), (010,110), (011,111), (100,101), (100,110), (101,111), (110,111)
    # Let's map to 0..7
    # 0:000, 1:001, 2:010, 3:011, 4:100, 5:101, 6:110, 7:111
    q3_edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]
    
    def contains_q3_in_subset(subset, graph_edges):
        # subset is a tuple of 8 vertices
        # graph_edges is a set of tuples
        # Check degrees first
        # Build local adjacency
        local_adj = {v: set() for v in subset}
        for u in subset:
            for v in subset:
                if u == v: continue
                if (u, v) in graph_edges or (v, u) in graph_edges:
                    local_adj[u].add(v)
        
        # If any degree < 3, return False
        for v in subset:
            if len(local_adj[v]) < 3:
                return False
        
        # Try to embed Q3
        # Map Q3 vertices 0..7 to subset vertices
        # Use backtracking
        # Order Q3 vertices by degree (all 3) or just fixed order
        # Let's use a simple recursive check
        # We can optimize by checking if the bipartition matches
        # Q3 is bipartite. Even parity: 0,3,5,6. Odd parity: 1,2,4,7.
        # Check if subset can be partitioned into two sets of 4 such that all Q3 edges are present.
        # This is still complex.
        
        # Simpler: Just try all permutations of subset mapped to 0..7? 8! = 40320.
        # Too slow if called 3000 times.
        
        # Better: Check if the induced subgraph is isomorphic to Q3 or contains it.
        # Since we checked degrees >= 3, and it's 8 vertices.
        # If the number of edges in induced subgraph is 12, it must be Q3?
        # No, there are other 3-regular graphs on 8 vertices (e.g., K_{4,4} minus a matching? No, K_{4,4} is 4-regular).
        # The only 3-regular bipartite graph on 8 vertices is Q3?
        # Let's check.
        # A 3-regular bipartite graph on 4+4 vertices.
        # The number of such graphs is small.
        # If the induced subgraph has exactly 12 edges and all degrees are 3, it is a 3-regular bipartite graph.
        # Is it necessarily Q3?
        # Q3 is the unique 3-regular bipartite graph on 8 vertices?
        # No, consider two disjoint cycles? No, it's connected?
        # Q3 is connected.
        # Are there other connected 3-regular bipartite graphs on 8 vertices?
        # The cube is one.
        # What about a "twisted" cube?
        # Actually, Q3 is the only 3-regular bipartite graph on 8 vertices that is Hamiltonian?
        # Let's assume for now that if the induced subgraph has 12 edges and all degrees 3, it contains Q3.
        # Wait, if it IS Q3, it contains Q3.
        # If it's another 3-regular bipartite graph, does it contain Q3?
        # If it's not Q3, it might still contain Q3 as a subgraph?
        # If it's 3-regular and has 12 edges, it IS a 3-regular graph.
        # If it contains Q3, it must have at least 12 edges.
        # So if it has 12 edges and contains Q3, it IS Q3.
        # So we just need to check if the induced subgraph is isomorphic to Q3.
        
        # Check if induced subgraph has 12 edges.
        edge_count = sum(len(local_adj[v]) for v in subset) // 2
        if edge_count < 12:
            return False
        if edge_count > 12:
            # It might contain Q3 as a proper subgraph
            # But if it has >12 edges, it's denser.
            # We need to check if Q3 is a subgraph.
            # This is harder.
            # However, in our greedy process, we start sparse and add edges.
            # Most failures will be due to <12 edges or degree <3.
            # If we get to >12 edges, we are in a dense region.
            # Let's just return True if edge_count >= 12 and degrees >= 3?
            # This is a heuristic. It might falsely reject valid edges.
            # But it's fast.
            # Let's try to be more precise.
            pass
            
        # For now, let's just check if it is isomorphic to Q3.
        # If edge_count == 12 and all degrees 3, check isomorphism.
        if edge_count == 12:
             # Check if it's Q3
             # Q3 has 6 4-cycles.
             # Count 4-cycles in induced subgraph.
             # If 6, it's likely Q3.
             pass
             
        return False

    # This is getting complicated. Let's use a simpler heuristic:
    # Just try to add edges and if we are unsure, reject.
    # Or, use a pre-computed list of "safe" additions.
    
    # Alternative: Use a known dense Q3-free graph.
    # The graph formed by the vertices of a 4-dimensional hypercube Q4?
    # Q4 has 16 vertices.
    # Is Q4 Q3-free?
    # Q4 contains Q3 as a subgraph?
    # Yes, Q4 is a Q3-prism? No.
    # Q4 contains many Q3s.
    # So Q4 is not Q3-free.
    
    # What is the complement of Q4?
    # Q4 has 32 edges.
    # Complement has 120 - 32 = 88 edges.
    # Does the complement of Q4 contain Q3?
    # This is a specific graph.
    
    # Let's try the **complement of a matching** approach again but with a twist.
    # No, that's too dense.
    
    # Let's stick to the **Greedy Random** approach with a conservative Q3 check.
    # We will only accept an edge if we are CERTAIN it doesn't create a Q3.
    # We check all 8-sets containing the new edge.
    # For each set, we check if the induced subgraph is isomorphic to Q3.
    # If edge_count < 12, reject.
    # If edge_count == 12 and degrees all 3, check if it's Q3.
    # If edge_count > 12, we assume it might contain Q3, so we reject to be safe.
    # This will limit the density, but it's safe.
    
    def is_q3(subset, graph_edges):
        # Check if induced subgraph on subset is isomorphic to Q3
        local_adj = {v: set() for v in subset}
        for u in subset:
            for v in subset:
                if u == v: continue
                if (u, v) in graph_edges or (v, u) in graph_edges:
                    local_adj[u].add(v)
        
        # Check degrees
        for v in subset:
            if len(local_adj[v]) != 3:
                return False
        
        # Check edge count
        edge_count = sum(len(local_adj[v]) for v in subset) // 2
        if edge_count != 12:
            return False
            
        # Check if it's Q3
        # Q3 is the only 3-regular bipartite graph on 8 vertices with girth 4?
        # Let's check the number of 4-cycles.
        # Q3 has 6 4-cycles.
        # Count 4-cycles in local_adj.
        # A 4-cycle is a set of 4 vertices {a,b,c,d} such that (a,b), (b,c), (c,d), (d,a) are edges.
        # In a 3-regular graph, we can count them.
        # This is O(1) for 8 vertices.
        # Let's just return True if it passes degree and edge count checks.
        # Risk: There might be another 3-regular bipartite graph on 8 vertices.
        # If so, and it doesn't contain Q3, we would falsely reject.
        # But if it IS Q3, we reject correctly.
        # If it's not Q3, we reject safely (maybe too safely).
        return True

    # Greedy loop
    start_time = time.time()
    non_edges = [(u, v) for u in range(n) for v in range(u+1, n) if (u, v) not in edges and (v, u) not in edges]
    random.shuffle(non_edges)
    
    for u, v in non_edges:
        if time.time() - start_time > 110:
            break
        # Try adding (u, v)
        edges.add((u, v))
        
        # Check for Q3
        found_q3 = False
        # Iterate through all 6 other vertices from the remaining 14
        other_vertices = [x for x in range(n) if x != u and x != v]
        # We need to pick 6 from 14.
        # This is C(14,6) = 3003.
        # To speed up, we can just check a random sample?
        # No, we need certainty.
        # But 3003 * is_q3() might be slow if is_q3 is slow.
        # is_q3 is O(1) basically.
        # 3003 * 10000 attempts = 30M ops. Might be slow in Python.
        
        # Optimization: Only check if the new edge is "critical".
        # If we just want to improve the score, we can do a random subset of checks.
        # But invalid results score 0.
        
        # Let's limit the checks to a random sample of 100 subsets per edge.
        # If we find a Q3, reject.
        # If we don't find one, accept (risk of false negative).
        # This is risky.
        
        # Better: Use a deterministic construction.
        # Let's return the seed if we can't guarantee speed.
        
        # Actually, let's try to construct a specific graph.
        # How about the **Clebsch graph**?
        # The Clebsch graph has 16 vertices, 3-regular? No, 5-regular.
        # It is a strong regular graph.
        # Does it contain Q3?
        # It is triangle-free.
        # Q3 is bipartite, so it has no triangles.
        # The Clebsch graph is the complement of the 4x4 rook graph?
        # It has 40 edges.
        # If it is Q3-free, 40 > 39.
        
        # Let's check if Clebsch graph is Q3-free.
        # The Clebsch graph is the folded 5-cube.
        # It is known to be Q3-free?
        # I don't know for sure.
        
        # Let's try **K_{4,4} minus a perfect matching**.
        # This graph has 12 edges.
        # It is Q3-free?
        # Q3 is 3-regular. K_{4,4} minus matching is 3-regular.
        # Is it isomorphic to Q3?
        # Q3 is bipartite.
        # K_{4,4} minus matching is bipartite.
        # Are they the same?
        # Q3 has 6 4-cycles.
        # K_{4,4} minus matching:
        # Vertices 1,2,3,4 in L, 1',2',3',4' in R.
        # Edges: (i, j') for i!=j.
        # 4-cycles: (1, 2', 2, 1')?
        # 1-2' (yes), 2'-2 (yes), 2-1' (yes), 1'-1 (yes).
        # This is a 4-cycle.
        # How many?
        # Each pair of vertices in L and R forms a 4-cycle?
        # There are C(4,2)*C(4,2) = 6*6 = 36 potential 4-cycles?
        # No, a 4-cycle is determined by 2 vertices in L and 2 in R.
        # For a fixed pair in L and pair in R, there is exactly one 4-cycle if all 4 edges are present.
        # In K_{4,4} minus matching, edges (1,1') and (2,2') are missing.
        # So the 4-cycle (1, 2', 2, 1') uses edges (1,2'), (2',2), (2,1'), (1',1).
        # (1,2') is present. (2',2) is present. (2,1') is present. (1',1) is present.
        # So this 4-cycle exists.
        # In fact, every pair of L and R vertices forms a 4-cycle.
        # So there are 36 4-cycles.
        # Q3 has 6 4-cycles.
        # So K_{4,4} minus matching is NOT Q3.
        # Does it contain Q3?
        # It has 12 edges. Q3 has 12 edges.
        # If it contains Q3, it must BE Q3.
        # Since it has 36 4-cycles and Q3 has 6, it is not Q3.
        # So it is Q3-free.
        
        # So we can take 4 disjoint copies of (K_{4,4} minus matching)?
        # That would be 4 * 12 = 48 edges.
        # But they are disjoint.
        # Total vertices: 4 * 8 = 32. We only have 16.
        # So we can take 2 copies.
        # 2 * 12 = 24 edges.
        # Worse than 39.
        
        # What if we connect them?
        
        # Let's go back to the **Greedy** approach but with a time limit and random restarts.
        # If we can't guarantee correctness, we return the seed.
        
        # Let's try to add edges to K_{3,13} within the 13-part.
        # We established that adding a matching is likely safe.
        # Let's add a maximum matching in the 13-part.
        # Size 6.
        # Edges: 39 + 6 = 45.
        # Is it Q3-free?
        # We need to verify.
        
        # Let's assume it is and return it.
        # If it fails, we get 0.
        # But 45 > 39.
        
        # Let's try to be smarter.
        # Return the seed.
        return [[u, v] for u, v in edges]

# EVOLVE-BLOCK-END

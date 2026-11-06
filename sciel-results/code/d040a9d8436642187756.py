import itertools
import time

def construct():
    # Initialize K16
    n = 16
    edges = set()
    for u in range(n):
        for v in range(u+1, n):
            edges.add((u, v))
    
    # Precompute all 8-vertex subsets
    subsets = list(itertools.combinations(range(n), 8))
    
    # Helper to check if a set of edges on 8 vertices contains Q3
    # We can map the 8 vertices to 0..7
    # Q3 edges: 
    # 0-1, 0-2, 0-4
    # 1-3, 1-5
    # 2-3, 2-6
    # 4-5, 4-6
    # 3-7
    # 5-7
    # 6-7
    # Wait, standard Q3 edges:
    # 0: 1, 2, 4
    # 1: 0, 3, 5
    # 2: 0, 3, 6
    # 3: 1, 2, 7
    # 4: 0, 5, 6
    # 5: 1, 4, 7
    # 6: 2, 4, 7
    # 7: 3, 5, 6
    q3_edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (4,5), (4,6), (5,7), (6,7), (3,7)]
    
    def contains_q3(subset, current_edges):
        # Map subset vertices to 0..7
        mapping = {v: i for i, v in enumerate(subset)}
        # Check if all Q3 edges are present
        # We need to check if there is a permutation of the 8 vertices that matches Q3
        # Since Q3 is symmetric, we can fix vertex 0 to subset[0]
        # Then try permutations of the other 7? 7! = 5040.
        # 12870 * 5040 is too slow.
        
        # Better: Check if the induced subgraph is isomorphic to Q3 or has Q3 as subgraph.
        # Since we are removing edges, the graph is a subgraph of K8.
        # We can check if the number of edges is at least 12.
        # If < 12, no Q3.
        # If >= 12, we need to check structure.
        
        # Actually, we can just check if the graph contains Q3 as a subgraph.
        # This is NP-complete in general, but for 8 vertices it's fast.
        # We can use a simple backtracking or just check all 8! permutations?
        # 40320 permutations.
        # 12870 * 40320 is too slow.
        
        # Optimization:
        # Q3 is 3-regular.
        # If the induced subgraph has a vertex with degree < 3, it cannot contain Q3?
        # No, Q3 is a subgraph, not induced.
        # But if a vertex has degree 2 in the induced subgraph, it cannot be part of a 3-regular subgraph using all 8 vertices.
        # Wait, the subgraph uses ALL 8 vertices?
        # "subgraph isomorphic to Q3".
        # Yes, Q3 has 8 vertices. So the subgraph must use 8 vertices.
        # So if any vertex in the subset has degree < 3 in the induced subgraph, it cannot be in a Q3 subgraph that uses all 8 vertices.
        # So check degrees first.
        
        degs = [0]*8
        for u in subset:
            for v in subset:
                if u != v and (min(u,v), max(u,v)) in current_edges:
                    degs[mapping[u]] += 1
                    # Avoid double counting
                    # Actually, just count neighbors in subset
            
        # Recalculate degrees properly
        degs = [0]*8
        for i, u in enumerate(subset):
            for j, v in enumerate(subset):
                if i != j and (min(u,v), max(u,v)) in current_edges:
                    degs[i] += 1
        
        if any(d < 3 for d in degs):
            return False
            
        # If all degrees >= 3, check if Q3 is a subgraph.
        # We can try to find a matching or just check if the graph is isomorphic to Q3?
        # No, it can be a supergraph.
        # But if it's a supergraph, it contains Q3.
        # Actually, if we remove edges, we want to break Q3s.
        # So we just need to know if Q3 exists.
        
        # Simple check: Try all permutations of subset to match Q3?
        # Too slow.
        
        # Alternative:
        # The number of edges in the induced subgraph must be at least 12.
        edge_count = sum(degs) // 2
        if edge_count < 12:
            return False
            
        # If edge_count >= 12, we assume it might contain Q3.
        # For speed, we can just return True if edge_count >= 12?
        # This might remove edges unnecessarily, but it's safe.
        # It will result in a graph with max degree 2 or edge_count < 12 for all 8-subsets.
        # This is a strong condition.
        # It ensures no 8 vertices have 12 edges.
        # This is sufficient to be Q3-free.
        # Is it necessary? No, but it's a good heuristic.
        # It might be too restrictive, leading to fewer edges.
        
        # Let's try to be more precise.
        # Check if the graph is isomorphic to Q3?
        # No, just contains Q3.
        
        # Given time constraints, let's use the "edge_count >= 12" heuristic first.
        # If it gives a good score, we can refine.
        return edge_count >= 12

    start_time = time.time()
    deadline = start_time + 110
    
    # Iteratively remove edges
    while time.time() < deadline:
        found_q3 = False
        for subset in subsets:
            if contains_q3(subset, edges):
                # Find an edge to remove
                # Remove a random edge from the subset?
                # Or the one with highest degree?
                # Let's remove a random edge from the subset to break the Q3.
                # Actually, we should remove an edge that is in the Q3.
                # Since we don't know the exact Q3, we pick any edge in the subset.
                subset_edges = [(u, v) for u in subset for v in subset if u < v and (u, v) in edges]
                if subset_edges:
                    # Pick the first one? Or random?
                    # Random might be better to avoid bias.
                    import random
                    e = random.choice(subset_edges)
                    edges.remove(e)
                    found_q3 = True
                    break
        if not found_q3:
            break
            
    # Convert to list of lists
    return [[u, v] for u, v in edges]

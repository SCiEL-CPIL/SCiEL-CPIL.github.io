import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 seconds safety margin
    
    best_edges = 0
    best_graph = set()
    
    n = 16
    
    # Precompute all Q3 structures? No, check on the fly.
    
    def contains_q3(graph):
        """
        graph: set of edges (u, v) with u < v
        Returns True if graph contains a Q3 subgraph.
        """
        # Build adjacency matrix for fast lookup
        adj = [[False] * n for _ in range(n)]
        for u, v in graph:
            adj[u][v] = True
            adj[v][u] = True
        
        # Iterate over all pairs of 4-sets (A, B)
        # A and B are disjoint
        # Check if induced subgraph on A U B is Q3
        
        # To optimize, we can first check if there are any 4-sets with correct degrees?
        # Or just iterate.
        
        # There are C(16,4) * C(12,4) / 2 pairs.
        # C(16,4) = 1820
        # C(12,4) = 495
        # Total ~ 450,000 pairs.
        
        # For each pair, check if it's Q3.
        # Q3 has 12 edges.
        # Degrees are 3 for all 8 vertices.
        
        # We can speed up by checking degrees first.
        
        # Precompute degrees in the graph
        deg = [0] * n
        for u, v in graph:
            deg[u] += 1
            deg[v] += 1
            
        # A vertex in Q3 must have degree at least 3 in the graph.
        # So we only consider vertices with deg >= 3.
        candidates = [i for i in range(n) if deg[i] >= 3]
        
        if len(candidates) < 8:
            return False
            
        # Iterate over 4-subsets of candidates for A
        for A in itertools.combinations(candidates, 4):
            A_set = set(A)
            # B must be chosen from candidates \ A
            B_candidates = [v for v in candidates if v not in A_set]
            for B in itertools.combinations(B_candidates, 4):
                # Check if induced subgraph on A U B is Q3
                # 1. Check degrees within A U B
                # 2. Check if bipartite between A and B
                # 3. Check if it's exactly Q3 (12 edges, specific structure)
                
                # Actually, just check if the induced subgraph is isomorphic to Q3.
                # Since Q3 is unique up to isomorphism for 8 vertices, 12 edges, bipartite 4-4, degree 3.
                
                # Check edges count
                edge_count = 0
                for u in A:
                    for v in B:
                        if adj[u][v]:
                            edge_count += 1
                            if edge_count > 12:
                                break
                    if edge_count > 12:
                        break
                if edge_count != 12:
                    continue
                    
                # Check degrees in induced subgraph
                valid = True
                for u in A:
                    d = 0
                    for v in B:
                        if adj[u][v]:
                            d += 1
                    if d != 3:
                        valid = False
                        break
                if not valid:
                    continue
                for u in B:
                    d = 0
                    for v in A:
                        if adj[u][v]:
                            d += 1
                    if d != 3:
                        valid = False
                        break
                if not valid:
                    continue
                    
                # If we get here, it's a Q3
                return True
                
        return False

    def find_q3(graph):
        """
        Returns a set of edges forming a Q3, or None.
        """
        adj = [[False] * n for _ in range(n)]
        for u, v in graph:
            adj[u][v] = True
            adj[v][u] = True
            
        deg = [0] * n
        for u, v in graph:
            deg[u] += 1
            deg[v] += 1
            
        candidates = [i for i in range(n) if deg[i] >= 3]
        
        if len(candidates) < 8:
            return None
            
        for A in itertools.combinations(candidates, 4):
            A_set = set(A)
            B_candidates = [v for v in candidates if v not in A_set]
            for B in itertools.combinations(B_candidates, 4):
                edge_count = 0
                edges = []
                for u in A:
                    for v in B:
                        if adj[u][v]:
                            edge_count += 1
                            edges.append((min(u,v), max(u,v)))
                            if edge_count > 12:
                                break
                    if edge_count > 12:
                        break
                if edge_count != 12:
                    continue
                    
                valid = True
                for u in A:
                    d = 0
                    for v in B:
                        if adj[u][v]:
                            d += 1
                    if d != 3:
                        valid = False
                        break
                if not valid:
                    continue
                for u in B:
                    d = 0
                    for v in A:
                        if adj[u][v]:
                            d += 1
                    if d != 3:
                        valid = False
                        break
                if not valid:
                    continue
                    
                return edges
                
        return None

    # Main search loop
    for attempt in range(100):
        if time.time() > deadline:
            break
            
        # Start with a random graph
        p = 0.8 + 0.1 * (attempt % 5) / 5.0
        graph = set()
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < p:
                    graph.add((u, v))
                    
        # Remove Q3s
        while True:
            q3_edges = find_q3(graph)
            if q3_edges is None:
                break
            # Remove a random edge from the Q3
            edge_to_remove = random.choice(q3_edges)
            graph.discard(edge_to_remove)
            
        # Try to add edges back
        # Generate all non-edges
        all_edges = set((u, v) for u in range(n) for v in range(u+1, n))
        non_edges = list(all_edges - graph)
        random.shuffle(non_edges)
        
        for edge in non_edges:
            if time.time() > deadline:
                break
            u, v = edge
            new_graph = graph | {edge}
            if not contains_q3(new_graph):
                graph = new_graph
                
        if len(graph) > best_edges:
            best_edges = len(graph)
            best_graph = graph
            
        # Early termination if we reach a high number
        if best_edges >= 60:
            break
            
    # Convert best_graph to list of lists
    return [[u, v] for u, v in best_graph]

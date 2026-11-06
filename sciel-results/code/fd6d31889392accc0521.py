    def has_q3(vertices, edge_set):
        """
        Check if the 8 vertices contain a Q3 subgraph.
        """
        # Degree check in induced subgraph
        # Q3 is 3-regular. So every vertex in the Q3 must have degree >= 3 in the induced subgraph?
        # No, in the *graph*, the degree can be higher.
        # But in the *induced* subgraph on the 8 vertices, if a vertex has degree < 3, it cannot be part of a Q3 subgraph *within these 8 vertices*?
        # Yes, because the Q3 subgraph uses only edges between these 8 vertices.
        # So if deg(v) < 3 in the induced subgraph, v cannot be in a Q3 subgraph of this induced graph.
        
        deg = {v: 0 for v in vertices}
        for a in vertices:
            for b in vertices:
                if a == b: continue
                u, v = (a, b) if a < b else (b, a)
                if (u, v) in edge_set:
                    deg[a] += 1
                    deg[b] += 1
        # Wait, this double counts.
        # Let's just count edges.
        induced_edges = []
        for a, b in combinations(vertices, 2):
            if (a, b) in edge_set:
                induced_edges.append((a, b))
                deg[a] += 1
                deg[b] += 1
        
        for v in vertices:
            if deg[v] < 3:
                return False
        
        # Now check bipartitions
        for side_a in combinations(vertices, 4):
            side_a_set = set(side_a)
            side_b = tuple(v for v in vertices if v not in side_a_set)
            
            # Quick check: In Q3, each vertex in A is connected to 3 in B.
            # So for each a in A, count edges to B.
            # If any a has < 3 edges to B, this partition is invalid.
            valid_partition = True
            for a in side_a:
                count = 0
                for b in side_b:
                    u, v = (a, b) if a < b else (b, a)
                    if (u, v) in edge_set:
                        count += 1
                if count < 3:
                    valid_partition = False
                    break
            if not valid_partition:
                continue
            
            # Build non-edge matrix
            non_edge = [[False]*4 for _ in range(4)]
            for i, a in enumerate(side_a):
                for j, b in enumerate(side_b):
                    u, v = (a, b) if a < b else (b, a)
                    if (u, v) not in edge_set:
                        non_edge[i][j] = True
            
            # Check for perfect matching in non_edge
            for p in permutations(range(4)):
                if all(non_edge[i][p[i]] for i in range(4)):
                    return True
        return False

import itertools
import random

def construct():
    n = 16
    # 1. Generate all edges of K16
    all_edges = set()
    for u in range(n):
        for v in range(u+1, n):
            all_edges.add((u, v))
    
    graph_edges = set(all_edges)
    
    # 2. Generate Q3 patterns on 0..7
    # Q3 edges: (i, j) where popcount(i^j) == 1
    q3_local_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            if bin(i ^ j).count('1') == 1:
                q3_local_edges.append((i, j))
                
    # Generate all unique embeddings (edge sets) of Q3 on 0..7
    # There are 8! / 48 = 840 such sets
    q3_patterns = set()
    # To generate all 840, we can iterate permutations of 0..7
    # But 40320 permutations is fine.
    for perm in itertools.permutations(range(8)):
        # perm maps local index 0..7 to vertex 0..7
        # We need the set of edges in the image
        edges_in_perm = frozenset()
        # Actually, we just need the set of edges between the permuted vertices
        # The structure is fixed. The edges are (perm[i], perm[j]) for (i,j) in q3_local_edges
        # Wait, the pattern is the set of edges in the subgraph induced by the vertices 0..7
        # that forms a Q3.
        # The set of edges is { (perm[i], perm[j]) for (i,j) in q3_local_edges }
        # Note: (u,v) should be normalized (min, max) for set consistency?
        # No, the pattern is just the set of pairs.
        # When checking against graph_edges, we will normalize.
        
        # Let's store the pattern as a frozenset of (u, v) pairs where u,v in 0..7
        # and (u,v) corresponds to an edge in Q3.
        # The mapping is: local vertex i maps to global vertex perm[i].
        # So the edge (i, j) in Q3 maps to (perm[i], perm[j]).
        
        pattern_edges = frozenset( (min(perm[i], perm[j]), max(perm[i], perm[j])) for i, j in q3_local_edges )
        q3_patterns.add(pattern_edges)
        
    q3_patterns_list = list(q3_patterns)
    random.shuffle(q3_patterns_list)
    
    # 3. Generate all vertex sets of size 8
    vertex_sets = list(itertools.combinations(range(n), 8))
    
    # 4. Greedy removal loop
    # To speed up, we can limit the number of iterations or just let it run.
    # Each iteration finds one Q3 and removes one edge.
    # Max edges to remove: 120 - 40 = 80.
    
    while True:
        # Shuffle vertex sets to avoid bias? 
        # Shuffling 12k items is fast.
        random.shuffle(vertex_sets)
        
        found_q3 = False
        removed_edge = None
        
        for vs in vertex_sets:
            # vs is a tuple of 8 vertices
            # Check if any q3_pattern fits in vs
            for pattern in q3_patterns_list:
                # Check if all edges in pattern exist in graph_edges, mapped by vs
                # pattern contains edges (u, v) where u, v are indices 0..7?
                # NO. In my generation above, pattern_edges contains pairs of values from perm, 
                # which are 0..7.
                # So pattern edges are between vertices 0..7.
                # We need to map pattern vertex x to vs[x].
                
                # Optimization: Check edges one by one
                valid = True
                for u, v in pattern:
                    # Map pattern vertices u, v to actual vertices vs[u], vs[v]
                    gu = vs[u]
                    gv = vs[v]
                    edge = (min(gu, gv), max(gu, gv))
                    if edge not in graph_edges:
                        valid = False
                        break
                
                if valid:
                    found_q3 = True
                    # Pick a random edge from this Q3 to remove
                    # Get the mapped edges
                    mapped_edges = []
                    for u, v in pattern:
                        gu = vs[u]
                        gv = vs[v]
                        mapped_edges.append((min(gu, gv), max(gu, gv)))
                    
                    removed_edge = random.choice(mapped_edges)
                    break
            
            if found_q3:
                break
        
        if not found_q3:
            break
        
        graph_edges.remove(removed_edge)
        
    return [list(e) for e in graph_edges]

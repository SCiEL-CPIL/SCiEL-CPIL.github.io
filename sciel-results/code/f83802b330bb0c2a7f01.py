import itertools

def construct():
    # Construct Mycielski of K4
    # K4 vertices: 0, 1, 2, 3
    # Edges: all pairs
    edge_set = set()
    
    # Original K4
    for i in range(4):
        for j in range(i + 1, 4):
            edge_set.add((i, j))
            edge_set.add((j, i))
            
    # Copies 4, 5, 6, 7
    # Copy i+4 connected to neighbors of i in K4
    # Neighbors of 0: 1, 2, 3
    # Neighbors of 1: 0, 2, 3
    # Neighbors of 2: 0, 1, 3
    # Neighbors of 3: 0, 1, 2
    for i in range(4):
        for j in range(4):
            if i != j:
                edge_set.add((i + 4, j))
                edge_set.add((j, i + 4))
                
    # Apex 8 connected to copies 4, 5, 6, 7
    for i in range(4, 8):
        edge_set.add((8, i))
        edge_set.add((i, 8))
        
    current_vertices = list(range(9))
    
    def find_mis(vertices, e_set):
        best_set = []
        for r in range(len(vertices), -1, -1):
            for subset in itertools.combinations(vertices, r):
                is_indep = True
                for i in range(len(subset)):
                    for j in range(i + 1, len(subset)):
                        u, v = subset[i], subset[j]
                        if (u, v) in e_set or (v, u) in e_set:
                            is_indep = False
                            break
                    if not is_indep:
                        break
                if is_indep:
                    return list(subset)
        return best_set

    # Add vertices 9 to 15
    for new_v in range(9, 16):
        mis = find_mis(current_vertices, edge_set)
        for u in mis:
            edge_set.add((new_v, u))
            edge_set.add((u, new_v))
        current_vertices.append(new_v)
        
    final_edges = []
    for u, v in edge_set:
        if u < v:
            final_edges.append([u, v])
            
    return final_edges

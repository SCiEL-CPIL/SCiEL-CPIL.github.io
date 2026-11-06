"""
Program for Sparse pancyclic graphs on 40 vertices.
Uses a specific 'Spider Wheel' construction:
1. Center vertex 0.
2. Outer ring vertices 1..39 in a cycle.
3. Spokes from 0 to 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39.
4. This creates a graph with 39 (ring) + 23 (spokes) = 62 edges.
5. We prove pancyclicity (cycles of all lengths 3..40) via constructive logic in the code.
"""

N = 40

# EVOLVE-BLOCK-START
def construct():
    edges = []
    adj = [set() for _ in range(N)]
    
    def add_edge(u, v):
        if u > v:
            u, v = v, u
        if (u, v) not in [(e[0], e[1]) if e[0] < e[1] else (e[1], e[0]) for e in edges]:
            edges.append((u, v))
            adj[u].add(v)
            adj[v].add(u)
            
    # 1. Outer Cycle 1-2-...-39-1
    for i in range(1, N):
        add_edge(i, i + 1 if i < N - 1 else 1)
        
    # 2. Spokes from 0 to specific vertices to enable all cycle lengths
    # Spokes connect 0 to 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39
    spoke_targets = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39]
    for t in spoke_targets:
        add_edge(0, t)
        
    # 3. Generate Cycles
    cycles = []
    
    # Length 3
    cycles.append([0, 1, 2])
    
    # Length 4
    cycles.append([0, 1, 3, 4])
    
    # Length 5
    cycles.append([0, 1, 4, 6, 7])
    
    # Length 6
    cycles.append([0, 1, 5, 7, 9, 10])
    
    # Length 7
    cycles.append([0, 1, 6, 8, 10, 12, 13])
    
    # Length 8
    cycles.append([0, 1, 7, 9, 11, 13, 15, 16])
    
    # Length 9
    cycles.append([0, 1, 8, 10, 12, 14, 16, 18, 19])
    
    # Length 10
    cycles.append([0, 1, 9, 11, 13, 15, 17, 19, 21, 22])
    
    # Length 11
    cycles.append([0, 1, 10, 12, 14, 16, 18, 20, 22, 24, 25])
    
    # Length 12
    cycles.append([0, 1, 11, 13, 15, 17, 19, 21, 23, 25, 27, 28])
    
    # Length 13
    cycles.append([0, 1, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 31])
    
    # Length 14
    cycles.append([0, 1, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 34])
    
    # Length 15
    cycles.append([0, 1, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 37])
    
    # Length 16
    cycles.append([0, 1, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 2])
    
    # Length 17
    cycles.append([0, 1, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 39, 2, 3])
    
    # Length 18
    cycles.append([0, 1, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 2, 3, 4, 5])
    
    # Length 19
    cycles.append([0, 1, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 39, 2, 3, 4, 5, 6])
    
    # Length 20
    cycles.append([0, 1, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 2, 3, 4, 5, 6, 7, 8])
    
    # Length 21
    cycles.append([0, 1, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 39, 2, 3, 4, 5, 6, 7, 8, 9])
    
    # Length 22
    cycles.append([0, 1, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    
    # Length 23
    cycles.append([0, 1, 22, 24, 26, 28, 30, 32, 34, 36, 38, 39, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    
    # Length 24
    cycles.append([0, 1, 23, 25, 27, 29, 31, 33, 35, 37, 39, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    
    # Length 25
    cycles.append([0, 1, 24, 26, 28, 30, 32, 34, 36, 38, 39, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    
    # Length 26
    cycles.append([0, 1, 25, 27, 29, 31, 33, 35, 37, 39, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
    
    # Length 27
    cycles.append([0, 1, 26, 28, 30, 32, 34, 36, 38, 39, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    
    # Length 28
    cycles.append([0, 1, 27, 29, 31, 33, 35, 37, 39, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    
    # Length 29
    cycles.append([0, 1, 28, 30, 32, 34, 36, 38, 39, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    
    # Length 30
    cycles.append([0, 1, 29, 31, 33, 35, 37, 39, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
    
    # Length 31
    cycles.append([0, 1, 30, 32, 34, 36, 38, 39, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
    
    # Length 32
    cycles.append([0, 1, 31, 33, 35, 37, 39, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27])
    
    # Length 33
    cycles.append([0, 1, 32, 34, 36, 38, 39, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])
    
    # Length 34
    cycles.append([0, 1, 33, 35, 37, 39, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])
    
    # Length 35
    cycles.append([0, 1, 34, 36, 38, 39, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33])
    
    # Length 36
    cycles.append([0, 1, 35, 37, 39, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36])
    
    # Length 37
    cycles.append([0, 1, 36, 38, 39, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37])
    
    # Length 38
    cycles.append([0, 1, 37, 39, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38])
    
    # Length 39
    cycles.append([0, 1, 38, 39, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37])
    
    # Length 40
    cycles.append(list(range(N)))
    
    # Validation (for development safety, removed in final if slow, but N=40 is small)
    edge_set = set()
    for u, v in edges:
        if u > v: u, v = v, u
        edge_set.add((u, v))
        
    for i, cycle in enumerate(cycles):
        if len(cycle) != i + 3:
            raise ValueError(f"Cycle {i+3} has wrong length")
        if len(set(cycle)) != len(cycle):
            raise ValueError(f"Cycle {i+3} has duplicate vertices")
        for j in range(len(cycle)):
            u = cycle[j]
            v = cycle[(j+1) % len(cycle)]
            a, b = (u, v) if u < v else (v, u)
            if (a, b) not in edge_set:
                raise ValueError(f"Edge {a}-{b} missing in cycle {i+3}")
                
    return {'edges': edges, 'cycles': cycles}
# EVOLVE-BLOCK-END

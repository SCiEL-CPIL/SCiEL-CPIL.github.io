N = 40

def construct():
    edges = []
    edge_set = set()
    
    def add_edge(u, v):
        if u > v:
            u, v = v, u
        if (u, v) not in edge_set:
            edge_set.add((u, v))
            edges.append((u, v))
    
    # Ladder graph: top path 0..19, bottom path 20..39, rungs (i, i+20)
    # Top path: 19 edges
    for i in range(19):
        add_edge(i, i + 1)
    
    # Bottom path: 19 edges
    for i in range(19):
        add_edge(i + 20, i + 21)
    
    # Rungs: 20 edges
    for i in range(20):
        add_edge(i, i + 20)
    
    # One extra chord to create triangle: (0, 21)
    add_edge(0, 21)
    
    # Total: 19 + 19 + 20 + 1 = 59 edges
    
    # Construct cycles of every length 3 to 40
    cycles = []
    
    for k in range(3, 41):
        if k % 2 == 1:
            # Odd cycle using chord (0, 21)
            # 0-1-...-m-(m+20)-(m+19)-...-21-0
            # Length = m + 1 + (m-1) + 1 = 2m + 1 => m = (k-1)//2
            m = (k - 1) // 2
            cyc = list(range(0, m + 1))
            cyc.extend(range(m + 20, 20, -1))
            cycles.append(cyc)
        else:
            # Even cycle using ladder structure
            # 0-1-...-m-(m+20)-(m+19)-...-20-0
            # Length = m + 1 + m + 1 = 2m + 2 => m = k//2 - 1
            m = k // 2 - 1
            cyc = list(range(0, m + 1))
            cyc.extend(range(m + 20, 19, -1))
            cycles.append(cyc)
    
    return {'edges': edges, 'cycles': cycles}

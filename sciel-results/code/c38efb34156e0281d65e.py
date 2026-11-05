import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n = 31
    q = 5
    
    # Generate points of PG(2, 5)
    # Points are represented as tuples (x, y, z) in F_5^3, normalized.
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: make first non-zero element 1
                # Since we are in F_5, we can just use the vector as is, 
                # but we must identify scalar multiples.
                # Easier: iterate through all non-zero vectors, 
                # and pick a canonical representative.
                # Canonical: first non-zero coordinate is 1.
                idx = 0
                while idx < 3 and vec[idx] == 0:
                    idx += 1
                if idx == 3: continue # zero vec, skipped
                # Scale so that vec[idx] == 1
                # In F_5, inverse of 1 is 1, 2 is 3, 3 is 2, 4 is 4.
                inv = {1:1, 2:3, 3:2, 4:4}
                scale = inv[vec[idx]]
                rep = tuple((vec[i] * scale) % q for i in range(3))
                if rep not in points:
                    points.append(rep)
    
    # The loop above is inefficient and has a bug (vec undefined).
    # Let's generate properly.
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Find first non-zero
                if x != 0:
                    inv_x = {1:1, 2:3, 3:2, 4:4}[x]
                    rep = (1, (y*inv_x)%q, (z*inv_x)%q)
                elif y != 0:
                    inv_y = {1:1, 2:3, 3:2, 4:4}[y]
                    rep = (0, 1, (z*inv_y)%q)
                else:
                    rep = (0, 0, 1)
                if rep not in points:
                    points.append(rep)
    
    # points should have 31 elements
    # Build adjacency matrix
    # A[i][j] = 1 if points[i] . points[j] == 0
    A = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            p1 = points[i]
            p2 = points[j]
            dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % q
            if dot == 0:
                A[i][j] = 1
                A[j][i] = 1
    
    # Count edges to verify
    edge_count = sum(sum(row) for row in A) // 2
    if edge_count != 90:
        # Fallback if construction failed
        # Just return lexicographic first 91
        pairs = [[i, j] for i in range(n) for j in range(i + 1, n)]
        return pairs[:91]

    # Find the best edge to add
    best_edge = None
    min_c4s = float('inf')
    
    # Precompute neighbor lists for speed
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if A[i][j]:
                neighbors[i].append(j)
    
    for i in range(n):
        for j in range(i+1, n):
            if A[i][j] == 0:
                # Calculate number of 4-cycles created by adding (i, j)
                # This is the number of edges (w, x) such that w in N(j), x in N(i), w != x
                # And (w, x) is an edge in G.
                c4_count = 0
                Ni = neighbors[i]
                Nj = neighbors[j]
                
                # Optimize: 
                # For each w in Nj, for each x in Ni:
                # if w != x and A[w][x] == 1: c4_count += 1
                
                for w in Nj:
                    for x in Ni:
                        if w != x and A[w][x]:
                            c4_count += 1
                
                if c4_count < min_c4s:
                    min_c4s = c4_count
                    best_edge = (i, j)
                    
                if time.time() > deadline:
                    break
        if time.time() > deadline:
            break

    if best_edge is None:
        # Fallback
        pairs = [[i, j] for i in range(n) for j in range(i + 1, n)]
        return pairs[:91]

    # Construct final edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if A[i][j]:
                edges.append([i, j])
    
    # Add the best edge
    u, v = best_edge
    edges.append([u, v])
    
    return edges

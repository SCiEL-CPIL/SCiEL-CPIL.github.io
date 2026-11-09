import numpy as np
import time
import itertools

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # 1. Generate points of PG(2,5)
    q = 5
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: make first non-zero element 1
                vec = [x, y, z]
                first_non_zero_idx = -1
                for i in range(3):
                    if vec[i] != 0:
                        first_non_zero_idx = i
                        break
                if first_non_zero_idx == -1:
                    continue
                inv = pow(vec[first_non_zero_idx], q - 2, q)
                vec = [(v * inv) % q for v in vec]
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    n = len(points)
    if n != 31:
        # Fallback to seed if something goes wrong
        pairs = [[i, j] for i in range(31) for j in range(i + 1, 31)]
        return pairs[:91]

    # 2. Define Polarity Matrix A
    # A = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
    A = np.array([
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ], dtype=int)
    
    # 3. Compute Adjacency
    # adj[i][j] = 1 if points[i] A points[j] == 0 and i != j
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val = 0
            for k in range(3):
                for l in range(3):
                    val = (val + points[i][k] * A[k, l] * points[j][l]) % q
            if val == 0:
                adj[i, j] = 1
    
    # 4. Extract Edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                edges.append([i, j])
                
    # Check edge count
    if len(edges) != 90:
        # If not 90, something is wrong. Try to adjust or fallback.
        # But theory says it should be 90.
        pass

    # 5. Find best edge to add
    # We need to add 1 edge.
    # Minimize number of new 4-cycles.
    # New 4-cycles = number of paths of length 3 between u and v in G.
    
    # Precompute adjacency list for speed
    adj_list = [[] for _ in range(n)]
    for u, v in edges:
        adj_list[u].append(v)
        adj_list[v].append(u)
        
    best_edge = None
    min_cycles = float('inf')
    
    # Iterate over all non-edges
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v] == 1:
                continue
            # Count paths of length 3 from u to v
            # Path u-a-b-v.
            # a is neighbor of u, b is neighbor of v, a-b is edge.
            # Or u-a-b-v means (u,a), (a,b), (b,v).
            # We can iterate a in neighbors(u), b in neighbors(v).
            # Check if (a,b) is edge.
            count = 0
            for a in adj_list[u]:
                for b in adj_list[v]:
                    if adj[a, b] == 1:
                        count += 1
            # Note: A 4-cycle is a set of 4 vertices.
            # The cycle is u-v-b-a-u? No.
            # The cycle uses edges (u,v) [new], (v,b), (b,a), (a,u).
            # So the path is u-a-b-v.
            # Each such path corresponds to a unique 4-cycle involving (u,v).
            # Are there duplicates?
            # If u-a-b-v is a path, the cycle is u-v-b-a-u.
            # Is it possible that another path u-c-d-v gives the same cycle?
            # Only if {a,b} = {c,d} and order is reversed?
            # Path u-a-b-v -> cycle u-v-b-a-u.
            # Path u-b-a-v -> cycle u-v-a-b-u.
            # These are the same set of edges: {(u,v), (v,b), (b,a), (a,u)}.
            # So if (a,b) is an edge, both (a in N(u), b in N(v)) and (b in N(u), a in N(v)) might be counted?
            # No, a must be in N(u) and b in N(v).
            # If we swap a and b, we need b in N(u) and a in N(v).
            # This is a different pair (b,a).
            # Does it correspond to the same cycle?
            # Cycle 1: u-v-b-a-u. Edges: (u,v), (v,b), (b,a), (a,u).
            # Cycle 2: u-v-a-b-u. Edges: (u,v), (v,a), (a,b), (b,u).
            # These are different sets of edges unless (v,b)=(v,a) which implies a=b, impossible.
            # So each valid path u-a-b-v corresponds to a distinct 4-cycle?
            # Wait. The cycle is defined by the 4 vertices {u,v,a,b}.
            # The edges are (u,v), (v,b), (b,a), (a,u).
            # If we found path u-a-b-v, the edges are (u,a), (a,b), (b,v).
            # So the cycle is u-a-b-v-u.
            # This uses edges (u,a), (a,b), (b,v), (v,u).
            # So yes, each path u-a-b-v defines a unique 4-cycle.
            # Can two different paths define the same 4-cycle?
            # Only if the set of vertices is the same and the edges are the same.
            # Path u-a-b-v uses (u,a), (a,b), (b,v).
            # Path u-c-d-v uses (u,c), (c,d), (d,v).
            # If the cycles are the same, the edges must be the same.
            # So {u,a}={u,c} => a=c.
            # Then {a,b}={c,d} => b=d.
            # So the path is unique.
            
            if count < min_cycles:
                min_cycles = count
                best_edge = [u, v]
                
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    if best_edge is None:
        # Fallback
        pairs = [[i, j] for i in range(31) for j in range(i + 1, 31)]
        return pairs[:91]
        
    edges.append(best_edge)
    return edges

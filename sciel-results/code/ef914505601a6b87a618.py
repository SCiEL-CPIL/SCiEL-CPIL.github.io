import time
import random
import numpy as np

def construct():
    n = 31
    target_edges = 91
    start_time = time.time()
    deadline = start_time + 110.0

    # 1. Construct Polarity Graph of PG(2,5)
    q = 5
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    key = (1, y * pow(x, q-2, q) % q, z * pow(x, q-2, q) % q)
                elif y != 0:
                    key = (0, 1, z * pow(y, q-2, q) % q)
                else:
                    key = (0, 0, 1)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    def is_on_polar(p, qpt):
        x, y, z = p
        X, Y, Z = qpt
        return (x * Y + y * X + z * Z) % q == 0

    edges = set()
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i + 1, n):
            if is_on_polar(points[i], points[j]):
                edges.add((i, j))
                adj[i, j] = 1
                adj[j, i] = 1
    
    assert len(edges) == 90, f"Expected 90 edges, got {len(edges)}"

    # Helper to count C4s using adjacency matrix
    # C4 = (tr(A^4) - 2m - 4*sum(deg_i choose 2)) / 8
    # tr(A^4) = sum(A^4[i,i])
    # A^4[i,i] is the number of closed walks of length 4 starting/ending at i.
    # Closed walks of length 4 are:
    # 1. i-j-i-j-i (back and forth) -> contributes 2 for each edge incident to i? No.
    #    Let's use the standard formula:
    #    Number of C4s = 1/8 * (sum_{i,j} (A^2[i,j])^2 - 2m - 4*sum_i C(d_i, 2))
    #    Wait, sum (A^2[i,j])^2 counts:
    #    - 4-cycles: each C4 contributes 8 (4 vertices * 2 directions? No, 4 vertices * 2 ways to traverse? Actually each C4 has 4 vertices. For each vertex, it sees 2 neighbors. The pair of neighbors is common.
    #    Let's stick to the reliable formula:
    #    C4 = (sum_{i,j} (A^2_{ij})^2 - 2m - 4 \sum_i \binom{d_i}{2}) / 8
    
    def count_c4(adj_mat):
        # adj_mat is n x n int8
        deg = adj_mat.sum(axis=1)
        # A^2
        A2 = adj_mat @ adj_mat
        # sum of squares of A2
        sum_sq = np.sum(A2 * A2)
        # 2m
        m = deg.sum() // 2
        # sum of C(d_i, 2) = sum(d_i*(d_i-1)/2)
        sum_choose2 = np.sum(deg * (deg - 1)) // 2
        
        return (sum_sq - 2 * m - 4 * sum_choose2) // 8

    # 2. Find best edge to add to reach 91 edges
    current_c4 = count_c4(adj)
    
    best_add = None
    min_new_c4 = float('inf')
    
    # Precompute A2 for speed? No, we need to update it or recompute.
    # Since we only add 1 edge, we can just test all non-edges.
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0:
                # Temporarily add edge
                adj[i, j] = 1
                adj[j, i] = 1
                c4_val = count_c4(adj)
                if c4_val < min_new_c4:
                    min_new_c4 = c4_val
                    best_add = (i, j)
                # Revert
                adj[i, j] = 0
                adj[j, i] = 0
                
    u_add, v_add = best_add
    adj[u_add, v_add] = 1
    adj[v_add, u_add] = 1
    edges.add((u_add, v_add))
    
    current_c4 = min_new_c4
    best_c4 = current_c4
    best_adj = adj.copy()

    random.seed(42)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # 3. Local Search
    # We perform 1-in-1-out swaps.
    # To speed up, we can approximate the change in C4s.
    # Delta C4 for removing (u,v) and adding (x,y) is complex.
    # Given n=31, count_c4 is very fast (~microseconds).
    # We can afford many iterations.
    
    stall_counter = 0
    stall_threshold = 500
    
    while time.time() < deadline - 5.0:
        improved = False
        # Pick a random edge to remove
        edge_list = list(edges)
        u1, v1 = random.choice(edge_list)
        
        # Try random non-edges to add
        # To ensure we don't pick the same edge, we pick a random pair
        # and check if it's in edges.
        
        # Optimization: Pre-calculate "potential" gains?
        # For now, random sampling is robust enough for n=31.
        
        for _ in range(50): # Try 50 random swaps per iteration
            u2, v2 = random.choice(all_pairs)
            if (u2, v2) in edges or (v2, u2) in edges:
                continue
            if (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                continue
            
            # Perform swap
            adj[u1, v1] = 0
            adj[v1, u1] = 0
            adj[u2, v2] = 1
            adj[v2, u2] = 1
            
            edges.discard((u1, v1))
            edges.add((u2, v2))
            
            new_c4 = count_c4(adj)
            
            if new_c4 < current_c4:
                current_c4 = new_c4
                improved = True
                if current_c4 < best_c4:
                    best_c4 = current_c4
                    best_adj = adj.copy()
                    stall_counter = 0
                break
            else:
                # Revert
                adj[u1, v1] = 1
                adj[v1, u1] = 1
                adj[u2, v2] = 0
                adj[v2, u2] = 0
                
                edges.discard((u2, v2))
                edges.add((u1, v1))
                
        if not improved:
            stall_counter += 1
        else:
            stall_counter = max(0, stall_counter - 1)
            
        if stall_counter >= stall_threshold:
            # Restart with perturbation from best
            adj = best_adj.copy()
            edges = set()
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i, j] == 1:
                        edges.add((i, j))
            current_c4 = best_c4
            
            # Perturb: remove 2 random edges, add 2 random non-edges
            for _ in range(2):
                e1 = random.choice(list(edges))
                u1, v1 = e1
                adj[u1, v1] = 0
                adj[v1, u1] = 0
                edges.discard(e1)
                
                # Find a non-edge to add
                while True:
                    u2, v2 = random.choice(all_pairs)
                    if (u2, v2) not in edges and (v2, u2) not in edges:
                        break
                adj[u2, v2] = 1
                adj[v2, u2] = 1
                edges.add((u2, v2))
                
            current_c4 = count_c4(adj)
            stall_counter = 0

    # Ensure we return exactly 91 edges
    # If we somehow lost an edge, add a random one (should not happen with swaps)
    while len(edges) < target_edges:
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) not in edges and (j, i) not in edges and adj[i, j] == 0:
                    edges.add((i, j))
                    adj[i, j] = 1
                    adj[j, i] = 1
                    break
            if len(edges) == target_edges:
                break
    while len(edges) > target_edges:
        e = random.choice(list(edges))
        edges.discard(e)
        adj[e[0], e[1]] = 0
        adj[e[1], e[0]] = 0

    return [[u, v] for u, v in sorted(edges)]

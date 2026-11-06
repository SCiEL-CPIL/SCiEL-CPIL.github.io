import numpy as np
import time
import random
import math

def count_c4(adj):
    """
    Count the number of 4-cycles in the graph represented by adjacency matrix adj.
    """
    n = adj.shape[0]
    A2 = adj @ adj
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    vals = A2[mask]
    return int(np.sum(vals * (vals - 1)) // 2)

def construct():
    n = 31
    m = 91
    start_time = time.time()
    deadline = start_time + 140  # 10s margin from 150s limit

    # 1. Construct the base graph: Orthogonal Polarity Graph of PG(2,5)
    # This graph has 31 vertices.
    # Points in PG(2,5) represented as (x, y, z) in GF(5)^3, normalized.
    # We use the standard representative:
    # (1, y, z) for y, z in 0..4 (25 points)
    # (0, 1, z) for z in 0..4 (5 points)
    # (0, 0, 1) (1 point)
    # Total 31 points.
    
    points = []
    # Index 0-24: (1, y, z)
    for y in range(5):
        for z in range(5):
            points.append((1, y, z))
    # Index 25-29: (0, 1, z)
    for z in range(5):
        points.append((0, 1, z))
    # Index 30: (0, 0, 1)
    points.append((0, 0, 1))
    
    adj = np.zeros((n, n), dtype=np.int8)
    edge_set = set()
    
    # Define polarity: P and Q are adjacent if dot product is 0 in GF(5)
    # dot(P, Q) = x1*x2 + y1*y2 + z1*z2 mod 5 == 0
    # Note: This is a symmetric bilinear form.
    # If dot(P, P) != 0, P is not absolute (no loop).
    # If dot(P, P) == 0, P is absolute. In simple graph, we don't add loop, but P is on its polar line.
    # The edge set is {(P, Q) : P != Q, dot(P, Q) == 0}.
    
    for i in range(n):
        for j in range(i + 1, n):
            p = points[i]
            q = points[j]
            dot = (p[0]*q[0] + p[1]*q[1] + p[2]*q[2]) % 5
            if dot == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                edge_set.add((i, j))
    
    current_edges = len(edge_set)
    # For GF(5) orthogonal polarity, we expect 81 edges.
    # We need 91 edges. So we need to add 10 more edges.
    
    # Calculate initial C4 count
    current_c4 = count_c4(adj)
    
    # 2. Greedy addition of remaining edges
    # We need to add (m - current_edges) edges.
    # For each potential edge not in edge_set, calculate the number of new C4s it creates.
    # New C4s involving (u, v) = number of pairs (w, x) such that (u, w), (w, x), (x, v) are edges?
    # Actually, adding (u, v) creates a C4 for every common neighbor pair?
    # A C4 is u-w-x-v-u.
    # If we add (u, v), the new C4s are those where (u, v) is one of the edges.
    # The C4s containing (u, v) are determined by the paths of length 3 from u to v?
    # No, a C4 is a cycle of length 4.
    # If we add edge (u, v), any existing path u-w-x-v becomes a C4 u-w-x-v-u.
    # So new C4s = number of length-3 paths from u to v.
    # Number of length-3 paths from u to v = sum_{w in N(u)} (A^2)_{w, v} ? No.
    # (A^3)_{u, v} counts walks of length 3.
    # Since the graph is simple and we are adding an edge, we just need to count the number of common neighbors of neighbors?
    # Simpler: new_c4s(u, v) = sum_{w} A[u, w] * (A^2)[w, v] ?
    # Let's just use the full recount for simplicity as n=31 is small, but we can be smarter.
    # Smart check:
    # C4s involving (u, v) = 1/2 * ( number of common neighbors of u and v )? No, that's for C4s where (u, v) is a diagonal.
    # If (u, v) is an edge, the C4s it is part of are u-w-x-v.
    # w is a neighbor of u, x is a neighbor of v, and (w, x) is an edge.
    # So new C4s = number of edges (w, x) where w in N(u) \ {v} and x in N(v) \ {u}.
    
    def count_new_c4s(u, v, adj):
        # u and v are not currently connected
        # Find neighbors of u and v
        nu = np.where(adj[u] == 1)[0]
        nv = np.where(adj[v] == 1)[0]
        
        # We need to count edges (w, x) with w in nu, x in nv
        # Note: w != v, x != u is guaranteed if (u, v) is not an edge yet?
        # If w=v, then v is in nu, meaning (u, v) is an edge. But we are checking non-edges.
        # So nu does not contain v. nv does not contain u.
        
        count = 0
        # Optimize: iterate over smaller set
        if len(nu) < len(nv):
            for w in nu:
                # Check neighbors of w in nv
                # adj[w, nv] gives 1 if edge exists
                count += np.sum(adj[w, nv])
        else:
            for x in nv:
                count += np.sum(adj[x, nu])
        
        # Each C4 u-w-x-v is counted once here?
        # The C4 is u-w-x-v-u.
        # The edge (w, x) is unique.
        # So this count is exactly the number of new C4s.
        return count

    # Candidate edges
    all_possible_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edge_set]
    
    edges_to_add = m - current_edges
    for _ in range(edges_to_add):
        if time.time() > deadline:
            break
        best_edge = None
        min_c4s = float('inf')
        
        # Shuffle to break ties randomly
        random.shuffle(all_possible_edges)
        
        for u, v in all_possible_edges:
            c4s = count_new_c4s(u, v, adj)
            if c4s < min_c4s:
                min_c4s = c4s
                best_edge = (u, v)
        
        if best_edge is not None:
            u, v = best_edge
            adj[u, v] = 1
            adj[v, u] = 1
            edge_set.add(best_edge)
            # Remove from candidate list
            # O(N) removal, but N is small (465-81=384)
            all_possible_edges.remove(best_edge)
    
    current_c4 = count_c4(adj)
    best_c4 = current_c4
    best_adj = adj.copy()
    best_edge_set = set(edge_set)
    
    # 3. Local Search (Simulated Annealing) to refine
    # Swap edges to reduce C4 count.
    # Since we started from a geometric structure, we are likely close to optimal.
    # We will use a simple hill-climbing with restarts or SA.
    
    T = 5.0
    T_min = 0.1
    cooling_rate = 0.995
    
    # Precompute list of current edges for fast swap generation
    current_edges_list = list(edge_set)
    
    steps = 0
    while T > T_min and time.time() < deadline:
        steps += 1
        # Pick a random edge to remove
        idx_rm = random.randint(0, len(current_edges_list) - 1)
        e_rm = current_edges_list[idx_rm]
        u1, v1 = e_rm
        
        # Pick a random edge to add
        # We can pick from all non-edges.
        # To avoid expensive checks, we can pick a random non-edge.
        # Generating a random non-edge:
        # Total edges possible: 465. Current: 91. Non-edges: 374.
        # Just pick random u, v and check.
        while True:
            u2 = random.randint(0, n-1)
            v2 = random.randint(0, n-1)
            if u2 == v2:
                continue
            if u2 > v2:
                u2, v2 = v2, u2
            e_add = (u2, v2)
            if e_add not in edge_set:
                break
        
        # Apply swap
        adj[u1, v1] = 0
        adj[v1, u1] = 0
        adj[u2, v2] = 1
        adj[v2, u2] = 1
        
        new_c4 = count_c4(adj)
        delta = new_c4 - current_c4
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            current_c4 = new_c4
            edge_set.remove(e_rm)
            edge_set.add(e_add)
            current_edges_list[idx_rm] = e_add
            
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_adj = adj.copy()
                best_edge_set = set(edge_set)
        else:
            # Reject
            adj[u1, v1] = 1
            adj[v1, u1] = 1
            adj[u2, v2] = 0
            adj[v2, u2] = 0
            
        T *= cooling_rate
        
    # Return the best found
    if best_edge_set:
        return [[u, v] for u, v in best_edge_set]
    else:
        # Fallback
        return [[i, j] for i in range(31) for j in range(i+1, 31)][:91]

# To make the local search faster, we can optimize count_c4 for the specific swap?
# But for n=31, count_c4 is ~microseconds. The loop overhead is dominant.
# We can try to run this.

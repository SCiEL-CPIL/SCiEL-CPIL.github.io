import numpy as np
import time
from collections import deque

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 50  # Safety margin

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)

    # Identify edge types
    c1_edges = []
    c2_edges = []
    m_edges = []

    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            c1_edges.append(i)
        elif u >= m and v >= m:
            c2_edges.append(i)
        else:
            m_edges.append(i)

    # Function to evaluate a set of signs
    def evaluate_signs(signs):
        # Check connectivity using BFS on the lift
        lift_n = 2 * n
        adj_lift = [[] for _ in range(lift_n)]
        
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            if s == 1:
                adj_lift[u*2].append(v*2)
                adj_lift[v*2].append(u*2)
                adj_lift[u*2+1].append(v*2+1)
                adj_lift[v*2+1].append(u*2+1)
            else:
                adj_lift[u*2].append(v*2+1)
                adj_lift[v*2+1].append(u*2)
                adj_lift[u*2+1].append(v*2)
                adj_lift[v*2].append(u*2+1)
        
        visited = [False] * lift_n
        queue = deque([0])
        visited[0] = True
        while queue:
            u = queue.popleft()
            for v in adj_lift[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        
        if not all(visited):
            return float('inf')
            
        # Build signed adjacency matrix of base graph
        A = np.zeros((n, n), dtype=float)
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
            
        eigvals = np.linalg.eigvalsh(A)
        return np.max(np.abs(eigvals))

    best_radius = float('inf')
    best_signs = None

    def try_signs(signs):
        nonlocal best_radius, best_signs
        r = evaluate_signs(signs)
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()

    # Helper to get cycle edge order
    def get_cycle_edges_indices(edge_indices, offset=0):
        pos_map = {}
        for idx in edge_indices:
            u, v = edges[idx]
            u = u - offset
            v = v - offset
            if (u + 1) % m == v:
                pos = u
            elif (v + 1) % m == u:
                pos = v
            else:
                continue
            pos_map[pos] = idx
        return sorted(pos_map.items())

    c1_ordered = get_cycle_edges_indices(c1_edges, 0)
    c2_ordered = get_cycle_edges_indices(c2_edges, m)
    m_ordered = sorted(m_edges, key=lambda i: edges[i][0] if edges[i][0] < m else edges[i][1])

    # Generate initial candidates based on structural patterns
    candidates = []
    
    # Pattern 1: Alt C1, Alt C2, -1 M
    s1 = np.ones(num_edges, dtype=int)
    for pos, idx in c1_ordered: s1[idx] = 1 if pos % 2 == 0 else -1
    for pos, idx in c2_ordered: s1[idx] = 1 if pos % 2 == 0 else -1
    for idx in m_edges: s1[idx] = -1
    candidates.append(s1)
    
    # Pattern 2: Alt C1, Anti-Alt C2, -1 M
    s2 = s1.copy()
    for pos, idx in c2_ordered: s2[idx] = -1 if pos % 2 == 0 else 1
    candidates.append(s2)
    
    # Pattern 3: Alt C1, Alt C2, +1 M
    s3 = s1.copy()
    for idx in m_edges: s3[idx] = 1
    candidates.append(s3)
    
    # Pattern 4: All +1 C1, All +1 C2, Alt M
    s4 = np.ones(num_edges, dtype=int)
    for pos, idx in enumerate(m_ordered): s4[idx] = 1 if pos % 2 == 0 else -1
    candidates.append(s4)
    
    # Pattern 5: Alt C1, Alt C2, Alt M
    s5 = s1.copy()
    for pos, idx in enumerate(m_ordered): s5[idx] = 1 if pos % 2 == 0 else -1
    candidates.append(s5)
    
    # Evaluate initial candidates
    for s in candidates:
        try_signs(s)

    # Local search: flip one edge at a time
    # We can optimize by noting that flipping an edge only changes the spectrum slightly?
    # No, we just recompute. But we can be smarter about which flips to try.
    # Start from the best candidate found so far.
    
    current_signs = best_signs.copy()
    current_radius = best_radius
    
    # Precompute edge endpoints for speed
    edge_u = np.array([u for u, v in edges])
    edge_v = np.array([v for u, v in edges])
    
    # We will perform a few iterations of local search
    # To speed up evaluation, we can use the fact that A is symmetric.
    # But eigvalsh is O(n^3). n is small (8, 10, 14, 18). So it's fast.
    
    # Let's do a simple coordinate descent / hill climbing
    # We will try flipping each edge and see if it improves.
    # If it improves, we keep the flip and restart the scan.
    # We limit the number of iterations to stay within time.
    
    max_iterations = 1000 # Should be plenty for small n
    improved = True
    iter_count = 0
    
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(num_edges):
            # Flip edge i
            current_signs[i] *= -1
            r = evaluate_signs(current_signs)
            if r < current_radius - 1e-9:
                current_radius = r
                best_radius = r
                best_signs = current_signs.copy()
                improved = True
                # We can continue flipping from this new state
                # But restarting the scan might be more thorough?
                # Let's just continue the scan, it's a simple hill climb.
            else:
                # Revert
                current_signs[i] *= -1
        
        iter_count += 1
        if iter_count > max_iterations:
            break

    return best_signs

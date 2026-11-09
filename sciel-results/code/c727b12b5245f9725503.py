import numpy as np
import time
from collections import deque

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)

    # Identify edge types: cycle 1, cycle 2, matching
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

    # Precompute edge endpoints
    edge_u = np.array([u for u, v in edges], dtype=int)
    edge_v = np.array([v for u, v in edges], dtype=int)

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

    # Fast evaluation function
    def evaluate_signs(signs):
        # 1. Check connectivity of the lift
        lift_n = 2 * n
        # We can check connectivity without building full adjacency if we are clever, 
        # but for n<=18, building is fast enough.
        # Actually, the lift is 3-regular. 
        # A 2-lift is connected iff the signed graph is not "balanced" in a specific way?
        # No, it's connected iff the product of signs around every cycle is +1? No, that's for bipartite double cover vs something else.
        # The 2-lift is connected iff the signed graph has no "bipartition" of vertices such that all edges within parts are + and across are -? 
        # Actually, the 2-lift is disconnected iff the signed graph is "bipartite" in the sense that there exists a vector x in {+1, -1}^n 
        # such that x_u * x_v * sign(u,v) = -1 for all edges? 
        # Let's just do BFS on the lift. It's small.
        
        # To speed up BFS, we can precompute the neighbor lists for each sign configuration? 
        # No, signs change.
        
        # Let's build the lift adjacency list.
        # Nodes are 0..2n-1.
        # For edge i (u, v) with sign s:
        # If s=1: (u,0)-(v,0), (u,1)-(v,1)
        # If s=-1: (u,0)-(v,1), (u,1)-(v,0)
        
        # We can represent the lift graph using bitmasks or just lists.
        # Since n is small (max 18), lift_n is 36.
        # E_lift = 2 * E_base = 2 * (3m) = 6m. Max 54 edges.
        
        adj_lift = [[] for _ in range(lift_n)]
        for i in range(num_edges):
            u = edge_u[i]
            v = edge_v[i]
            s = signs[i]
            u0 = u * 2
            u1 = u * 2 + 1
            v0 = v * 2
            v1 = v * 2 + 1
            
            if s == 1:
                adj_lift[u0].append(v0)
                adj_lift[v0].append(u0)
                adj_lift[u1].append(v1)
                adj_lift[v1].append(u1)
            else:
                adj_lift[u0].append(v1)
                adj_lift[v1].append(u0)
                adj_lift[u1].append(v0)
                adj_lift[v0].append(u1)
                
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
            
        # 2. Compute spectral radius
        A = np.zeros((n, n), dtype=float)
        A[edge_u, edge_v] = signs
        A[edge_v, edge_u] = signs
        
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

    # Generate initial candidates based on structural insights
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
    
    # Pattern 6: Random start from seed logic (just in case)
    rng = np.random.default_rng(3)
    for _ in range(5):
        s_rand = rng.choice([-1, 1], size=num_edges).astype(int)
        candidates.append(s_rand)

    # Evaluate initial candidates
    for s in candidates:
        try_signs(s)

    if best_signs is None:
        # Fallback
        return np.ones(num_edges, dtype=int)

    # Local Search
    current_signs = best_signs.copy()
    current_radius = best_radius

    # Phase 1: Single Flip Hill Climb
    max_passes = 20
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(num_edges):
            if time.time() > deadline:
                break
            current_signs[i] *= -1
            r = evaluate_signs(current_signs)
            if r < current_radius - 1e-9:
                current_radius = r
                best_radius = r
                best_signs = current_signs.copy()
                improved = True
            else:
                current_signs[i] *= -1
        if not improved:
            break

    # Phase 2: Pairwise Flip Search
    current_signs = best_signs.copy()
    current_radius = best_radius
    
    max_pair_passes = 5
    for _ in range(max_pair_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(num_edges):
            if time.time() > deadline:
                break
            for j in range(i + 1, num_edges):
                if time.time() > deadline:
                    break
                current_signs[i] *= -1
                current_signs[j] *= -1
                r = evaluate_signs(current_signs)
                if r < current_radius - 1e-9:
                    current_radius = r
                    best_radius = r
                    best_signs = current_signs.copy()
                    improved = True
                    # Restart pairwise search from new best
                    current_signs = best_signs.copy()
                    current_radius = best_radius
                    break
                else:
                    current_signs[i] *= -1
                    current_signs[j] *= -1
            if improved:
                break
        if not improved:
            break

    # Phase 3: Triple Flip Search (limited)
    # Only if time permits
    if time.time() < deadline - 10:
        current_signs = best_signs.copy()
        current_radius = best_radius
        max_triple_passes = 2
        for _ in range(max_triple_passes):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(num_edges):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, num_edges):
                    if time.time() > deadline - 5:
                        break
                    for k in range(j + 1, num_edges):
                        if time.time() > deadline - 5:
                            break
                        current_signs[i] *= -1
                        current_signs[j] *= -1
                        current_signs[k] *= -1
                        r = evaluate_signs(current_signs)
                        if r < current_radius - 1e-9:
                            current_radius = r
                            best_radius = r
                            best_signs = current_signs.copy()
                            improved = True
                            current_signs = best_signs.copy()
                            current_radius = best_radius
                            break
                        else:
                            current_signs[i] *= -1
                            current_signs[j] *= -1
                            current_signs[k] *= -1
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break

    return best_signs

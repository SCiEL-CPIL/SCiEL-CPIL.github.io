import numpy as np
import time
from collections import deque

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit

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

    # Precompute edge endpoints
    edge_u = np.array([u for u, v in edges], dtype=int)
    edge_v = np.array([v for u, v in edges], dtype=int)

    # Precompute lift adjacency for each sign
    lift_adj_plus = [None] * num_edges
    lift_adj_minus = [None] * num_edges

    for i in range(num_edges):
        u = edge_u[i]
        v = edge_v[i]
        u0, u1 = u * 2, u * 2 + 1
        v0, v1 = v * 2, v * 2 + 1

        # Sign +1: (u,0)-(v,0) and (u,1)-(v,1)
        lift_adj_plus[i] = [(u0, v0), (v0, u0), (u1, v1), (v1, u1)]

        # Sign -1: (u,0)-(v,1) and (u,1)-(v,0)
        lift_adj_minus[i] = [(u0, v1), (v1, u0), (u1, v0), (v0, u1)]

    def evaluate_signs(signs):
        # signs is a numpy array of 1s and -1s
        lift_n = 2 * n
        adj_lift = [[] for _ in range(lift_n)]
        for i in range(num_edges):
            if signs[i] == 1:
                for u_l, v_l in lift_adj_plus[i]:
                    adj_lift[u_l].append(v_l)
            else:
                for u_l, v_l in lift_adj_minus[i]:
                    adj_lift[u_l].append(v_l)

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

        # Compute spectral radius
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

    # Generate diverse initial candidates
    candidates = []

    # Pattern 1: Alt C1, Alt C2, -1 M
    s = np.ones(num_edges, dtype=int)
    for pos, idx in c1_ordered: s[idx] = 1 if pos % 2 == 0 else -1
    for pos, idx in c2_ordered: s[idx] = 1 if pos % 2 == 0 else -1
    for idx in m_edges: s[idx] = -1
    candidates.append(s.copy())

    # Pattern 2: Alt C1, Anti-Alt C2, -1 M
    s = np.ones(num_edges, dtype=int)
    for pos, idx in c1_ordered: s[idx] = 1 if pos % 2 == 0 else -1
    for pos, idx in c2_ordered: s[idx] = -1 if pos % 2 == 0 else 1
    for idx in m_edges: s[idx] = -1
    candidates.append(s.copy())

    # Pattern 3: Anti-Alt C1, Alt C2, -1 M
    s = np.ones(num_edges, dtype=int)
    for pos, idx in c1_ordered: s[idx] = -1 if pos % 2 == 0 else 1
    for pos, idx in c2_ordered: s[idx] = 1 if pos % 2 == 0 else -1
    for idx in m_edges: s[idx] = -1
    candidates.append(s.copy())

    # Pattern 4: Alt C1, Alt C2, +1 M
    s = np.ones(num_edges, dtype=int)
    for pos, idx in c1_ordered: s[idx] = 1 if pos % 2 == 0 else -1
    for pos, idx in c2_ordered: s[idx] = 1 if pos % 2 == 0 else -1
    for idx in m_edges: s[idx] = 1
    candidates.append(s.copy())

    # Pattern 5: All +1 C1, All +1 C2, Alt M
    s = np.ones(num_edges, dtype=int)
    for pos, idx in enumerate(m_ordered): s[idx] = 1 if pos % 2 == 0 else -1
    candidates.append(s.copy())

    # Pattern 6: Alt C1, Alt C2, Alt M
    s = np.ones(num_edges, dtype=int)
    for pos, idx in c1_ordered: s[idx] = 1 if pos % 2 == 0 else -1
    for pos, idx in c2_ordered: s[idx] = 1 if pos % 2 == 0 else -1
    for pos, idx in enumerate(m_ordered): s[idx] = 1 if pos % 2 == 0 else -1
    candidates.append(s.copy())

    # Pattern 7: All -1
    s = -np.ones(num_edges, dtype=int)
    candidates.append(s.copy())

    # Pattern 8: All +1
    s = np.ones(num_edges, dtype=int)
    candidates.append(s.copy())

    # Evaluate initial candidates
    for s in candidates:
        try_signs(s)

    if best_signs is None:
        # Fallback: all +1
        return np.ones(num_edges, dtype=int)

    # Phase 1: Single Flip Hill Climb
    current_signs = best_signs.copy()
    current_radius = best_radius
    max_passes = 30
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

    # Phase 2: Pairwise Flip Search (2-opt style)
    current_signs = best_signs.copy()
    current_radius = best_radius
    max_pair_passes = 10
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
                    # Restart the pairwise search from the new state
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

    # Phase 3: Triple flip search if time permits
    if time.time() < deadline - 5:
        current_signs = best_signs.copy()
        current_radius = best_radius
        for _ in range(5):
            if time.time() > deadline:
                break
            improved = False
            for i in range(num_edges):
                if time.time() > deadline:
                    break
                for j in range(i + 1, num_edges):
                    if time.time() > deadline:
                        break
                    for k in range(j + 1, num_edges):
                        if time.time() > deadline:
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

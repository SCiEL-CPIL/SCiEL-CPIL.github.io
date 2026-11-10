import time
import random
import itertools
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    def check_graph(n, adj):
        """Return list of violating 6-subsets (indices of vertices)."""
        violations = []
        for combo in itertools.combinations(range(n), 6):
            degs = [0]*6
            for i in range(6):
                for j in range(6):
                    if i != j and adj[combo[i]][combo[j]]:
                        degs[i] += 1
            if len(set(degs)) == 1:
                violations.append(combo)
        return violations

    def try_flip(n, adj, violations):
        """Try to fix violations by flipping one edge. Return (success, new_adj, new_violations)."""
        # Build edge -> affected violations map
        edge_map = {}
        for v in violations:
            for i in range(6):
                for j in range(i+1, 6):
                    u, w = v[i], v[j]
                    key = (min(u,w), max(u,w))
                    edge_map.setdefault(key, []).append(v)

        best_gain = 0
        best_edge = None
        best_new_viol = None

        for edge, viols in edge_map.items():
            u, w = edge
            # Count how many of these violations are fixed by flipping
            fixed = 0
            for v in viols:
                idx_u = v.index(u)
                idx_w = v.index(w)
                du = sum(1 for k in range(6) if k != idx_u and adj[v[idx_u]][v[k]])
                dw = sum(1 for k in range(6) if k != idx_w and adj[v[idx_w]][v[k]])
                if du != dw:
                    fixed += 1
            if fixed > best_gain:
                best_gain = fixed
                best_edge = edge
                if best_gain >= len(viols):
                    break

        if best_edge is None or best_gain == 0:
            return False, adj, violations

        u, w = best_edge
        adj[u][w] = 1 - adj[u][w]
        adj[w][u] = adj[u][w]

        # Recompute violations (only need to recheck affected ones, but for simplicity recompute all)
        new_viol = check_graph(n, adj)
        return True, adj, new_viol

    def local_search(n, adj, max_iters=50000):
        violations = check_graph(n, adj)
        iters = 0
        while violations and iters < max_iters:
            if time.time() > deadline:
                break
            # Try all violating edges, pick best
            improved = False
            # Group violations by edges
            edge_count = {}
            for v in violations:
                for i in range(6):
                    for j in range(i+1, 6):
                        u, w = v[i], v[j]
                        key = (min(u,w), max(u,w))
                        edge_count[key] = edge_count.get(key, 0) + 1

            # Try flipping each edge that appears in violations
            candidates = list(edge_count.keys())
            random.shuffle(candidates)

            best_gain = 0
            best_edge = None
            for edge in candidates:
                u, w = edge
                # Count violations fixed
                fixed = 0
                for v in violations:
                    if u in v and w in v:
                        idx_u = v.index(u)
                        idx_w = v.index(w)
                        du = sum(1 for k in range(6) if k != idx_u and adj[v[idx_u]][v[k]])
                        dw = sum(1 for k in range(6) if k != idx_w and adj[v[idx_w]][v[k]])
                        if du != dw:
                            fixed += 1
                if fixed > best_gain:
                    best_gain = fixed
                    best_edge = edge

            if best_edge is None or best_gain == 0:
                break

            u, w = best_edge
            adj[u][w] = 1 - adj[u][w]
            adj[w][u] = adj[u][w]
            violations = check_graph(n, adj)
            iters += 1

        return adj, len(violations)

    # Start with n=20
    best_n = 5
    best_edges = []

    for n in [20, 21]:
        if time.time() > deadline - 10:
            break
        found = False
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            # Random initialization
            adj = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < 0.5:
                        adj[i][j] = 1
                        adj[j][i] = 1

            adj, nv = local_search(n, adj, max_iters=20000)
            if nv == 0:
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i][j]:
                            edges.append([i, j])
                best_n = n
                best_edges = edges
                found = True
                break
            # If close, keep trying
            if nv <= 3:
                # More aggressive repair
                for extra in range(20):
                    if time.time() > deadline - 3:
                        break
                    adj, nv2 = local_search(n, adj, max_iters=5000)
                    if nv2 == 0:
                        edges = []
                        for i in range(n):
                            for j in range(i+1, n):
                                if adj[i][j]:
                                    edges.append([i, j])
                        best_n = n
                        best_edges = edges
                        found = True
                        break

        if found:
            break

    if best_n <= 5:
        # Fallback: return a small valid graph
        return {'n': 5, 'edges': []}

    return {'n': best_n, 'edges': best_edges}

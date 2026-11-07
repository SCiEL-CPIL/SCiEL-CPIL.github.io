import time
from random import Random
from collections import deque

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 140.0
    trees = instance_trees(20, 611)
    N = 20

    def edge_key(u, v):
        return (u, v) if u < v else (v, u)

    # Precompute tree structures
    tree_info = []
    for i in range(20):
        k = i + 1
        edges = trees[i]
        if k == 1:
            tree_info.append({'k': 1, 'edges': [], 'adj': [[]], 'root_order': [0], 'parent': [-2], 'max_deg': 0})
            continue
        if k == 2:
            tree_info.append({'k': 2, 'edges': [(0, 1)], 'adj': [[1], [0]], 'root_order': [0, 1], 'parent': [-2, 0], 'max_deg': 1})
            continue
        adj = [[] for _ in range(k)]
        for (u, v) in edges:
            adj[u].append(v); adj[v].append(u)
        parent = [-1] * k
        parent[0] = -2
        order = [0]
        q = deque([0])
        while q:
            x = q.popleft()
            for y in adj[x]:
                if parent[y] == -1:
                    parent[y] = x
                    q.append(y)
                    order.append(y)
        max_deg = max(len(adj[v]) for v in range(k))
        tree_info.append({'k': k, 'edges': edges, 'adj': adj, 'root_order': order, 'parent': parent, 'max_deg': max_deg})

    def try_pack(seed_val):
        rng = Random(seed_val)
        used_edges = set()
        used_hosts = [0] * N  # count of times each host is used
        result = [None] * 20

        # Process larger/more-complex trees first
        order_idx = list(range(20))
        order_idx.sort(key=lambda i: (-(tree_info[i]['max_deg']), -(i + 1)))

        for idx in order_idx:
            info = tree_info[idx]
            k = info['k']

            if k == 1:
                # Pick least-used host
                best_h = min(range(N), key=lambda h: used_hosts[h])
                result[idx] = [best_h]
                used_hosts[best_h] += 1
                continue

            if k == 2:
                # Find any unused edge, prefer least-constrained endpoints
                best_e = None
                best_score = float('inf')
                for a in range(N):
                    for b in range(a + 1, N):
                        e = (a, b)
                        if e in used_edges:
                            continue
                        score = used_hosts[a] + used_hosts[b]
                        if score < best_score:
                            best_score = score
                            best_e = e
                if best_e is None:
                    return None
                result[idx] = [best_e[0], best_e[1]]
                used_edges.add(best_e)
                used_hosts[best_e[0]] += 1
                used_hosts[best_e[1]] += 1
                continue

            # General tree: backtracking with strong heuristics
            adj = info['adj']
            parent = info['parent']
            order = info['root_order']
            edges = info['edges']

            # Build child lists for faster iteration
            children = [[] for _ in range(k)]
            for v in range(k):
                pv = parent[v]
                if pv >= 0:
                    children[pv].append(v)

            assignment = [-1] * k
            local_used = [False] * N

            # Precompute for each tree vertex: which host vertices are "compatible"
            # based on degree constraints (a host with degree d can support at most d tree-edges)

            def get_candidates(v, pv):
                """Get candidate host vertices for tree vertex v given parent placement."""
                cands = []
                for h in range(N):
                    if local_used[h]:
                        continue
                    if pv >= 0:
                        ph = assignment[pv]
                        if edge_key(h, ph) in used_edges:
                            continue
                    cands.append(h)
                return cands

            def dfs(idx_v):
                if time.time() > deadline:
                    return False
                if idx_v == k:
                    return True

                v = order[idx_v]
                pv = parent[v]
                cands = get_candidates(v, pv)

                if not cands:
                    return False

                # MRV: sort candidates by how constrained they are
                # A candidate is more constrained if it has fewer free edges to unoccupied hosts
                def constraint_score(h):
                    cnt = 0
                    for nb in range(N):
                        if nb == h or local_used[nb]:
                            continue
                        if edge_key(h, nb) in used_edges:
                            continue
                        cnt += 1
                    return cnt

                cands.sort(key=constraint_score)

                # Add randomness: shuffle the top few
                if len(cands) > 3:
                    top = cands[:3]
                    rng.shuffle(top)
                    cands = top + cands[3:]

                for h in cands:
                    assignment[v] = h
                    local_used[h] = True
                    if dfs(idx_v + 1):
                        return True
                    local_used[h] = False

                assignment[v] = -1
                return False

            if dfs(0):
                result[idx] = [assignment[i] for i in range(k)]
                for (u, v) in edges:
                    e = edge_key(assignment[u], assignment[v])
                    used_edges.add(e)
                    used_hosts[assignment[u]] += 1
                    used_hosts[assignment[v]] += 1
            else:
                return None

        # Verify
        seen = set()
        for i in range(20):
            r = result[i]
            if r is None:
                return None
            if len(set(r)) != len(r):
                return None
            for x in r:
                if not (0 <= x <= 19):
                    return None
            for (u, v) in trees[i]:
                e = edge_key(r[u], r[v])
                if e in seen:
                    return None
                seen.add(e)

        return result

    # Try many seeds with diverse orderings
    for s in range(2000):
        if time.time() > deadline:
            break
        res = try_pack(s)
        if res is not None:
            out = []
            for i in range(20):
                r = res[i]
                if r is None:
                    out.append(None)
                else:
                    out.append([int(x) for x in r])
            return out

    # Deterministic fallback
    used_edges = set()
    used_hosts = [0] * N
    result = [None] * 20
    order_idx = list(range(20))
    order_idx.sort(key=lambda i: (-(tree_info[i]['max_deg']), -(i + 1)))

    for idx in order_idx:
        info = tree_info[idx]
        k = info['k']
        if k == 1:
            best_h = min(range(N), key=lambda h: used_hosts[h])
            result[idx] = [best_h]
            used_hosts[best_h] += 1
            continue
        if k == 2:
            best_e = None
            best_score = float('inf')
            for a in range(N):
                for b in range(a + 1, N):
                    e = (a, b)
                    if e in used_edges:
                        continue
                    score = used_hosts[a] + used_hosts[b]
                    if score < best_score:
                        best_score = score
                        best_e = e
            if best_e is None:
                return [None] * 20
            result[idx] = [best_e[0], best_e[1]]
            used_edges.add(best_e)
            used_hosts[best_e[0]] += 1
            used_hosts[best_e[1]] += 1
            continue
        # Greedy BFS
        adj = info['adj']
        parent = info['parent']
        order = info['root_order']
        edges = info['edges']
        assignment = [-1] * k
        local_used = [False] * N
        success = True
        for idx_v in range(k):
            v = order[idx_v]
            pv = parent[v]
            placed = False
            for h in range(N):
                if local_used[h]:
                    continue
                if pv >= 0:
                    ph = assignment[pv]
                    if edge_key(h, ph) in used_edges:
                        continue
                assignment[v] = h
                local_used[h] = True
                placed = True
                break
            if not placed:
                success = False
                break
        if success:
            result[idx] = [assignment[i] for i in range(k)]
            for (u, v) in edges:
                e = edge_key(assignment[u], assignment[v])
                used_edges.add(e)
                used_hosts[assignment[u]] += 1
                used_hosts[assignment[v]] += 1
        else:
            return [None] * 20

    return [[int(x) for x in r] if r is not None else None for r in result]

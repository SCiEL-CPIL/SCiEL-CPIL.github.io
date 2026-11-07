import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 118.0

    n = 20
    trees = instance_trees(n, 611)

    # Precompute adjacency for each tree
    tree_adj = []
    for k in range(1, n + 1):
        edges = trees[k - 1]
        adj = [[] for _ in range(k)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)

    # Precompute parent and BFS order for each tree (rooted at 0)
    tree_parent = []
    tree_order = []
    for k in range(1, n + 1):
        if k == 1:
            tree_parent.append([-1])
            tree_order.append([0])
            continue
        adj = tree_adj[k - 1]
        parent = [-1] * k
        parent[0] = -2
        order_list = [0]
        for v in order_list:
            for nb in adj[v]:
                if parent[nb] == -1:
                    parent[nb] = v
                    order_list.append(nb)
        tree_parent.append(parent)
        tree_order.append(order_list)

    def time_left():
        return deadline - time.time()

    def try_solution(rng, time_budget):
        """Try to find a full embedding. Returns embedding list or None."""
        used_edges = set()
        embedding = [None] * n

        # Order of embedding: T20 first (spanning tree), then T19, then T2..T18, then T1
        order = [19, 18] + list(range(1, 18)) + [0]

        for idx in order:
            if time.time() > deadline:
                return None
            k = idx + 1
            tree_edges = trees[idx]
            adj = tree_adj[idx]
            parent = tree_parent[idx]
            order_list = tree_order[idx]

            if k == 1:
                # Place on the single unused vertex
                used_vertices = set()
                for e in embedding:
                    if e is not None:
                        used_vertices.update(e)
                free = [v for v in range(n) if v not in used_vertices]
                if not free:
                    return None
                embedding[idx] = [free[0]]
                continue

            # Determine free vertices
            used_vertices = set()
            for e in embedding:
                if e is not None:
                    used_vertices.update(e)
            free = [v for v in range(n) if v not in used_vertices]
            if len(free) < k:
                return None

            # Try multiple random assignments
            mapping = [-1] * k
            found = False

            max_attempts = 200 if k <= 5 else (80 if k <= 10 else 30)
            for attempt in range(max_attempts):
                if time.time() > deadline:
                    return None

                # Shuffle free vertices and assign
                shuffled_free = free[:]
                rng.shuffle(shuffled_free)

                # Assign root (tree vertex 0) to a random free vertex
                mapping = [-1] * k
                mapping[0] = shuffled_free[0]
                remaining_free = shuffled_free[1:]

                # Assign remaining tree vertices in BFS order
                success = True
                for pos in range(1, len(order_list)):
                    tv = order_list[pos]
                    p_tv = parent[tv]
                    p_gv = mapping[p_tv]

                    # Find candidates: free vertices not equal to parent, and edge to parent not used
                    cands = []
                    for gv in remaining_free:
                        if gv == p_gv:
                            continue
                        e = (min(p_gv, gv), max(p_gv, gv))
                        if e in used_edges:
                            continue
                        cands.append(gv)

                    if not cands:
                        success = False
                        break

                    rng.shuffle(cands)
                    gv = cands[0]

                    # Check against already-mapped neighbors (children assigned earlier? No, BFS order means children come after)
                    # But check against any already-mapped vertex for edge conflicts
                    ok = True
                    for nb in adj[tv]:
                        if nb == p_tv:
                            continue
                        if mapping[nb] != -1:
                            e = (min(gv, mapping[nb]), max(gv, mapping[nb]))
                            if e in used_edges:
                                ok = False
                                break
                    if not ok:
                        success = False
                        break

                    mapping[tv] = gv
                    remaining_free.remove(gv)

                if not success:
                    continue

                # Verify all edges are disjoint
                all_ok = True
                for u, v in tree_edges:
                    gu, gv = mapping[u], mapping[v]
                    if gu == -1 or gv == -1:
                        all_ok = False
                        break
                    e = (min(gu, gv), max(gu, gv))
                    if e in used_edges:
                        all_ok = False
                        break
                if not all_ok:
                    continue

                # Check injectivity
                if len(set(mapping)) != k:
                    continue

                found = True
                break

            if not found:
                return None

            # Commit
            for u, v in tree_edges:
                gu, gv = mapping[u], mapping[v]
                e = (min(gu, gv), max(gu, gv))
                used_edges.add(e)
            embedding[idx] = mapping[:]

        # Final verification
        if len(used_edges) != 190:
            return None
        for e in embedding:
            if e is not None:
                if len(e) != len(set(e)):
                    return None
                for v in e:
                    if not isinstance(v, int) or isinstance(v, bool) or v < 0 or v >= n:
                        return None
        return embedding

    best_result = None
    best_edges = 0

    restart = 0
    while time.time() < deadline - 2.0:
        restart += 1
        rng = Random(611 + restart * 7919)
        try:
            result = try_solution(rng, None)
            if result is not None:
                # Count edges
                used_edges = set()
                for idx in range(n):
                    e = result[idx]
                    if e is not None:
                        for u, v in trees[idx]:
                            used_edges.add((min(e[u], e[v]), max(e[u], e[v])))
                if len(used_edges) == 190:
                    best_result = [list(e) if e is not None else None for e in result]
                    best_edges = 190
                    break
        except Exception:
            pass

    if best_result is not None:
        return best_result

    # Fallback: return best partial or empty
    return [None] * 20


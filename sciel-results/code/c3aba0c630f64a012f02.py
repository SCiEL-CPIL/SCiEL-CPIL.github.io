import time
from random import Random

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

def _all_edges(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

def _edge_set(edges):
    return set((min(u, v), max(u, v)) for (u, v) in edges)

def construct():
    n = 20
    deadline = time.time() + 145.0
    trees = instance_trees(n, 611)

    all_edges = _all_edges(n)
    all_edge_set = set(all_edges)

    # Precompute tree degrees
    tree_deg_list = []
    for k in range(n):
        deg = [0] * k
        for (u, v) in trees[k]:
            deg[u] += 1
            deg[v] += 1
        tree_deg_list.append(deg)

    best_solution = None
    best_count = -1

    def solve_once(seed_val):
        rng = Random(seed_val)
        remaining_edges = set(all_edges)
        t20_edges = _edge_set(trees[19])
        for e in t20_edges:
            remaining_edges.discard(e)

        assignments = [None] * n
        assignments[19] = list(range(20))

        to_embed = list(range(18, -1, -1))
        # Shuffle for diversification
        rng.shuffle(to_embed)

        def embed_tree(tree_k):
            kk = tree_k
            tree_edges = trees[tree_k]
            if not tree_edges:
                assignments[tree_k] = [0]
                return True

            tree_deg = tree_deg_list[tree_k]
            verts_ordered = sorted(range(kk), key=lambda v: (-tree_deg[v], v))

            host_map = [-1] * kk
            used_hosts = set()
            used_host_edges = set()

            def bt(pos):
                if time.time() > deadline:
                    return False
                if pos == len(verts_ordered):
                    for (u, v) in tree_edges:
                        he = (min(host_map[u], host_map[v]), max(host_map[u], host_map[v]))
                        if he not in remaining_edges or he in used_host_edges:
                            return False
                    assignments[tree_k] = list(host_map)
                    return True

                tv = verts_ordered[pos]
                for hv in range(n):
                    if hv in used_hosts:
                        continue
                    ok = True
                    for (u, v) in tree_edges:
                        if u == tv:
                            other = v
                        elif v == tv:
                            other = u
                        else:
                            continue
                        if host_map[other] != -1:
                            he = (min(hv, host_map[other]), max(hv, host_map[other]))
                            if he not in remaining_edges or he in used_host_edges:
                                ok = False
                                break
                    if not ok:
                        continue

                    host_map[tv] = hv
                    used_hosts.add(hv)
                    added = []
                    for (u, v) in tree_edges:
                        if u == tv or v == tv:
                            other = v if u == tv else u
                            if host_map[other] != -1:
                                he = (min(hv, host_map[other]), max(hv, host_map[other]))
                                used_host_edges.add(he)
                                added.append(he)

                    if bt(pos + 1):
                        return True

                    for he in added:
                        used_host_edges.discard(he)
                    used_hosts.discard(hv)
                    host_map[tv] = -1

                return False

            return bt(0)

        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == len(to_embed):
                return True
            k = to_embed[idx]
            if embed_tree(k):
                for (u, v) in trees[k]:
                    hv_u = assignments[k][u]
                    hv_v = assignments[k][v]
                    he = (min(hv_u, hv_v), max(hv_u, hv_v))
                    remaining_edges.discard(he)
                if backtrack(idx + 1):
                    return True
                for (u, v) in trees[k]:
                    hv_u = assignments[k][u]
                    hv_v = assignments[k][v]
                    he = (min(hv_u, hv_v), max(hv_u, hv_v))
                    remaining_edges.add(he)
                assignments[k] = None
                return False
            else:
                assignments[k] = None
                return False

        return backtrack(0)

    for attempt in range(100):
        if time.time() > deadline:
            break
        if solve_once(611 + attempt):
            result = [None] * 20
            total_edges = 0
            for k in range(20):
                if assignments[k] is not None:
                    result[k] = [int(x) for x in assignments[k]]
                    total_edges += len(trees[k])
            if total_edges > best_count:
                best_count = total_edges
                best_solution = result
            if best_count == 190:
                break

    if best_solution is not None:
        return best_solution
    return [None] * 19 + [list(range(20))]

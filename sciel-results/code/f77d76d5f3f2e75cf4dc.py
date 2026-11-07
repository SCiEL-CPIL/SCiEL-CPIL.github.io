import time
import random
import sys

sys.setrecursionlimit(100000)

def instance_trees(n=20, seed=611):
    rng = random.Random(seed)
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
    deadline = start_time + 145

    trees = instance_trees(20, 611)
    n_trees = 20
    n_vertices = 20

    tree_edges = []
    tree_adj = []
    for i in range(n_trees):
        edges = trees[i]
        k = i + 1
        adj_list = [[] for _ in range(k)]
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        tree_adj.append(adj_list)
        tree_edges.append(edges)

    best_score = 0
    best_mappings = [None] * n_trees

    def edge_key(u, v):
        return (min(u, v), max(u, v))

    def pack_tree_bt(t_idx, used_edges, local_rng, max_nodes=5000):
        """Backtracking pack with node limit."""
        k = t_idx + 1
        adj_t = tree_adj[t_idx]
        edges_t = tree_edges[t_idx]

        if k == 1:
            for hv in range(n_vertices):
                return [hv]
            return None

        root = max(range(k), key=lambda v: len(adj_t[v]))
        order = [root]
        visited = {root}
        i = 0
        while i < len(order):
            for nb in adj_t[order[i]]:
                if nb not in visited:
                    visited.add(nb)
                    order.append(nb)
            i += 1

        mapping = [-1] * k
        used_verts = set()
        node_count = [0]

        def bt(pos):
            if node_count[0] > max_nodes or time.time() > deadline:
                return False
            if pos == k:
                for u, v in edges_t:
                    e = edge_key(mapping[u], mapping[v])
                    if e in used_edges:
                        return False
                return True

            tv = order[pos]
            cands = []
            for hv in range(n_vertices):
                if hv in used_verts:
                    continue
                ok = True
                for nb in adj_t[tv]:
                    if mapping[nb] != -1:
                        e = edge_key(hv, mapping[nb])
                        if e in used_edges:
                            ok = False
                            break
                if ok:
                    cands.append(hv)

            if not cands:
                return False

            local_rng.shuffle(cands)

            for hv in cands:
                node_count[0] += 1
                mapping[tv] = hv
                used_verts.add(hv)
                if bt(pos + 1):
                    return True
                used_verts.remove(hv)
                mapping[tv] = -1

            return False

        host_order = list(range(n_vertices))
        local_rng.shuffle(host_order)
        for h0 in host_order:
            mapping[order[0]] = h0
            used_verts = {h0}
            node_count[0] = 0
            if bt(1):
                return mapping[:]
            used_verts.clear()
            mapping[order[0]] = -1
        return None

    def try_pack(seed_val):
        local_rng = random.Random(seed_val)
        used_edges = set()
        mappings = [None] * n_trees
        tree_order = list(range(n_trees))
        tree_order.sort(key=lambda x: -(x + 1))

        for t_idx in tree_order:
            if time.time() > deadline:
                break
            m = pack_tree_bt(t_idx, used_edges, local_rng, max_nodes=2000)
            if m is not None:
                for u, v in tree_edges[t_idx]:
                    used_edges.add(edge_key(m[u], m[v]))
                mappings[t_idx] = m

        score = sum(len(tree_edges[i]) for i in range(n_trees) if mappings[i] is not None)
        return mappings, score

    attempt = 0
    while time.time() < deadline:
        mappings, score = try_pack(attempt)
        if score > best_score:
            best_score = score
            best_mappings = [mappings[i][:] if mappings[i] is not None else None for i in range(n_trees)]
            if best_score == 190:
                break
        attempt += 1

    return best_mappings

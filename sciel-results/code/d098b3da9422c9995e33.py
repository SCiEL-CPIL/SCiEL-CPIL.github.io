import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
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
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    deadline = time.time() + 145.0
    trees = instance_trees(20, 611)
    n = 20

    # Build all edges of K20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))

    # Precompute tree info
    tree_info = []
    for t_idx in range(20):
        tree_edges = trees[t_idx]
        k = t_idx + 1
        if k <= 2:
            tree_info.append((tree_edges, None, None, k))
            continue
        tree_adj = {v: [] for v in range(k)}
        for u, v in tree_edges:
            tree_adj[u].append(v)
            tree_adj[v].append(u)
        # Order by degree descending, then by index for determinism
        ordered_verts = sorted(range(k), key=lambda v: (-len(tree_adj[v]), v))
        tree_info.append((tree_edges, tree_adj, ordered_verts, k))

    def try_embed(t_idx, available, rng):
        """Try to embed tree t_idx into available edges using greedy + backtracking."""
        tree_edges, tree_adj, ordered_verts, k = tree_info[t_idx]

        if k == 1:
            return [rng.randint(0, n - 1)]
        if k == 2:
            for edge in available:
                return [edge[0], edge[1]]
            return None

        # Build adjacency of available graph for degree tracking
        # host_degree[h] = number of available edges incident to h
        host_degree = [0] * n
        for h in range(n):
            for nb in range(n):
                if h == nb:
                    continue
                edge = (min(h, nb), max(h, nb))
                if edge in available:
                    host_degree[h] += 1

        host_map = {}
        used_hosts = set()

        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == k:
                # Verify all edges
                for u, v in tree_edges:
                    hu, hv = host_map[u], host_map[v]
                    edge = (min(hu, hv), max(hu, hv))
                    if edge not in available:
                        return False
                return True

            tv = ordered_verts[pos]
            # Find candidates: unused host vertices connected to all already-assigned neighbors
            candidates = []
            for h in range(n):
                if h in used_hosts:
                    continue
                ok = True
                for nb in tree_adj[tv]:
                    if nb in host_map:
                        hn = host_map[nb]
                        edge = (min(h, hn), max(h, hn))
                        if edge not in available:
                            ok = False
                            break
                if ok:
                    candidates.append(h)

            if not candidates:
                return False

            # Sort by current host degree (prefer higher degree)
            candidates.sort(key=lambda h: -host_degree[h])
            # Add some randomness for diversity across restarts
            if rng.random() < 0.2:
                rng.shuffle(candidates)

            for h in candidates:
                host_map[tv] = h
                used_hosts.add(h)

                # Update host_degree: remove edges from h to all unused hosts
                for nb_h in range(n):
                    if nb_h == h or nb_h in used_hosts:
                        continue
                    edge = (min(h, nb_h), max(h, nb_h))
                    if edge in available:
                        host_degree[nb_h] -= 1

                if backtrack(pos + 1):
                    return True

                # Undo degree updates
                for nb_h in range(n):
                    if nb_h == h or nb_h in used_hosts:
                        continue
                    edge = (min(h, nb_h), max(h, nb_h))
                    if edge in available:
                        host_degree[nb_h] += 1

                del host_map[tv]
                used_hosts.remove(h)

            return False

        if backtrack(0):
            return [host_map[i] for i in range(k)]
        return None

    # Process trees from largest to smallest
    tree_order = list(range(20))
    tree_order.sort(key=lambda idx: -(idx + 1))

    best_result = None
    best_edges_used = -1

    for attempt in range(200):
        if time.time() > deadline - 2:
            break

        rng = Random(attempt * 31 + 7)
        avail = set(all_edges)
        res = [None] * 20
        success = True

        for t_idx in tree_order:
            if time.time() > deadline - 2:
                success = False
                break
            k = tree_info[t_idx][3]

            if k == 1:
                res[t_idx] = [rng.randint(0, n - 1)]
                continue

            embedded = None
            for trial in range(12):
                if time.time() > deadline - 2:
                    success = False
                    break
                embedded = try_embed(t_idx, avail, rng)
                if embedded is not None:
                    break

            if not success:
                break
            if embedded is None:
                success = False
                break

            res[t_idx] = embedded
            if k >= 2:
                for u, v in trees[t_idx]:
                    hu, hv = embedded[u], embedded[v]
                    edge = (min(hu, hv), max(hu, hv))
                    avail.discard(edge)

        if success:
            edges_used = 190 - len(avail)
            if edges_used > best_edges_used:
                best_edges_used = edges_used
                best_result = res
                if edges_used == 190:
                    break

    if best_result is None:
        # Fallback: try a simpler approach
        return [None] * 19 + [list(range(20))]

    final = []
    for entry in best_result:
        if entry is None:
            final.append(None)
        else:
            final.append([int(x) for x in entry])
    return final

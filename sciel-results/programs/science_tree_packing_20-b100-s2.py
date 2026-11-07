import time
import random
import sys

sys.setrecursionlimit(100000)

# Fixed instance generation
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
    deadline = start_time + 145  # 5s margin

    trees = instance_trees(20, 611)
    n_trees = 20
    n_vertices = 20

    # Precompute tree adjacencies and edge lists
    tree_adj = []
    tree_edges = []
    for i in range(n_trees):
        edges = trees[i]
        k = i + 1
        adj_list = [[] for _ in range(k)]
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        tree_adj.append(adj_list)
        tree_edges.append(edges)

    # Sort trees by size descending (number of vertices)
    tree_indices = list(range(n_trees))
    tree_indices.sort(key=lambda x: -(x + 1))

    best_score = 0
    best_mappings = [None] * n_trees

    # We will try multiple random restarts
    rng = random.Random(12345)

    def try_packing(attempt_seed):
        nonlocal best_score, best_mappings
        local_rng = random.Random(attempt_seed)
        used_edges = set()
        mappings = [None] * n_trees

        # Precompute: for each tree, a topological-like ordering for backtracking
        # We'll use a heuristic: process leaves first? Or high-degree first?
        # Let's try a simple BFS/DFS order from root.
        # Actually, for a tree, we can use any order. Let's use a constrained order:
        # Start with a high-degree vertex, then its neighbors, etc.

        def pack_tree(tree_order_idx):
            t_idx = tree_indices[tree_order_idx]
            k = t_idx + 1
            adj_t = tree_adj[t_idx]

            if k == 1:
                mappings[t_idx] = [0]
                return True

            # Determine a good vertex ordering for this tree
            # Use a greedy ordering: start with a vertex of max degree,
            # then repeatedly add a neighbor of the current set.
            # This gives a "rooted" BFS-like order.

            # Find root: vertex with max degree
            root = max(range(k), key=lambda v: len(adj_t[v]))

            # BFS ordering
            order = [root]
            visited = {root}
            i = 0
            while i < len(order):
                for nb in adj_t[order[i]]:
                    if nb not in visited:
                        visited.add(nb)
                        order.append(nb)
                i += 1

            # Reverse: we want to assign leaves first? No, typically we assign
            # the root first, then children. But for constraint propagation,
            # it might be better to assign constrained vertices first.
            # Let's try assigning in BFS order (root first).

            # For backtracking, we need to map tree vertices (in order) to host vertices.
            # mapping[tv] = hv
            # We'll use a recursive backtracking that assigns one vertex at a time.

            def backtrack(pos, mapping, used_verts):
                if pos == k:
                    # All vertices mapped. Check all edges are valid.
                    for u, v in tree_edges[t_idx]:
                        hu = mapping[u]
                        hv = mapping[v]
                        e = (min(hu, hv), max(hu, hv))
                        if e in used_edges:
                            return False
                    # Commit
                    for u, v in tree_edges[t_idx]:
                        hu = mapping[u]
                        hv = mapping[v]
                        used_edges.add((min(hu, hv), max(hu, hv)))
                    mappings[t_idx] = mapping[:]
                    return True

                tv = order[pos]
                # Try all available host vertices
                # Heuristic: try vertices that have fewer conflicts with already-mapped neighbors
                candidates = []
                for hv in range(n_vertices):
                    if hv in used_verts:
                        continue
                    # Check edges to already-mapped neighbors
                    conflict = False
                    for nb in adj_t[tv]:
                        if nb in used_verts_set:
                            hn = mapping[nb]
                            e = (min(hv, hn), max(hv, hn))
                            if e in used_edges:
                                conflict = True
                                break
                    if not conflict:
                        candidates.append(hv)

                if not candidates:
                    return False

                # Shuffle for diversity
                local_rng.shuffle(candidates)

                for hv in candidates:
                    mapping[tv] = hv
                    used_verts.add(hv)
                    if backtrack(pos + 1, mapping, used_verts):
                        return True
                    used_verts.remove(hv)
                    mapping[tv] = -1

                return False

            # We need a helper to check if a vertex is mapped
            # Let's use a dict or a list with -1 for unmapped.
            # Actually, let's restructure: mapping is a list of length k, initialized to -1.
            # used_verts is a set.

            # Start with the root
            # Try all possible host vertices for the root
            host_candidates = list(range(n_vertices))
            local_rng.shuffle(host_candidates)

            for h0 in host_candidates:
                mapping = [-1] * k
                mapping[order[0]] = h0
                used_verts = {h0}
                used_verts_set = {order[0]}  # Set of tree vertices already mapped

                # We need to track which tree vertices are mapped
                # Let's modify backtrack to use a separate set for mapped tree vertices.

                def backtrack2(pos, mapping, used_verts, mapped_tv):
                    if pos == k:
                        # Check all edges
                        for u, v in tree_edges[t_idx]:
                            hu = mapping[u]
                            hv = mapping[v]
                            e = (min(hu, hv), max(hu, hv))
                            if e in used_edges:
                                return False
                        # Commit
                        for u, v in tree_edges[t_idx]:
                            hu = mapping[u]
                            hv = mapping[v]
                            used_edges.add((min(hu, hv), max(hu, hv)))
                        mappings[t_idx] = mapping[:]
                        return True

                    tv = order[pos]
                    candidates = []
                    for hv in range(n_vertices):
                        if hv in used_verts:
                            continue
                        conflict = False
                        for nb in adj_t[tv]:
                            if nb in mapped_tv:
                                hn = mapping[nb]
                                e = (min(hv, hn), max(hv, hn))
                                if e in used_edges:
                                    conflict = True
                                    break
                        if not conflict:
                            candidates.append(hv)

                    if not candidates:
                        return False

                    local_rng.shuffle(candidates)

                    for hv in candidates:
                        mapping[tv] = hv
                        used_verts.add(hv)
                        mapped_tv.add(tv)
                        if backtrack2(pos + 1, mapping, used_verts, mapped_tv):
                            return True
                        mapped_tv.remove(tv)
                        used_verts.remove(hv)
                        mapping[tv] = -1

                    return False

                if backtrack2(1, mapping, used_verts, used_verts_set):
                    return True

            return False

        # Main loop
        success = True
        for i in range(n_trees):
            if time.time() > deadline:
                success = False
                break
            if pack_tree(i):
                pass
            else:
                success = False
                break

        # Calculate score
        score = 0
        for i in range(n_trees):
            if mappings[i] is not None:
                score += len(tree_edges[i])

        if score > best_score:
            best_score = score
            best_mappings = [mappings[i][:] if mappings[i] is not None else None for i in range(n_trees)]

        return success

    # Run multiple attempts
    attempt = 0
    while time.time() < deadline:
        try_packing(attempt)
        attempt += 1
        if best_score == 190:
            break

    return best_mappings

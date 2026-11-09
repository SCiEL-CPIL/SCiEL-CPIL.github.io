import random
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill

    # Generate the trees
    from random import Random
    def instance_trees(n=20, seed=611):
        rng = Random(seed); trees = []
        for k in range(1, n + 1):
            if k <= 2:
                trees.append([] if k == 1 else [(0, 1)]); continue
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

    trees = instance_trees(20, 611)
    n = 20

    # Build adjacency for each tree
    tree_adj = []
    for t in trees:
        adj = defaultdict(list)
        for (u, v) in t:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)

    # Order trees by size descending (harder first)
    order = sorted(range(n), key=lambda i: len(trees[i]), reverse=True)

    # We'll assign embeddings in this order
    # embeddings[i] = list of n host vertices for tree i, or None
    # used_edges = set of (min(u,v), max(u,v)) tuples for host K20 edges used

    best_result = [None] * n
    best_count = 0

    def count_embedded(result):
        total = 0
        for i in range(n):
            if result[i] is not None:
                total += len(trees[i])
        return total

    # Try to embed all trees using randomized backtracking with restarts
    def solve_with_restarts(max_restarts=50000):
        nonlocal best_result, best_count

        rng = random.Random(42)

        for restart in range(max_restarts):
            if time.time() > deadline:
                break

            # Shuffle the order slightly for diversity
            cur_order = order[:]
            if restart > 0:
                # Partial shuffle: keep largest few in place, shuffle rest
                shuffle_start = min(5, len(cur_order))
                cur_order[shuffle_start:] = cur_order[shuffle_start:]
                rng.shuffle(cur_order[shuffle_start:])

            embeddings = [None] * n
            used_edges = set()
            # Track which host vertices are used by which tree (for vertex constraints)
            # Actually, trees can share vertices, just not edges. So no vertex constraint.

            success = True

            for idx in cur_order:
                k = idx  # tree index
                if k == 0:
                    embeddings[k] = None  # T1 has no edges
                    continue
                if k == 1:
                    # T2: single edge (0,1)
                    # Find a free edge in K20
                    found = False
                    for u in range(n):
                        if found:
                            break
                        for v in range(u + 1, n):
                            if (u, v) not in used_edges:
                                embeddings[k] = [0] * n
                                embeddings[k][0] = u
                                embeddings[k][1] = v
                                used_edges.add((u, v))
                                found = True
                                break
                    if not found:
                        success = False
                        break
                    continue

                # For larger trees, use backtracking
                # Build the tree's adjacency
                adj = tree_adj[k]
                num_nodes = k + 1  # tree has k+1 nodes (0..k)

                # Pick a root
                root = 0
                # BFS order from root
                parent = [-1] * num_nodes
                order_list = [root]
                visited = [False] * num_nodes
                visited[root] = True
                qi = 0
                while qi < len(order_list):
                    u = order_list[qi]
                    qi += 1
                    for v in adj[u]:
                        if not visited[v]:
                            visited[v] = True
                            parent[v] = u
                            order_list.append(v)

                # Now try to assign host vertices in BFS order
                # assignment: tree_node -> host_vertex
                assignment = [-1] * num_nodes
                assignment[root] = -1  # to be assigned

                # Backtrack
                def backtrack(pos):
                    if time.time() > deadline:
                        return False
                    if pos == len(order_list):
                        # Check all edges are valid (they are by construction) and not used
                        for node in range(num_nodes):
                            for nb in adj[node]:
                                if node < nb:
                                    hu = assignment[node]
                                    hv = assignment[nb]
                                    edge = (min(hu, hv), max(hu, hv))
                                    if edge in used_edges:
                                        return False
                        # All good, commit
                        for node in range(num_nodes):
                            embeddings[k][node] = assignment[node]
                        for node in range(num_nodes):
                            for nb in adj[node]:
                                if node < nb:
                                    hu = assignment[node]
                                    hv = assignment[nb]
                                    edge = (min(hu, hv), max(hu, hv))
                                    used_edges.add(edge)
                        return True

                    node = order_list[pos]
                    # Try all host vertices
                    candidates = list(range(n))
                    rng.shuffle(candidates)

                    for hv in candidates:
                        assignment[node] = hv
                        # Check: if node has a parent, the edge (parent, node) must not be used
                        if parent[node] != -1:
                            hu = assignment[parent[node]]
                            edge = (min(hu, hv), max(hu, hv))
                            if edge in used_edges:
                                assignment[node] = -1
                                continue
                        # Check: for any already-assigned neighbor that is not parent
                        # Actually in BFS, only parent is assigned before. Children come after.
                        # But wait, in BFS tree, a node's neighbors that are already assigned
                        # are just its parent (since we go in BFS order, all other neighbors
                        # are either children (not yet assigned) or... actually in a tree,
                        # in BFS order, a node can have neighbors that are at the same level
                        # or previous levels. Let me reconsider.
                        # In a tree, BFS from root: each node (except root) has exactly one
                        # parent. Other neighbors are children. So in BFS order, when we assign
                        # a node, only its parent is already assigned. So we only need to check
                        # the parent edge.
                        # But wait - what about non-tree edges? There are none in a tree.
                        # So we're fine.

                        if backtrack(pos + 1):
                            return True

                    assignment[node] = -1
                    return False

                embeddings[k] = [-1] * num_nodes
                if not backtrack(0):
                    success = False
                    break

            if success:
                # Verify all embeddings
                for i in range(n):
                    if i == 0:
                        best_result[i] = None
                    else:
                        best_result[i] = list(embeddings[i])
                best_count = count_embedded(best_result)
                return True

        return False

    solve_with_restarts(max_restarts=100000)

    # If we didn't get a full packing, try to improve by adding more trees
    # Actually, the above tries to embed ALL trees. If it fails, best_result may be incomplete.
    # Let's also try a greedy approach as fallback

    if best_count < 190:
        # Try greedy: embed trees one by one, largest first, using random search per tree
        rng = random.Random(123)
        greedy_result = [None] * n
        used_edges = set()

        for idx in order:
            if time.time() > deadline:
                break
            k = idx
            if k == 0:
                greedy_result[k] = None
                continue
            if k == 1:
                found = False
                for u in range(n):
                    if found:
                        break
                    for v in range(u + 1, n):
                        if (u, v) not in used_edges:
                            greedy_result[k] = [0] * n
                            greedy_result[k][0] = u
                            greedy_result[k][1] = v
                            used_edges.add((u, v))
                            found = True
                            break
                if not found:
                    continue
                continue

            num_nodes = k + 1
            adj = tree_adj[k]
            root = 0
            parent = [-1] * num_nodes
            bfs_order = [root]
            visited = [False] * num_nodes
            visited[root] = True
            qi = 0
            while qi < len(bfs_order):
                u = bfs_order[qi]
                qi += 1
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        parent[v] = u
                        bfs_order.append(v)

            # Try random assignments with limited backtracking
            found = False
            for attempt in range(10000):
                if time.time() > deadline:
                    break
                assignment = [-1] * num_nodes
                ok = True
                for pos in range(len(bfs_order)):
                    node = bfs_order[pos]
                    candidates = list(range(n))
                    rng.shuffle(candidates)
                    placed = False
                    for hv in candidates:
                        assignment[node] = hv
                        if parent[node] != -1:
                            hu = assignment[parent[node]]
                            edge = (min(hu, hv), max(hu, hv))
                            if edge in used_edges:
                                continue
                        placed = True
                        break
                    if not placed:
                        ok = False
                        break
                if ok:
                    # Verify all edges
                    valid = True
                    for node in range(num_nodes):
                        for nb in adj[node]:
                            if node < nb:
                                hu = assignment[node]
                                hv = assignment[nb]
                                edge = (min(hu, hv), max(hu, hv))
                                if edge in used_edges:
                                    valid = False
                                    break
                        if not valid:
                            break
                    if valid:
                        greedy_result[k] = list(assignment)
                        for node in range(num_nodes):
                            for nb in adj[node]:
                                if node < nb:
                                    hu = assignment[node]
                                    hv = assignment[nb]
                                    edge = (min(hu, hv), max(hu, hv))
                                    used_edges.add(edge)
                        found = True
                        break

        greedy_count = count_embedded(greedy_result)
        if greedy_count > best_count:
            best_result = greedy_result
            best_count = greedy_count

    # Ensure correct types
    final = []
    for i in range(n):
        if best_result[i] is None:
            final.append(None)
        else:
            final.append([int(x) for x in best_result[i]])

    return final

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
    deadline = start_time + 115

    trees = instance_trees(20, 611)
    N = 20

    # Precompute tree properties
    tree_edges = []
    tree_degrees = []
    tree_adj = []
    for k_idx in range(20):
        k = k_idx + 1
        es = set()
        deg = [0] * k
        adj = [[] for _ in range(k)]
        for u, v in trees[k_idx]:
            es.add((min(u, v), max(u, v)))
            deg[u] += 1
            deg[v] += 1
            adj[u].append(v)
            adj[v].append(u)
        tree_edges.append(es)
        tree_degrees.append(deg)
        tree_adj.append(adj)

    # All host edges
    all_host_edges = set()
    for i in range(N):
        for j in range(i + 1, N):
            all_host_edges.add((i, j))

    def time_up():
        return time.time() > deadline

    def get_used_edges(mapping, tree_idx):
        edges = set()
        for u, v in trees[tree_idx]:
            hu, hv = mapping[u], mapping[v]
            edges.add((min(hu, hv), max(hu, hv)))
        return edges

    def verify_packing(mappings):
        """Verify that a packing is valid: distinct vertices, edge-disjoint, correct tree structure."""
        used_edges = set()
        for i in range(20):
            m = mappings[i]
            if m is None:
                continue
            k = i + 1
            if len(m) != k:
                return False
            # Check distinctness
            if len(set(m)) != k:
                return False
            # Check all values in range
            for v in m:
                if not isinstance(v, int) or v < 0 or v >= N:
                    return False
            # Check edges
            for u, v in trees[i]:
                hu, hv = m[u], m[v]
                e = (min(hu, hv), max(hu, hv))
                if e in used_edges:
                    return False
                used_edges.add(e)
        return True

    def embed_tree_bipartite(tree_idx, available_edges, rng, max_attempts=5):
        """
        Embed tree using a bipartite matching-inspired greedy approach.
        Left nodes: tree vertices (ordered by degree descending).
        Right nodes: host vertices.
        Edge exists if host has enough remaining degree and edges to assigned neighbors exist.
        """
        k = tree_idx + 1
        if k == 1:
            return [0]
        if k == 2:
            for e in available_edges:
                return [e[0], e[1]]
            return None

        deg = tree_degrees[tree_idx]
        adj = tree_adj[tree_idx]

        # Order by degree descending
        order = sorted(range(k), key=lambda x: (-deg[x], x))
        pos_in_order = [0] * k
        for i, v in enumerate(order):
            pos_in_order[v] = i

        for attempt in range(max_attempts):
            assigned = [-1] * k
            used_hosts = set()
            used_edges_local = set()

            # Track remaining degree per host
            rem_deg = [0] * N
            for e in available_edges:
                rem_deg[e[0]] += 1
                rem_deg[e[1]] += 1

            success = True
            for idx in range(k):
                if time_up():
                    return None

                v = order[idx]
                v_deg = deg[v]

                # Get assigned neighbors
                assigned_neighbors = []
                for u in adj[v]:
                    if pos_in_order[u] < idx:
                        assigned_neighbors.append((u, assigned[pos_in_order[u]]))

                # Find candidate hosts
                candidates = []
                for h in range(N):
                    if h in used_hosts:
                        continue
                    # Check edges to assigned neighbors
                    valid = True
                    for u, hu in assigned_neighbors:
                        edge = (min(h, hu), max(h, hu))
                        if edge not in available_edges or edge in used_edges_local:
                            valid = False
                            break
                    if not valid:
                        continue
                    # Degree constraint
                    if rem_deg[h] < v_deg:
                        continue
                    candidates.append(h)

                if not candidates:
                    success = False
                    break

                # Score candidates: prefer those with fewer options (MRV)
                # and those that minimize conflict
                scored = []
                for h in candidates:
                    # Count how many unassigned tree neighbors could still connect to h
                    # (this is a soft constraint)
                    conflict_score = 0
                    for u in adj[v]:
                        if pos_in_order[u] > idx:
                            # u is unassigned; check if h has available edge capacity
                            pass  # Can't check yet
                    scored.append((rem_deg[h], h))

                scored.sort(key=lambda x: (x[0], x[1]))
                rng.shuffle(scored)

                # Try top candidates
                found = False
                for _, h in scored:
                    if time_up():
                        return None
                    assigned[pos_in_order[v]] = h
                    used_hosts.add(h)
                    rem_deg[h] -= 1

                    # Add edges to used_edges_local
                    for u, hu in assigned_neighbors:
                        edge = (min(h, hu), max(h, hu))
                        used_edges_local.add(edge)

                    found = True
                    break

                if not found:
                    success = False
                    break

            if success:
                # Verify the embedding
                mapping = [0] * k
                for i, v in enumerate(order):
                    mapping[v] = assigned[i]

                # Double-check: all edges must be in available_edges and distinct
                check_edges = set()
                valid = True
                for u, v in trees[tree_idx]:
                    hu, hv = mapping[u], mapping[v]
                    e = (min(hu, hv), max(hu, hv))
                    if e in check_edges or e not in available_edges:
                        valid = False
                        break
                    check_edges.add(e)

                if valid:
                    return mapping

        return None

    def try_packing(order, rng_master, max_attempts_per_tree=3):
        mappings = [None] * 20
        used_edges = set()
        available = set(all_host_edges)
        value = 0

        for tree_idx in order:
            if time_up():
                break
            k = tree_idx + 1
            if k == 1:
                mappings[tree_idx] = [0]
                continue

            found = False
            for attempt in range(max_attempts_per_tree):
                if time_up():
                    break
                rng = Random(rng_master * 10000 + tree_idx * 100 + attempt)
                m = embed_tree_bipartite(tree_idx, available, rng, max_attempts=3)
                if m is not None:
                    mappings[tree_idx] = m
                    used_edges |= get_used_edges(m, tree_idx)
                    available = all_host_edges - used_edges
                    value += len(tree_edges[tree_idx])
                    found = True
                    break

            if not found:
                mappings[tree_idx] = None

        return mappings, value

    # Multiple strategies
    order_desc = list(range(19, -1, -1))
    order_asc = list(range(20))
    max_deg = [max(tree_degrees[i]) if tree_degrees[i] else 0 for i in range(20)]
    order_deg = sorted(range(20), key=lambda i: -max_deg[i])

    best_mappings = [None] * 20
    best_value = 0

    strategies = [
        (order_desc, 1), (order_desc, 42), (order_desc, 123), (order_desc, 777),
        (order_asc, 1), (order_asc, 42),
        (order_deg, 1), (order_deg, 42), (order_deg, 123),
    ]

    for order, rng_master in strategies:
        if time_up():
            break
        mappings, value = try_packing(order, rng_master)
        if value > best_value:
            best_value = value
            best_mappings = mappings[:]

    # Try to fill missing trees
    if best_value < 190 and not time_up():
        missing = [i for i in range(20) if best_mappings[i] is None]
        if missing:
            used_edges = set()
            for i in range(20):
                if best_mappings[i] is not None:
                    used_edges |= get_used_edges(best_mappings[i], i)
            available = all_host_edges - used_edges

            missing_sorted = sorted(missing, key=lambda i: -(i + 1))
            for tree_idx in missing_sorted:
                if time_up():
                    break
                k = tree_idx + 1
                if k == 1:
                    best_mappings[tree_idx] = [0]
                    continue
                for attempt in range(15):
                    if time_up():
                        break
                    rng = Random(99999 + tree_idx * 1000 + attempt)
                    m = embed_tree_bipartite(tree_idx, available, rng, max_attempts=5)
                    if m is not None:
                        best_mappings[tree_idx] = m
                        used_edges |= get_used_edges(m, tree_idx)
                        available = all_host_edges - used_edges
                        best_value += len(tree_edges[tree_idx])
                        break

    # Verify and fallback if needed
    if not verify_packing(best_mappings):
        # Rebuild from scratch with a simple greedy
        best_mappings = [None] * 20
        best_value = 0
        used_edges = set()
        available = set(all_host_edges)
        for tree_idx in range(19, -1, -1):
            if time_up():
                break
            k = tree_idx + 1
            if k == 1:
                best_mappings[tree_idx] = [0]
                continue
            for attempt in range(10):
                if time_up():
                    break
                rng = Random(50000 + tree_idx * 100 + attempt)
                m = embed_tree_bipartite(tree_idx, available, rng, max_attempts=3)
                if m is not None:
                    best_mappings[tree_idx] = m
                    used_edges |= get_used_edges(m, tree_idx)
                    available = all_host_edges - used_edges
                    best_value += len(tree_edges[tree_idx])
                    break

    # Final check
    if not verify_packing(best_mappings):
        # Absolute fallback: return all None
        best_mappings = [None] * 20
        best_value = 0

    result = []
    for i in range(20):
        if best_mappings[i] is None:
            result.append(None)
        else:
            result.append([int(x) for x in best_mappings[i]])

    return result

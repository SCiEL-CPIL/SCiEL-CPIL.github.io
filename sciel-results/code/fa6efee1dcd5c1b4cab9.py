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
    deadline = start_time + 115  # 115s limit to be safe under 150s

    trees = instance_trees(20, 611)

    # Precompute tree edge sets (undirected)
    tree_edges = []
    for t in trees:
        e = set()
        for u, v in t:
            e.add((min(u, v), max(u, v)))
        tree_edges.append(e)

    # Host graph K20
    N = 20
    all_host_edges = set()
    for i in range(N):
        for j in range(i + 1, N):
            all_host_edges.add((i, j))

    # Precompute tree degrees and adjacency
    tree_degrees = []
    tree_adj = []
    for k_idx in range(20):
        k = k_idx + 1
        deg = [0] * k
        adj = [[] for _ in range(k)]
        for u, v in trees[k_idx]:
            deg[u] += 1
            deg[v] += 1
            adj[u].append(v)
            adj[v].append(u)
        tree_degrees.append(deg)
        tree_adj.append(adj)

    def time_up():
        return time.time() > deadline

    def embed_tree(tree_idx, available_edges, rng, timeout_sec=5.0):
        """
        Try to embed tree tree_idx into K20 using available_edges.
        Returns list of length k (tree_idx+1) mapping tree vertex i to host vertex,
        or None if not found.
        """
        k = tree_idx + 1
        if k == 1:
            # Single vertex: pick any host vertex
            return [0]
        if k == 2:
            # Need one edge
            for e in available_edges:
                return [e[0], e[1]]
            return None

        deg = tree_degrees[tree_idx]
        adj = tree_adj[tree_idx]

        # Order tree vertices by degree descending (high-degree first for pruning)
        order = sorted(range(k), key=lambda x: -deg[x])
        pos_in_order = [0] * k
        for i, v in enumerate(order):
            pos_in_order[v] = i

        assigned = [-1] * k  # assigned[i] = host vertex for order[i], -1 if unassigned
        used_hosts = set()

        # Track remaining degree capacity per host vertex
        # rem_deg[h] = number of unused edges incident to host h
        rem_deg = [0] * N
        for e in available_edges:
            rem_deg[e[0]] += 1
            rem_deg[e[1]] += 1

        # For pruning: for each tree vertex, we need deg[v] distinct neighbors
        # So host must have rem_deg >= deg[v] (roughly)

        # DFS with backtracking
        # Use iterative approach with explicit stack for deep recursion safety
        # But for k<=20, recursion depth is fine

        def dfs(idx):
            if time_up():
                return False
            if idx == k:
                return True

            v = order[idx]
            v_deg = deg[v]

            # Get already-assigned neighbors of v
            assigned_neighbors = []
            for u in adj[v]:
                if pos_in_order[u] < idx:
                    assigned_neighbors.append((u, assigned[pos_in_order[u]]))

            # Candidate host vertices: not used, and for each assigned neighbor,
            # the edge (h, hu) must be in available_edges
            # Also, rem_deg[h] >= v_deg (need enough unused edges)

            # Sort candidates by: fewest remaining options (MRV heuristic)
            # A host h is "constrained" if it has few available edges to unassigned tree neighbors
            # Simple heuristic: prefer hosts with lower rem_deg (more constrained)

            candidates = []
            for h in range(N):
                if h in used_hosts:
                    continue
                # Check edges to assigned neighbors
                valid = True
                for u, hu in assigned_neighbors:
                    edge = (min(h, hu), max(h, hu))
                    if edge not in available_edges:
                        valid = False
                        break
                if not valid:
                    continue
                # Degree constraint: host needs at least v_deg unused edges
                # (this is a necessary condition, not sufficient)
                if rem_deg[h] < v_deg:
                    continue
                candidates.append(h)

            if not candidates:
                return False

            # Shuffle with rng for diversity, but bias toward constrained hosts
            # Sort by rem_deg ascending (most constrained first) then shuffle ties
            candidates.sort(key=lambda h: rem_deg[h])
            rng.shuffle(candidates)

            for h in candidates:
                # Assign
                assigned[pos_in_order[v]] = h
                used_hosts.add(h)
                # Update rem_deg
                old_rem = rem_deg[h]
                rem_deg[h] -= 1  # will use at least one edge (to a neighbor or later)
                # Actually, we'll use deg[v] edges from h total, but we assign one at a time
                # Better: don't decrement here, just track

                # Try next
                if dfs(idx + 1):
                    return True

                # Undo
                assigned[pos_in_order[v]] = -1
                used_hosts.discard(h)
                rem_deg[h] = old_rem

            return False

        if dfs(0):
            # Construct mapping in original vertex order
            mapping = [0] * k
            for i, v in enumerate(order):
                mapping[v] = assigned[i]
            return mapping
        return None

    def get_used_edges(mapping, tree_idx):
        """Get the set of host edges used by embedding mapping of tree tree_idx."""
        edges = set()
        for u, v in trees[tree_idx]:
            hu, hv = mapping[u], mapping[v]
            edges.add((min(hu, hv), max(hu, hv)))
        return edges

    def try_packing(order, rng_master):
        """
        Try to pack trees in the given order.
        Returns (mappings, value) or (None, 0) on timeout.
        """
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
                value += 0  # T1 has 0 edges
                continue

            # Try multiple rng seeds for this tree
            found = False
            for attempt in range(3):
                if time_up():
                    break
                rng = Random(rng_master * 10000 + tree_idx * 100 + attempt)
                m = embed_tree(tree_idx, available, rng)
                if m is not None:
                    mappings[tree_idx] = m
                    used_edges |= get_used_edges(m, tree_idx)
                    available = all_host_edges - used_edges
                    value += len(tree_edges[tree_idx])
                    found = True
                    break

            if not found:
                mappings[tree_idx] = None
                # Continue with smaller trees

        return mappings, value

    # Strategy 1: Pack in descending order of tree size (T20 first, then T19, ...)
    # This is good because larger trees are harder to fit
    order1 = list(range(19, -1, -1))

    # Strategy 2: Pack in ascending order
    order2 = list(range(20))

    # Strategy 3: Pack by difficulty (high max-degree first)
    max_deg = [max(tree_degrees[i]) if tree_degrees[i] else 0 for i in range(20)]
    order3 = sorted(range(20), key=lambda i: -max_deg[i])

    best_mappings = [None] * 20
    best_value = 0

    # Try each strategy with a few rng_master values
    strategies = [
        (order1, 1),
        (order1, 42),
        (order1, 123),
        (order2, 1),
        (order2, 42),
        (order3, 1),
        (order3, 42),
        (order3, 123),
    ]

    for order, rng_master in strategies:
        if time_up():
            break
        mappings, value = try_packing(order, rng_master)
        if value > best_value:
            best_value = value
            best_mappings = mappings

    # If we haven't achieved full packing, try a more intensive search on the best partial
    # Focus on filling in missing trees
    if best_value < 190 and not time_up():
        # Try to fill in missing trees with more attempts
        missing = [i for i in range(20) if best_mappings[i] is None]
        if missing:
            # Rebuild used_edges from current mappings
            used_edges = set()
            for i in range(20):
                if best_mappings[i] is not None:
                    used_edges |= get_used_edges(best_mappings[i], i)
            available = all_host_edges - used_edges

            # Try to embed missing trees, largest first
            missing_sorted = sorted(missing, key=lambda i: -(i + 1))
            for tree_idx in missing_sorted:
                if time_up():
                    break
                k = tree_idx + 1
                if k == 1:
                    best_mappings[tree_idx] = [0]
                    best_value += 0
                    continue
                found = False
                for attempt in range(10):
                    if time_up():
                        break
                    rng = Random(99999 + tree_idx * 1000 + attempt)
                    m = embed_tree(tree_idx, available, rng)
                    if m is not None:
                        best_mappings[tree_idx] = m
                        used_edges |= get_used_edges(m, tree_idx)
                        available = all_host_edges - used_edges
                        best_value += len(tree_edges[tree_idx])
                        found = True
                        break

    # Ensure all entries are either None or valid lists of ints
    result = []
    for i in range(20):
        if best_mappings[i] is None:
            result.append(None)
        else:
            # Ensure it's a list of Python ints
            result.append([int(x) for x in best_mappings[i]])

    return result

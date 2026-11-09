"""Packing trees T1..T20 into K20 using backtracking edge-disjoint embedding."""
import time
from random import Random
from collections import defaultdict
from functools import lru_cache


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


def _build_tree_info(tree_edges):
    """Build adjacency and degree info for a tree."""
    if not tree_edges:
        return {}, {}, []
    verts = set()
    for u, v in tree_edges:
        verts.add(u)
        verts.add(v)
    verts = sorted(verts)
    adj = {v: [] for v in verts}
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)
    degree = {v: len(adj[v]) for v in verts}
    return adj, degree, verts


def _embed_with_backtracking(tree_edges, n, used_edges, rng, max_nodes=500000):
    """
    Embed tree_edges into K_n using backtracking with LCV heuristic and forward checking.
    """
    if not tree_edges:
        return {}

    adj, degree, verts = _build_tree_info(tree_edges)
    k = len(verts)

    # Try different roots: prefer high-degree vertices first, then leaves
    high_deg = [v for v in verts if degree[v] >= 3]
    leaves = [v for v in verts if degree[v] == 1]
    mid = [v for v in verts if degree[v] == 2]
    
    root_candidates = []
    if high_deg:
        root_candidates.extend(high_deg)
    if mid:
        root_candidates.extend(mid)
    if leaves:
        root_candidates.extend(leaves)
    if not root_candidates:
        root_candidates = verts[:]

    # Precompute available neighbors for each host vertex
    # neighbor_list[h] contains all x such that (h,x) is NOT in used_edges
    neighbor_list = [set() for _ in range(n)]
    for h in range(n):
        for x in range(n):
            if h == x: continue
            e = (min(h, x), max(h, x))
            if e not in used_edges:
                neighbor_list[h].add(x)

    for attempt in range(50):
        # Pick a root with some randomization
        if attempt < len(root_candidates):
            root = root_candidates[attempt % len(root_candidates)]
        else:
            root = rng.choice(root_candidates)

        # Build BFS order from root
        order = [root]
        parent = {root: -1}
        visited = {root}
        queue = [root]
        idx = 0
        while idx < len(queue):
            u = queue[idx]
            idx += 1
            for w in adj[u]:
                if w not in visited:
                    visited.add(w)
                    parent[w] = u
                    queue.append(w)
                    order.append(w)

        # Children map: for each vertex, list of children in the rooted tree
        children = {v: [] for v in order}
        for v in order[1:]:
            p = parent[v]
            children[p].append(v)

        mapping = {}
        host_used = set()
        node_count = [0]

        def backtrack(pos):
            if node_count[0] > max_nodes:
                return False
            node_count[0] += 1

            if pos == len(order):
                return True

            v = order[pos]
            p = parent[v]

            if p == -1:
                candidates = [h for h in range(n) if h not in host_used]
            else:
                hp = mapping[p]
                candidates = [h for h in range(n)
                              if h not in host_used
                              and (min(hp, h), max(hp, h)) not in used_edges]

            if not candidates:
                return False

            # LCV Heuristic: Score each candidate host vertex by how well it
            # can accommodate all of v's children. The score is the minimum,
            # over all children, of the number of free neighbors that child
            # would have (excluding h itself). This is a tighter forward
            # check than just counting total free neighbors of h.
            def heuristic_score(h):
                child_list = children[v]
                if not child_list:
                    # Leaf: just count free neighbors
                    return sum(1 for x in neighbor_list[h] if x not in host_used)
                
                # For each child c, the number of free neighbors of h
                # (excluding h itself, which is always excluded since h is used)
                # that are not in host_used
                # All children share the same free neighbor set from h's perspective
                free_set = frozenset(x for x in neighbor_list[h] if x not in host_used)
                # The bottleneck is: each child needs at least 1 free neighbor
                # of h. Since they all share the same pool, we need
                # len(free_set) >= len(child_list)
                if len(free_set) < len(child_list):
                    return -1000000 + len(free_set)
                
                # Secondary: prefer h with more free neighbors overall
                return len(free_set)

            candidates.sort(key=heuristic_score, reverse=True)

            # Add some randomization: shuffle top candidates slightly
            if len(candidates) > 1:
                # Keep top few but randomize order among them
                top_k = min(7, len(candidates))
                top = candidates[:top_k]
                rng.shuffle(top)
                candidates = top + candidates[top_k:]

            for h in candidates:
                mapping[v] = h
                host_used.add(h)
                
                # Forward check: verify children have available neighbors
                feasible = True
                if children[v]:
                    free_neighbors = set()
                    for x in neighbor_list[h]:
                        if x not in host_used:
                            free_neighbors.add(x)
                    if len(free_neighbors) < len(children[v]):
                        feasible = False
                
                if feasible and backtrack(pos + 1):
                    return True
                
                del mapping[v]
                host_used.discard(h)

            return False

        if backtrack(0):
            return dict(mapping)

    return None


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    trees = instance_trees(20, 611)
    n = 20

    # Process trees in order of decreasing number of edges (most constrained first)
    # But we need to return in original order
    def tree_constraint(i):
        if not trees[i]:
            return (0, 0)
        adj_t = defaultdict(list)
        for u, v in trees[i]:
            adj_t[u].append(v)
            adj_t[v].append(u)
        max_deg = max((len(adj_t[v]) for v in adj_t), default=0)
        return (len(trees[i]), max_deg)
    
    order = list(range(20))
    order.sort(key=lambda i: tree_constraint(i), reverse=True)

    # Try multiple global random seeds
    best_total = 0
    best_result = [None] * 20

    for global_seed in range(200):
        if time.time() > deadline - 3.0:
            break

        rng = Random(global_seed)
        used_edges = set()
        result = [None] * 20
        success_count = 0

        for idx in order:
            if time.time() > deadline - 2.0:
                break

            k = idx + 1
            tree_edges = trees[idx]

            if not tree_edges:
                # Tree with 1 vertex (k=1)
                result[idx] = [0]
                success_count += 1
                continue

            mapping = _embed_with_backtracking(tree_edges, n, used_edges, rng)
            if mapping is not None:
                result_list = []
                for v in range(k):
                    result_list.append(int(mapping[v]))
                result[idx] = result_list
                success_count += 1

                for u, v in tree_edges:
                    hu, hv = mapping[u], mapping[v]
                    e = (min(hu, hv), max(hu, hv))
                    used_edges.add(e)

        # Calculate total edges embedded
        total_edges = sum(len(trees[i]) for i in range(20) if result[i] is not None)

        if total_edges > best_total:
            best_total = total_edges
            best_result = [list(r) if r is not None else None for r in result]

        # Early exit if we found a perfect packing (all 190 edges)
        if best_total >= 190:
            break

    return best_result

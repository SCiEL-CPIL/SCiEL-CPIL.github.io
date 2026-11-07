"""Rewritten program for packing trees T1..T20 into K20."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import deque

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
    deadline = start_time + 145.0  # Safety margin before 150s limit

    trees = instance_trees(20, 611)
    num_trees = 20
    n_hosts = 20

    # Pre-compute tree data
    tree_data = []
    for i in range(num_trees):
        k = i + 1
        if k == 1:
            tree_data.append({'edges': [], 'adj': [[]], 'size': 1})
        elif k == 2:
            tree_data.append({'edges': [(0, 1)], 'adj': [[1], [0]], 'size': 2})
        else:
            edges = trees[i]
            adj = [[] for _ in range(k)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            tree_data.append({'edges': edges, 'adj': adj, 'size': k})

    # All possible edges in K20
    all_edges = set()
    for u in range(n_hosts):
        for v in range(u + 1, n_hosts):
            all_edges.add((u, v))

    # Strategy: Embed trees one by one, largest first, using CSP with MRV
    # We use randomized restarts to explore different embedding possibilities

    # Order: largest trees first (index 19 down to 0)
    order = list(range(19, -1, -1))

    best_score = 0
    best_mappings = [None] * 20

    # Precompute degree of each vertex in each tree
    tree_degrees = []
    for i in range(num_trees):
        k = tree_data[i]['size']
        if k <= 1:
            tree_degrees.append([0] * k)
        else:
            deg = [0] * k
            for u, v in tree_data[i]['edges']:
                deg[u] += 1
                deg[v] += 1
            tree_degrees.append(deg)

    def embed_tree_csp(tree_idx, used_edges, rng):
        """
        Embed tree 'tree_idx' into K20 avoiding 'used_edges'.
        Returns (mapping, new_used_edges) or (None, None).
        """
        k = tree_data[tree_idx]['size']
        if k == 1:
            for v in range(n_hosts):
                return [v], used_edges
            return None, None

        if k == 2:
            for u in range(n_hosts):
                for v in range(u + 1, n_hosts):
                    e_key = (u, v)
                    if e_key not in used_edges:
                        return [u, v], used_edges | {e_key}
            return None, None

        # For k > 2: CSP with MRV
        adj = tree_data[tree_idx]['adj']
        edges = tree_data[tree_idx]['edges']
        degrees = tree_degrees[tree_idx]

        # Choose root: highest degree vertex
        root = max(range(k), key=lambda i: degrees[i])

        # Precompute: for each tree vertex, its neighbors in the tree
        # We'll do BFS/DFS from root to get an ordering

        # Find a good ordering: root first, then neighbors, etc.
        # Actually, let's just use MRV dynamically

        def backtrack(assigned, used_hosts):
            if len(assigned) == k:
                # Verify all edges are valid
                for u, v in edges:
                    hu = assigned[u]
                    hv = assigned[v]
                    e_key = (min(hu, hv), max(hu, hv))
                    if e_key in used_edges:
                        return None
                return assigned.copy()

            # Select unassigned vertex with MRV (Minimum Remaining Values)
            unassigned = [v for v in range(k) if v not in assigned]
            best_v = -1
            min_domain_size = n_hosts + 1
            best_domain = []

            for v in unassigned:
                valid_hosts = []
                for h in range(n_hosts):
                    if h in used_hosts:
                        continue
                    valid = True
                    for u in adj[v]:
                        if u in assigned:
                            hu = assigned[u]
                            e_key = (min(h, hu), max(h, hu))
                            if e_key in used_edges:
                                valid = False
                                break
                    if valid:
                        valid_hosts.append(h)

                if len(valid_hosts) < min_domain_size:
                    min_domain_size = len(valid_hosts)
                    best_v = v
                    best_domain = valid_hosts
                    if min_domain_size == 0:
                        break

            if best_v == -1 or min_domain_size == 0:
                return None

            # Randomize the domain order
            rng.shuffle(best_domain)

            for h in best_domain:
                new_assigned = dict(assigned)
                new_assigned[best_v] = h
                new_used_hosts = used_hosts | {h}
                result = backtrack(new_assigned, new_used_hosts)
                if result is not None:
                    return result

            return None

        # Try different roots and initial assignments
        root_hosts = list(range(n_hosts))
        rng.shuffle(root_hosts)

        for h in root_hosts:
            initial_assigned = {root: h}
            initial_used_hosts = {h}
            result = backtrack(initial_assigned, initial_used_hosts)
            if result is not None:
                mapping = [0] * k
                for tv, hv in result.items():
                    mapping[tv] = hv
                new_used_edges = set(used_edges)
                for u, v in edges:
                    hu = mapping[u]
                    hv = mapping[v]
                    e_key = (min(hu, hv), max(hu, hv))
                    new_used_edges.add(e_key)
                return mapping, new_used_edges

        return None, None

    # Main search loop with randomized restarts
    max_restarts = 5000
    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        rng = random.Random(restart * 7919 + 12345)
        current_used_edges = set()
        current_mappings = [None] * 20
        success = True

        for tree_idx in order:
            if time.time() > deadline:
                success = False
                break

            mapping, new_used_edges = embed_tree_csp(tree_idx, current_used_edges, rng)
            if mapping is None:
                # Try to embed remaining smaller trees
                success = False
                for small_idx in range(num_trees):
                    if time.time() > deadline:
                        break
                    if current_mappings[small_idx] is None:
                        m2, e2 = embed_tree_csp(small_idx, current_used_edges, rng)
                        if m2 is not None:
                            current_mappings[small_idx] = m2
                            current_used_edges = e2
                break

            current_mappings[tree_idx] = mapping
            current_used_edges = new_used_edges

        # Calculate score
        score = 0
        for i in range(num_trees):
            if current_mappings[i] is not None:
                score += len(tree_data[i]['edges'])

        if score > best_score:
            best_score = score
            best_mappings = [m[:] if m else None for m in current_mappings]

        if best_score == 190:
            break

    # Final result
    final_result = []
    for i in range(num_trees):
        if best_mappings[i] is None:
            final_result.append(None)
        else:
            final_result.append([int(x) for x in best_mappings[i]])

    return final_result

# EVOLVE-BLOCK-END

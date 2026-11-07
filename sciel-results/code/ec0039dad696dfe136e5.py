import time
import random
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
    deadline = start_time + 145

    trees = instance_trees(20, 611)

    # Precompute tree properties
    tree_info = []
    for i in range(20):
        edges = trees[i]
        k = i + 1
        adj = [[] for _ in range(k)]
        deg = [0] * k
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            deg[u] += 1
            deg[v] += 1
        tree_info.append({
            'k': k,
            'edges': edges,
            'adj': adj,
            'deg': deg,
            'num_edges': len(edges)
        })

    # Sort trees by size descending
    base_order = sorted(range(20), key=lambda i: -tree_info[i]['k'])

    def try_pack_greedy(attempt_seed, order):
        """Greedy packer with randomized choices"""
        local_rng = random.Random(attempt_seed)
        used_edges = set()
        avail_deg = [19] * 20
        mappings = [None] * 20
        success = True

        for t_idx in order:
            if time.time() > deadline:
                success = False
                break
            info = tree_info[t_idx]
            k = info['k']
            adj_t = info['adj']
            deg_t = info['deg']

            if k == 1:
                candidates = [hv for hv in range(20) if avail_deg[hv] > 0]
                if not candidates:
                    success = False
                    break
                hv = local_rng.choice(candidates)
                mappings[t_idx] = [hv]
                continue

            # BFS order from highest-degree root
            root = max(range(k), key=lambda v: deg_t[v])
            visited = [False] * k
            visited[root] = True
            proc_order = [root]
            parent = [-1] * k
            queue = deque([root])
            while queue:
                u = queue.popleft()
                for v in adj_t[u]:
                    if not visited[v]:
                        visited[v] = True
                        parent[v] = u
                        proc_order.append(v)
                        queue.append(v)

            mapping = [-1] * k
            used_hosts = set()
            local_used_edges = set()
            local_avail = avail_deg[:]
            feasible = True

            for pos in range(k):
                tree_v = proc_order[pos]
                p_tree = parent[tree_v]

                if p_tree >= 0:
                    p_host = mapping[p_tree]
                    candidates = []
                    for hv in range(20):
                        if hv in used_hosts:
                            continue
                        e = (min(p_host, hv), max(p_host, hv))
                        if e in used_edges or e in local_used_edges:
                            continue
                        candidates.append(hv)
                else:
                    candidates = [hv for hv in range(20) if hv not in used_hosts]

                if not candidates:
                    feasible = False
                    break

                # Score: prefer vertices with more remaining capacity
                def score_cand(hv):
                    s = local_avail[hv] * 10.0
                    s += local_rng.random() * 2.0
                    return s

                candidates.sort(key=score_cand, reverse=True)
                hv = candidates[0]

                mapping[tree_v] = hv
                used_hosts.add(hv)
                if p_tree >= 0:
                    p_host = mapping[p_tree]
                    e = (min(p_host, hv), max(p_host, hv))
                    local_used_edges.add(e)
                    local_avail[p_host] -= 1
                    local_avail[hv] -= 1

            if not feasible:
                success = False
                break

            # Commit
            for u, v in info['edges']:
                hu = mapping[u]
                hv = mapping[v]
                e = (min(hu, hv), max(hu, hv))
                if e in used_edges:
                    success = False
                    break
                used_edges.add(e)
                avail_deg[hu] -= 1
                avail_deg[hv] -= 1

            if not success:
                break
            mappings[t_idx] = mapping[:]

        if not success:
            return None, 0

        score = sum(tree_info[i]['num_edges'] for i in range(20) if mappings[i] is not None)
        return mappings, score

    def try_pack_backtrack(attempt_seed, order):
        """Backtracking packer with tighter limits"""
        local_rng = random.Random(attempt_seed)
        used_edges = set()
        avail_deg = [19] * 20
        mappings = [None] * 20

        def pack_tree_bt(t_idx, max_nodes=30000):
            info = tree_info[t_idx]
            k = info['k']
            adj_t = info['adj']
            deg_t = info['deg']

            if k == 1:
                for hv in range(20):
                    if avail_deg[hv] > 0:
                        return [hv]
                return None

            root = max(range(k), key=lambda v: deg_t[v])
            visited = [False] * k
            visited[root] = True
            proc_order = [root]
            parent = [-1] * k
            queue = deque([root])
            while queue:
                u = queue.popleft()
                for v in adj_t[u]:
                    if not visited[v]:
                        visited[v] = True
                        parent[v] = u
                        proc_order.append(v)
                        queue.append(v)

            mapping = [-1] * k
            used_hosts = set()
            local_used_edges = set()
            local_avail = avail_deg[:]
            node_count = [0]

            def backtrack(pos):
                nonlocal local_avail
                if time.time() > deadline or node_count[0] > max_nodes:
                    return False

                if pos == k:
                    for u, v in info['edges']:
                        hu = mapping[u]
                        hv = mapping[v]
                        e = (min(hu, hv), max(hu, hv))
                        if e in used_edges:
                            return False
                        used_edges.add(e)
                        avail_deg[hu] -= 1
                        avail_deg[hv] -= 1
                    return True

                tree_v = proc_order[pos]
                p_tree = parent[tree_v]
                p_host = mapping[p_tree] if p_tree >= 0 else -1

                if p_host >= 0:
                    candidates = []
                    for hv in range(20):
                        if hv in used_hosts:
                            continue
                        e = (min(p_host, hv), max(p_host, hv))
                        if e in used_edges or e in local_used_edges:
                            continue
                        candidates.append(hv)
                else:
                    candidates = [hv for hv in range(20) if hv not in used_hosts]

                if not candidates:
                    return False

                candidates.sort(key=lambda hv: local_avail[hv], reverse=True)
                trial = candidates[:min(len(candidates), 5)]
                local_rng.shuffle(trial)

                for hv in trial:
                    node_count[0] += 1
                    mapping[tree_v] = hv
                    used_hosts.add(hv)
                    if p_host >= 0:
                        e = (min(p_host, hv), max(p_host, hv))
                        local_used_edges.add(e)
                        local_avail[p_host] -= 1
                        local_avail[hv] -= 1

                    if backtrack(pos + 1):
                        return True

                    used_hosts.discard(hv)
                    if p_host >= 0:
                        e = (min(p_host, hv), max(p_host, hv))
                        local_used_edges.discard(e)
                        local_avail[p_host] += 1
                        local_avail[hv] += 1
                    mapping[tree_v] = -1

                return False

            if backtrack(0):
                return mapping[:]
            return None

        for t_idx in order:
            if time.time() > deadline:
                break
            info = tree_info[t_idx]
            k = info['k']

            if k == 1:
                for hv in range(20):
                    if avail_deg[hv] > 0:
                        mappings[t_idx] = [hv]
                        break
                continue

            result = pack_tree_bt(t_idx)
            if result is not None:
                mappings[t_idx] = result

        score = sum(tree_info[i]['num_edges'] for i in range(20) if mappings[i] is not None)
        return mappings, score

    best_mappings = [None] * 20
    best_score = 0

    # Try multiple orders
    orders = [base_order]
    for s in range(30):
        o = base_order[:]
        random.Random(s * 7919 + 42).shuffle(o)
        orders.append(o)

    attempt = 0
    while time.time() < deadline:
        order = orders[attempt % len(orders)]
        # First try greedy (fast), fall back to backtracking
        mappings, score = try_pack_greedy(attempt * 7919 + 1, order)
        if mappings is None or score < best_score:
            mappings, score = try_pack_backtrack(attempt * 7919 + 2, order)

        if mappings is not None and score > best_score:
            best_score = score
            best_mappings = mappings[:]
            if best_score == 190:
                break
        attempt += 1

    result = []
    for i in range(20):
        if best_mappings[i] is not None:
            result.append([int(v) for v in best_mappings[i]])
        else:
            result.append(None)
    return result

if __name__ == "__main__":
    start = time.time()
    result = construct()
    elapsed = time.time() - start
    print(f"Time: {elapsed:.2f}s")
    
    # Verify
    trees = instance_trees(20, 611)
    used_edges = set()
    total_edges = 0
    valid = True
    for i in range(20):
        if result[i] is None:
            continue
        mapping = result[i]
        k = i + 1
        if len(mapping) != k:
            print(f"Tree {i}: wrong length {len(mapping)} vs {k}")
            valid = False
            break
        if len(set(mapping)) != k:
            print(f"Tree {i}: duplicate host vertices")
            valid = False
            break
        for v in mapping:
            if not isinstance(v, int) or v < 0 or v > 19:
                print(f"Tree {i}: invalid vertex {v}")
                valid = False
                break
        if not valid:
            break
        for u, v in trees[i]:
            hu = mapping[u]
            hv = mapping[v]
            e = (min(hu, hv), max(hu, hv))
            if e in used_edges:
                print(f"Tree {i}: edge conflict {e}")
                valid = False
                break
            used_edges.add(e)
            total_edges += 1
        if not valid:
            break
    
    print(f"Valid: {valid}")
    print(f"Total edges: {total_edges}")
    if valid:
        print(f"Score: {total_edges}")
    else:
        print("INVALID RESULT")

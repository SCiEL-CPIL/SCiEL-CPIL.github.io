import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    t_start = time.time()
    DEADLINE = 85.0

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    adj_set = adj
    degrees = [len(adj[v]) for v in range(n)]
    sorted_verts = sorted(range(n), key=lambda x: degrees[x], reverse=True)

    def time_up():
        return time.time() - t_start > DEADLINE

    def find_shortest_path(u, v, banned):
        if u == v:
            return None
        if v in adj_set[u]:
            return [u, v]
        parent = {u: None}
        q = deque([u])
        while q:
            cur = q.popleft()
            for nxt in adj_set[cur]:
                if nxt == v:
                    path = [v]
                    node = cur
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                if nxt in parent or nxt in banned:
                    continue
                parent[nxt] = cur
                q.append(nxt)
        return None

    def find_all_shortest_paths(u, v, banned, max_paths=10):
        """Find all shortest paths from u to v avoiding banned vertices."""
        if u == v:
            return []
        if v in adj_set[u]:
            return [[u, v]]
        
        # BFS to find distance
        dist = {u: 0}
        q = deque([u])
        while q:
            cur = q.popleft()
            for nxt in adj_set[cur]:
                if nxt in banned or nxt in dist:
                    continue
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
        
        if v not in dist:
            return []
        
        target_dist = dist[v]
        paths = []
        
        # DFS to find all paths of length target_dist
        def dfs(cur, path, depth):
            if len(paths) >= max_paths:
                return
            if cur == v and depth == target_dist:
                paths.append(list(path))
                return
            if depth >= target_dist:
                return
            for nxt in adj_set[cur]:
                if nxt in banned:
                    continue
                if nxt in dist and dist[nxt] == depth + 1:
                    path.append(nxt)
                    dfs(nxt, path, depth + 1)
                    path.pop()
        
        dfs(u, [u], 0)
        return paths

    def find_paths_greedy(branches, rng, attempts=50):
        """Try to find disjoint paths for all pairs using multiple orderings."""
        bset = set(branches)
        pairs = list(combinations(branches, 2))
        deg_order = sorted(pairs, key=lambda p: degrees[p[0]] + degrees[p[1]])
        
        for attempt in range(attempts):
            if time_up():
                break
            if attempt == 0:
                order = list(deg_order)
            elif attempt % 3 == 1:
                order = list(deg_order)
                order.reverse()
            else:
                order = list(pairs)
                rng.shuffle(order)
            
            used_interior = set()
            paths = {}
            success = True
            for a, b in order:
                banned = used_interior | bset
                banned.discard(a)
                banned.discard(b)
                path = find_shortest_path(a, b, banned)
                if path is None:
                    success = False
                    break
                interior = set(path[1:-1])
                if interior & bset or interior & used_interior:
                    success = False
                    break
                used_interior |= interior
                paths[frozenset((a, b))] = path
            if success:
                return paths
        return None

    def find_paths_backtrack(branches, rng, max_nodes=50000):
        """Backtracking search for disjoint paths."""
        bset = set(branches)
        pairs = list(combinations(branches, 2))
        # Sort pairs by constraint: fewer available interior vertices first
        pairs.sort(key=lambda p: len(adj_set[p[0]] | adj_set[p[1]]))
        
        used_interior = set()
        paths = {}
        nodes_visited = 0
        
        def bt(idx):
            nonlocal nodes_visited
            if time_up():
                return False
            if nodes_visited > max_nodes:
                return False
            if idx == len(pairs):
                return True
            a, b = pairs[idx]
            banned = used_interior | bset
            banned.discard(a)
            banned.discard(b)
            
            # Try all shortest paths first
            candidates = find_all_shortest_paths(a, b, banned, max_paths=20)
            
            # Also try some slightly longer paths if needed
            if not candidates:
                # Try BFS with relaxed constraint
                path = find_shortest_path(a, b, banned)
                if path:
                    candidates = [path]
            
            for path in candidates:
                interior = set(path[1:-1])
                if interior & bset or interior & used_interior:
                    continue
                used_interior_new = used_interior | interior
                paths[frozenset((a, b))] = path
                if bt(idx + 1):
                    return True
                del paths[frozenset((a, b))]
            
            return False
        
        if bt(0):
            return paths
        return None

    def validate(branches, paths):
        if not branches or len(branches) < 2:
            return 0
        bset = set(branches)
        if len(bset) != len(branches):
            return 0
        expected_pairs = set(frozenset((a, b)) for a, b in combinations(branches, 2))
        if len(paths) != len(expected_pairs):
            return 0
        used_interior = set()
        for fkey, path in paths.items():
            if len(path) < 2:
                return 0
            a, b = path[0], path[-1]
            if a not in bset or b not in bset:
                return 0
            if a == b:
                return 0
            if fkey not in expected_pairs:
                return 0
            interior = set(path[1:-1])
            if interior & bset:
                return 0
            if interior & used_interior:
                return 0
            for i in range(len(path) - 1):
                if path[i + 1] not in adj_set[path[i]]:
                    return 0
            used_interior |= interior
        return len(branches)

    def local_search_improve(branches, paths, rng):
        """Try to add one more branch to an existing valid solution."""
        bset = set(branches)
        candidates_to_add = [v for v in range(n) if v not in bset]
        candidates_to_add.sort(key=lambda v: degrees[v], reverse=True)
        
        for v in candidates_to_add:
            if time_up():
                return None
            new_branches = list(branches) + [v]
            # Re-find paths with the new branch
            new_paths = find_paths_greedy(new_branches, rng, attempts=30)
            if new_paths is None:
                new_paths = find_paths_backtrack(new_branches, rng, max_nodes=20000)
            if new_paths is not None:
                cnt = validate(new_branches, new_paths)
                if cnt > len(branches):
                    return (new_branches, new_paths)
        return None

    def hill_climbing(branches, paths, rng, max_iters=50):
        """Hill climbing: swap branches to find better configurations."""
        current_branches = list(branches)
        current_paths = paths
        current_count = len(current_branches)
        improved = True
        
        while improved and not time_up():
            improved = False
            # Try swapping one branch for another
            for i in range(len(current_branches)):
                if time_up():
                    break
                for j in range(n):
                    if j in current_branches:
                        continue
                    new_branches = list(current_branches)
                    new_branches[i] = j
                    new_paths = find_paths_greedy(new_branches, rng, attempts=20)
                    if new_paths is None:
                        new_paths = find_paths_backtrack(new_branches, rng, max_nodes=10000)
                    if new_paths is not None:
                        cnt = validate(new_branches, new_paths)
                        if cnt > current_count:
                            current_branches = new_branches
                            current_paths = new_paths
                            current_count = cnt
                            improved = True
                            break
                if improved:
                    break
            # Try adding a branch
            if not improved:
                result = local_search_improve(current_branches, current_paths, rng)
                if result:
                    current_branches, current_paths = result
                    current_count = len(current_branches)
                    improved = True
        
        return current_branches, current_paths

    best_count = 0
    best_result = None
    rng = random.Random(42)

    # Phase 1: Try to find large clique subdivisions
    # Start with high k and work down
    for k in range(10, 1, -1):
        if time_up():
            break
        
        # Generate diverse candidate sets
        candidates = set()
        
        # Top k by degree
        candidates.add(tuple(sorted_verts[:k]))
        
        # Top k-1 + random
        for _ in range(5):
            rest = [v for v in range(n) if v not in sorted_verts[:k-1]]
            if rest:
                cand = sorted_verts[:k-1] + [rng.choice(rest)]
                candidates.add(tuple(cand))
        
        # Random samples
        for _ in range(30):
            candidates.add(tuple(rng.sample(range(n), k)))
        
        # Local search: start with top k, try swapping
        base = sorted_verts[:k]
        for _ in range(10):
            if time_up():
                break
            cand = list(base)
            idx = rng.randint(0, k-1)
            rest = [v for v in range(n) if v not in cand]
            if rest:
                cand[idx] = rng.choice(rest)
                candidates.add(tuple(cand))
        
        for cand_tuple in candidates:
            if time_up():
                break
            cand = list(cand_tuple)
            
            # First try greedy
            paths = find_paths_greedy(cand, rng, attempts=30)
            if paths is None:
                # Try backtracking
                paths = find_paths_backtrack(cand, rng, max_nodes=20000)
            
            if paths is not None:
                cnt = validate(cand, paths)
                if cnt > best_count:
                    best_count = cnt
                    best_result = (cand, paths)
                    break
        
        if best_result is not None and best_count >= k:
            break

    # Phase 2: Hill climbing to improve
    if not time_up() and best_result is not None:
        improved_branches, improved_paths = hill_climbing(
            best_result[0], best_result[1], rng, max_iters=30
        )
        cnt = validate(improved_branches, improved_paths)
        if cnt > best_count:
            best_count = cnt
            best_result = (improved_branches, improved_paths)

    # Phase 3: Greedy incremental growth
    if not time_up():
        if best_result is not None:
            current = list(best_result[0])
        else:
            current = sorted_verts[:2]
        
        current_paths = find_paths_greedy(current, rng, attempts=50)
        if current_paths is None:
            current_paths = find_paths_backtrack(current, rng, max_nodes=30000)
        
        for _ in range(15):
            if time_up():
                break
            if current_paths is None:
                break
            
            added = False
            candidates_to_add = [v for v in range(n) if v not in set(current)]
            candidates_to_add.sort(key=lambda v: degrees[v], reverse=True)
            
            for v in candidates_to_add:
                if time_up():
                    break
                new_branches = current + [v]
                new_paths = find_paths_greedy(new_branches, rng, attempts=20)
                if new_paths is None:
                    new_paths = find_paths_backtrack(new_branches, rng, max_nodes=15000)
                
                if new_paths is not None:
                    cnt = validate(new_branches, new_paths)
                    if cnt > best_count:
                        best_count = cnt
                        best_result = (new_branches, new_paths)
                        current = new_branches
                        current_paths = new_paths
                        added = True
                        break
            if not added:
                break

    # Fallback
    if best_result is None:
        u, v = edges[0]
        branches = [u, v]
        paths = {frozenset((u, v)): [u, v]}
        best_result = (branches, paths)

    branches, paths = best_result

    branch_list = [int(b) for b in sorted(branches)]
    path_list = []
    for a, b in combinations(sorted(branches), 2):
        fkey = frozenset((a, b))
        p = paths.get(fkey)
        if p is None:
            p = [a, b]
        path_list.append([int(x) for x in p])

    return {'branches': branch_list, 'paths': path_list}

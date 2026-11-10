import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    n = P['n']
    adj_list = [[] for _ in range(n)]
    adj_set = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj_list[u].append(v)
        adj_list[v].append(u)
        adj_set[u].add(v)
        adj_set[v].add(u)

    def time_ok():
        return time.time() < deadline

    def bfs_path(src, dst, blocked):
        if src == dst:
            return [src]
        parent = {src: None}
        queue = deque([src])
        while queue:
            node = queue.popleft()
            if node == dst:
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            for neighbor in adj_list[node]:
                if neighbor not in parent and neighbor not in blocked:
                    parent[neighbor] = node
                    queue.append(neighbor)
        return None

    def find_paths_for_branches(branches):
        """Returns dict pair->path or None if infeasible."""
        branch_set = set(branches)
        pairs = list(combinations(sorted(branches), 2))

        pair_info = {}
        non_direct_pairs = []
        for u, v in pairs:
            if u in adj_set[v]:
                pair_info[(u, v)] = ('direct', None)
            else:
                common = set()
                for w in adj_list[u]:
                    if w not in branch_set and w in adj_set[v]:
                        common.add(w)
                if len(common) == 0:
                    return None
                pair_info[(u, v)] = ('need', common)
                non_direct_pairs.append((u, v))

        if non_direct_pairs:
            # Hopcroft-Karp bipartite matching
            all_internals = set()
            for p in non_direct_pairs:
                all_internals.update(pair_info[p][1])

            left_nodes = list(range(len(non_direct_pairs)))
            right_nodes = list(all_internals)
            right_to_idx = {v: i for i, v in enumerate(right_nodes)}

            adj_match = [[] for _ in range(len(left_nodes))]
            for i, p in enumerate(non_direct_pairs):
                for w in pair_info[p][1]:
                    adj_match[i].append(right_to_idx[w])

            pair_u = [-1] * len(left_nodes)
            pair_v = [-1] * len(right_nodes)
            dist = [0] * len(left_nodes)

            def hk_bfs():
                q = deque()
                found = False
                for u in range(len(left_nodes)):
                    if pair_u[u] == -1:
                        dist[u] = 0
                        q.append(u)
                    else:
                        dist[u] = -1
                while q:
                    u = q.popleft()
                    for v in adj_match[u]:
                        pu = pair_v[v]
                        if pu == -1:
                            found = True
                        elif dist[pu] == -1:
                            dist[pu] = dist[u] + 1
                            q.append(pu)
                return found

            def hk_dfs(u):
                for v in adj_match[u]:
                    pu = pair_v[v]
                    if pu == -1 or (dist[pu] == dist[u] + 1 and hk_dfs(pu)):
                        pair_u[u] = v
                        pair_v[v] = u
                        return True
                dist[u] = -1
                return False

            matching_size = 0
            while hk_bfs():
                for u in range(len(left_nodes)):
                    if pair_u[u] == -1:
                        if hk_dfs(u):
                            matching_size += 1

            if matching_size != len(left_nodes):
                # Fall back to BFS backtracking
                return find_paths_longer(branches)

            assignment = {}
            for i, p in enumerate(non_direct_pairs):
                assignment[p] = [right_nodes[pair_u[i]]]
        else:
            assignment = {}

        result_paths = {}
        for pair in pairs:
            u, v = pair
            if pair_info[pair][0] == 'direct':
                result_paths[pair] = [u, v]
            else:
                internal = assignment[pair]
                result_paths[pair] = [u] + internal + [v]

        return result_paths

    def find_paths_longer(branches):
        """Fallback: BFS-based backtracking for pairs needing longer paths."""
        branch_set = set(branches)
        pairs = list(combinations(sorted(branches), 2))
        
        non_direct = [(u, v) for u, v in pairs if u not in adj_set[v]]
        if not non_direct:
            return {pair: [u, v] for pair in pairs for u, v in [pair]}
        
        non_direct.sort(key=lambda p: len(adj_set[p[0]] & adj_set[p[1]] - branch_set))
        
        used_int = set()
        assignment = {}
        
        def backtrack(idx):
            if not time_ok():
                return False
            if idx == len(non_direct):
                return True
            u, v = non_direct[idx]
            blocked = branch_set | used_int
            blocked.discard(u)
            blocked.discard(v)
            path = bfs_path(u, v, blocked)
            if path is None:
                return False
            internal = path[1:-1]
            if len(internal) != len(set(internal)):
                return False
            for w in internal:
                if w in branch_set:
                    return False
            for w in internal:
                used_int.add(w)
            assignment[(u, v)] = internal
            if backtrack(idx + 1):
                return True
            for w in internal:
                used_int.discard(w)
            del assignment[(u, v)]
            return False
        
        if not backtrack(0):
            return None
        
        result_paths = {}
        for pair in pairs:
            u, v = pair
            if u in adj_set[v]:
                result_paths[pair] = [u, v]
            else:
                result_paths[pair] = [u] + assignment[pair] + [v]
        return result_paths

    def find_maximal_cliques():
        cliques = []
        R = set()
        Pset = set(range(n))
        X = set()
        def bronk(R, Pset, X):
            if not Pset and not X:
                cliques.append(frozenset(R))
                return
            if Pset or X:
                U = Pset | X
                pivot = max(U, key=lambda v: len(Pset & adj_set[v]))
                candidates = Pset - adj_set[pivot]
            else:
                candidates = Pset
            for v in list(candidates):
                if not time_ok():
                    return
                bronk(R | {v}, Pset & adj_set[v], X & adj_set[v])
                Pset.remove(v)
                X.add(v)
        bronk(R, Pset, X)
        return cliques

    cliques = find_maximal_cliques()
    cliques = sorted(cliques, key=lambda c: -len(c))

    best_branches = None
    best_paths_dict = None

    for clique in cliques:
        if not time_ok():
            break
        clique_list = list(clique)
        if best_branches is not None and len(clique_list) <= len(best_branches):
            continue
        paths_result = find_paths_for_branches(clique_list)
        if paths_result is not None:
            best_branches = clique_list
            best_paths_dict = paths_result

    if best_branches is not None:
        current = best_branches[:]
        expanded = True
        while expanded and time_ok():
            expanded = False
            current_set = set(current)
            candidates = [v for v in range(n) if v not in current_set]
            candidates.sort(key=lambda v: -len(adj_set[v]))
            for v in candidates:
                if not time_ok():
                    break
                new_branches = current + [v]
                paths_result = find_paths_for_branches(new_branches)
                if paths_result is not None:
                    current = new_branches
                    best_branches = current[:]
                    best_paths_dict = paths_result
                    expanded = True
                    break

    if best_branches is None or len(best_branches) < 2:
        u, v = P['edges'][0]
        best_branches = [int(u), int(v)]
        best_paths = [[int(u), int(v)]]
    else:
        best_branches = [int(v) for v in best_branches]
        best_paths = []
        for pair in combinations(sorted(best_branches), 2):
            p = best_paths_dict.get(pair)
            if p is not None:
                best_paths.append([int(v) for v in p])

    return {'branches': best_branches, 'paths': best_paths}

"""Large clique subdivisions in a pinned graph - conflict-graph MIS approach."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict, deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    deadline = time.time() + 75.0  # Safety margin before 90s limit

    edge_set = set()
    for u, v in edges:
        edge_set.add((u, v))
        edge_set.add((v, u))

    def find_all_short_paths(src, dst, forbidden, max_len, max_paths=6):
        """Find up to max_paths simple paths from src to dst with length <= max_len,
        avoiding forbidden vertices. Returns list of paths (each a list of vertices)."""
        if src == dst:
            return [[src, dst]] if (src, dst) in edge_set else []

        blocked = set(forbidden)
        # BFS to find shortest paths, then try slightly longer
        results = []
        # Use BFS layer by layer
        queue = deque()
        queue.append((src, [src]))
        visited_at_depth = {src: 0}
        found_at_depth = {}

        while queue:
            node, path = queue.popleft()
            depth = len(path) - 1
            if depth >= max_len:
                continue
            if node == dst:
                results.append(path)
                if len(results) >= max_paths:
                    return results
                continue
            for neighbor in adj[node]:
                if neighbor in blocked:
                    continue
                if neighbor in path:
                    continue
                new_path = path + [neighbor]
                queue.append((neighbor, new_path))

        return results

    def solve_mis(conflict_matrix, num_nodes):
        """Find maximum independent set using Bron-Kerbosch on complement graph.
        conflict_matrix[i][j] = True if i and j conflict (cannot both be chosen).
        We want max set with no conflicts = max clique in complement."""
        if num_nodes == 0:
            return []

        # Build complement adjacency: can_coexist[i][j] = not conflict
        can_coexist = [[False] * num_nodes for _ in range(num_nodes)]
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if not conflict_matrix[i][j]:
                    can_coexist[i][j] = True
                    can_coexist[j][i] = True

        best_set = []

        def bron_kerbosch(R, P, X):
            nonlocal best_set
            if not P and not X:
                if len(R) > len(best_set):
                    best_set = list(R)
                return
            # Choose pivot
            pivot_candidates = P | X
            if not pivot_candidates:
                return
            pivot = max(pivot_candidates, key=lambda v: len(P & can_coexist_set[v]))
            # Branch on vertices in P not adjacent to pivot
            candidates = P - can_coexist_set[pivot]
            for v in list(candidates):
                new_R = R | {v}
                new_P = P & can_coexist_set[v]
                new_X = X & can_coexist_set[v]
                bron_kerbosch(new_R, new_P, new_X)
                P.remove(v)
                X.add(v)

        can_coexist_set = [set() for _ in range(num_nodes)]
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and can_coexist[i][j]:
                    can_coexist_set[i].add(j)

        bron_kerbosch(set(), set(range(num_nodes)), set())
        return best_set

    # Strategy: enumerate candidate branch sets of increasing size,
    # for each find all simple paths between pairs, build conflict graph,
    # solve MIS to find max disjoint path set.

    degree = {v: len(adj[v]) for v in range(n)}
    sorted_vertices = sorted(range(n), key=lambda v: degree[v], reverse=True)

    best_result = None
    best_size = 0

    # Candidate pool: top vertices by degree
    candidate_pool = sorted_vertices[:16]

    # Try branch sets of size 2, 3, 4, 5, 6, ...
    # For each size, try combinations from candidate pool
    max_branches_to_try = 8

    for k in range(2, max_branches_to_try + 1):
        if time.time() > deadline - 15:
            break

        # Only try combinations that include at least some high-degree vertices
        # To limit search: first element must be from top k, etc.
        # Actually just try all combinations from candidate_pool
        count_tried = 0
        for combo in combinations(candidate_pool, k):
            if time.time() > deadline - 15:
                break
            count_tried += 1
            if count_tried > 5000:  # Limit per size
                break

            branches = list(combo)
            pairs = list(combinations(range(k), 2))
            num_pairs = len(pairs)

            # For each pair, find candidate paths
            # Paths must avoid all branches as interior
            forbidden_interior = set(branches)
            all_paths_for_pair = []
            feasible = True

            for idx, (i, j) in enumerate(pairs):
                bi, bj = branches[i], branches[j]
                # Find paths of length 1 (direct edge), 2 (one interior), 3 (two interiors)
                paths = find_all_short_paths(bi, bj, forbidden_interior - {bi, bj}, max_len=3, max_paths=5)
                if not paths:
                    feasible = False
                    break
                all_paths_for_pair.append(paths)

            if not feasible:
                continue

            # Build conflict graph
            # Total paths = sum of paths per pair
            total_paths = sum(len(p) for p in all_paths_for_pair)
            if total_paths == 0:
                continue

            # Assign global indices
            path_index = []  # (pair_idx, path) for each global index
            for pair_idx, paths in enumerate(all_paths_for_pair):
                for p in paths:
                    path_index.append((pair_idx, p))

            # Build conflict matrix
            conflict = [[False] * total_paths for _ in range(total_paths)]
            for a in range(total_paths):
                pair_a, path_a = path_index[a]
                interior_a = set(path_a[1:-1])
                for b in range(a + 1, total_paths):
                    pair_b, path_b = path_index[b]
                    # Two paths from same pair conflict (we can only pick one per pair)
                    if pair_a == pair_b:
                        conflict[a][b] = True
                        conflict[b][a] = True
                    else:
                        interior_b = set(path_b[1:-1])
                        if interior_a & interior_b:
                            conflict[a][b] = True
                            conflict[b][a] = True

            # We need exactly one path per pair. This is a constrained MIS:
            # pick exactly one path from each pair, no two picked paths share interior.
            # This is a constraint satisfaction / exact cover problem.
            # Use backtracking with the conflict structure.

            # Backtracking: pick one path per pair
            chosen = [None] * num_pairs
            used_interior = set()
            success = False

            def backtrack(pair_idx):
                nonlocal success, used_interior
                if success:
                    return True
                if time.time() > deadline - 15:
                    return False
                if pair_idx == num_pairs:
                    success = True
                    return True

                bi, bj = branches[pairs[pair_idx][0]], branches[pairs[pair_idx][1]]
                for p in all_paths_for_pair[pair_idx]:
                    interior = set(p[1:-1])
                    if interior & used_interior:
                        continue
                    # Check: does this path's interior conflict with any already-chosen path's interior?
                    # (already checked via used_interior)
                    used_interior.update(interior)
                    chosen[pair_idx] = p
                    if backtrack(pair_idx + 1):
                        return True
                    used_interior.difference_update(interior)
                    chosen[pair_idx] = None

                return False

            if backtrack(0):
                # Found valid set of k branches
                result_paths = [list(chosen[i]) for i in range(num_pairs)]
                if k > best_size:
                    best_size = k
                    best_result = {'branches': list(branches), 'paths': result_paths}

    # Fallback: try to find at least 2 branches
    if best_result is None:
        for u, v in edges:
            best_result = {'branches': [u, v], 'paths': [[u, v]]}
            best_size = 2
            break

    # Ensure types are correct
    branches = [int(b) for b in best_result['branches']]
    paths = [[int(v) for v in p] for p in best_result['paths']]
    best_result['branches'] = branches
    best_result['paths'] = paths

    return best_result

# EVOLVE-BLOCK-END

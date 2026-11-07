"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    degree = [len(adj[v]) for v in range(n)]

    # Bron-Kerbosch for maximum clique
    max_clique = []

    def bronk(R, P_set, X_set):
        nonlocal max_clique
        if time.time() > deadline - 10:
            return
        if not P_set and not X_set:
            if len(R) > len(max_clique):
                max_clique = list(R)
            return
        U = P_set | X_set
        if not U:
            return
        pivot = None
        max_inter = -1
        for u in U:
            inter = len(P_set & adj[u])
            if inter > max_inter:
                max_inter = inter
                pivot = u
        candidates = list(P_set - adj[pivot])
        for v in candidates:
            bronk(R | {v}, P_set & adj[v], X_set & adj[v])
            P_set.discard(v)
            X_set.add(v)
            if time.time() > deadline - 10:
                return

    all_vertices = set(range(n))
    bronk(set(), all_vertices, set())

    if len(max_clique) < 2:
        max_clique = [edges[0][0], edges[0][1]]

    def find_disjoint_paths(branches):
        """
        Find vertex-disjoint interior paths for all pairs of branches.
        Returns a list of paths corresponding to combinations(range(k), 2) or None.
        """
        k = len(branches)
        if k < 2:
            return None
        pairs = list(combinations(range(k), 2))
        branch_set = set(branches)
        
        # Precompute candidate paths for each pair
        # We limit the number of candidates to keep search space manageable
        candidates = {}
        for idx, (i, j) in enumerate(pairs):
            bi, bj = branches[i], branches[j]
            forbidden = branch_set - {bi, bj}
            
            # BFS to find shortest paths
            def bfs_shortest(src, dst, forbidden):
                if src == dst:
                    return [[src]]
                visited = {src}
                queue = deque([(src, [src])])
                shortest_len = float('inf')
                paths = []
                while queue:
                    node, path = queue.popleft()
                    if len(path) > shortest_len:
                        break
                    if node == dst:
                        shortest_len = len(path)
                        paths.append(list(path))
                        continue
                    for nb in adj[node]:
                        if nb in forbidden or nb in visited:
                            continue
                        visited.add(nb)
                        queue.append((nb, path + [nb]))
                # Sort by length
                paths.sort(key=len)
                return paths[:10]

            cands = bfs_shortest(bi, bj, forbidden)
            if not cands:
                return None
            candidates[idx] = cands

        # Order pairs by constraint (fewest candidates first)
        order = sorted(range(len(pairs)), key=lambda x: len(candidates[x]))
        
        assignment = [None] * len(pairs)
        used_interior = set()

        def backtrack(pos):
            if time.time() > deadline - 5:
                return False
            if pos == len(order):
                return True
            
            pair_idx = order[pos]
            for path in candidates[pair_idx]:
                interior = set(path[1:-1])
                if interior & used_interior:
                    continue
                
                assignment[pair_idx] = path
                used_interior.update(interior)
                
                if backtrack(pos + 1):
                    return True
                
                used_interior.difference_update(interior)
                assignment[pair_idx] = None
                
            return False

        if backtrack(0):
            # Reconstruct result in standard combination order
            result_paths = [None] * len(pairs)
            for idx in range(len(pairs)):
                result_paths[idx] = assignment[idx]
            return result_paths
        return None

    # Main Search
    best_branches = list(max_clique)
    best_paths = find_disjoint_paths(best_branches)
    best_size = len(best_branches) if best_paths else 0

    candidate_pool = sorted(range(n), key=lambda v: degree[v], reverse=True)

    # Greedy Extension
    def try_extend(current_branches):
        nonlocal best_branches, best_paths, best_size
        current = list(current_branches)
        paths = find_disjoint_paths(current)
        if not paths:
            return
        
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            for v in candidate_pool:
                if v in current:
                    continue
                new_branches = current + [v]
                new_paths = find_disjoint_paths(new_branches)
                if new_paths:
                    current = new_branches
                    paths = new_paths
                    improved = True
                    if len(current) > best_size:
                        best_branches = list(current)
                        best_paths = paths
                        best_size = len(current)
                    break # Restart loop with new current

    try_extend(best_branches)

    # Random Restarts
    for _ in range(10):
        if time.time() > deadline - 20:
            break
        # Pick a random seed set of 2-4 vertices
        seed_size = random.randint(2, 4)
        seed_vertices = random.sample(candidate_pool[:15], seed_size)
        try_extend(seed_vertices)

    # Local Search (Swap)
    if time.time() < deadline - 25:
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            current = list(best_branches)
            improved = False
            for i in range(len(current)):
                if time.time() > deadline - 10:
                    break
                for v in candidate_pool:
                    if v in current:
                        continue
                    new_branches = current[:i] + [v] + current[i+1:]
                    new_paths = find_disjoint_paths(new_branches)
                    if new_paths and len(new_branches) > len(best_branches):
                        best_branches = new_branches
                        best_paths = new_paths
                        best_size = len(new_branches)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

    # Fallback if no valid paths found
    if best_paths is None:
        # Try to find any valid set of 2
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = [[u, v]]
        best_size = 2

    # Prepare Output
    branches_out = [int(b) for b in best_branches]
    
    # Ensure paths correspond to the order of combinations
    k = len(branches_out)
    pairs = list(combinations(range(k), 2))
    paths_out = []
    
    if best_paths:
        # best_paths is already in the order of combinations
        for path in best_paths:
            paths_out.append([int(x) for x in path])
    else:
        # Should not happen due to fallback, but just in case
        for i, j in pairs:
            paths_out.append([branches_out[i], branches_out[j]])

    return {'branches': branches_out, 'paths': paths_out}

# EVOLVE-BLOCK-END

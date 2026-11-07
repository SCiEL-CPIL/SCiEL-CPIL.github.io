"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations
from collections import deque

sys.setrecursionlimit(10000)

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin before 90s kill

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    adj_list = [sorted(adj[i]) for i in range(n)]

    # Build Bron-Kerbosch for maximal cliques
    def bron_kerbosch(R, P_set, X, cliques):
        if time.time() > deadline:
            return
        if not P_set and not X:
            cliques.append(list(R))
            return
        if P_set or X:
            pivot = max(P_set | X, key=lambda v: len(adj[v] & P_set))
            candidates = P_set - adj[pivot]
        else:
            candidates = P_set
        for v in list(candidates):
            bron_kerbosch(R | {v}, P_set & adj[v], X & adj[v], cliques)
            P_set.remove(v)
            X.add(v)

    cliques = []
    bron_kerbosch(set(), set(range(n)), set(), cliques)
    cliques.sort(key=len, reverse=True)

    # Max-flow based vertex-disjoint path check
    # For a set of branches B, we need vertex-disjoint paths between every pair.
    # We model this as a flow problem: split each vertex v into v_in -> v_out (capacity 1 for non-branches, infinity for branches).
    # For each pair (u,v) in B, we need a path from u_out to v_in.
    # Actually, we need vertex-disjoint paths for ALL pairs simultaneously.
    # This is equivalent to: can we route |B|(|B|-1)/2 units of flow, one for each pair?
    # Better approach: For each pair (u,v), we need a path. All paths must be vertex-disjoint in their interiors.
    # We can check this by: for each non-branch vertex, it can be used by at most 1 path.
    # This is a multi-commodity flow problem, which is NP-hard in general.
    # But for small k, we can use a different approach.

    # Alternative: Use the fact that for k branches, we need C(k,2) internally vertex-disjoint paths.
    # We can check feasibility using max-flow on an augmented network.
    # Create a network where:
    #   - Each vertex v has an in-node and out-node with edge v_in -> v_out (cap 1 if v not in B, cap INF if v in B)
    #   - For each edge (u,v) in G, add edge u_out -> v_in (cap INF) and v_out -> u_in (cap INF)
    #   - For each pair (a,b) in B, we need to send 1 unit from a_out to b_in
    #   - Source connects to each branch's out-node, each branch's in-node connects to sink
    #   - But we need to ensure each pair gets exactly one path...
    # This is complex. Let me use a simpler approach.

    # Simpler: For a given set B of branches, check if there exist internally vertex-disjoint paths for all pairs.
    # Use a max-flow formulation:
    #   - Split each vertex v into v_in and v_out.
    #   - Edge v_in -> v_out has capacity 1 if v not in B, else capacity = len(B) (or some large number).
    #   - For each edge (u,v) in G: edge u_out -> v_in with cap INF, and v_out -> u_in with cap INF.
    #   - Super source S connects to each branch b's b_out with cap 1.
    #   - Each branch b's b_in connects to super sink T with cap 1.
    #   - Wait, this doesn't work directly because we need paths between specific pairs, not just from source to sink.

    # Actually, the correct formulation for "are there internally vertex-disjoint paths between every pair of terminals" is:
    # This is equivalent to asking if the terminals form a "clique" in the topological minor sense.
    # A simpler check: For each pair (u,v), find a path. Then check all paths are internally vertex-disjoint.
    # But finding the paths is the hard part.

    # Let me use a different strategy: greedy with backtracking, but using flow to check feasibility.
    # For a candidate set B, we can check feasibility by:
    #   1. For each non-branch vertex, it can appear in at most one path.
    #   2. We need to assign each pair (u,v) a path.
    # This is like a matching/packing problem.

    # For small k (<= 8 or so), we can use a DFS-based approach with memoization.
    # For larger k, we use a heuristic.

    # Let me implement a flow-based check for a given set of branches:
    # Create a flow network:
    #   Nodes: for each vertex v, create v_in and v_out.
    #   Edges:
    #     - v_in -> v_out: capacity 1 if v not in B, else capacity = k (number of branches)
    #     - For each edge (u,v) in G: u_out -> v_in (cap INF), v_out -> u_in (cap INF)
    #     - Super source S -> b_out for each b in B, cap 1
    #     - b_in -> super sink T for each b in B, cap 1
    #   Wait, this still doesn't capture the pair-specific paths.

    # I think the cleanest approach for small k is:
    # 1. Order the pairs.
    # 2. For each pair, find a path avoiding already-used interior vertices.
    # 3. Use backtracking if a choice leads to dead end.
    # But this is what the original code does, and it's order-sensitive.

    # Better: Use max-flow to check if a set of branches is feasible.
    # The key insight: A set B of k vertices has a clique subdivision iff there exist k(k-1)/2 internally vertex-disjoint paths connecting all pairs.
    # This is equivalent to: in the graph with B removed, can we find vertex-disjoint paths for all pairs?
    # No, the paths can share branch vertices but not interior vertices.

    # Let me try a different max-flow formulation:
    # For each pair (u,v) in B, we need a path from u to v.
    # Interior vertices (not in B) can be used by at most one path.
    # This is a vertex-disjoint paths problem for multiple pairs, which is NP-hard in general.
    # But for k <= 7 or so, we can brute-force or use flow.

    # For k=2: just need one path.
    # For k=3: need 3 paths, (a,b), (a,c), (b,c), internally vertex-disjoint.
    # For k=4: need 6 paths.

    # Let me use a practical approach:
    # 1. Find all maximal cliques.
    # 2. For each clique, try to extend it greedily.
    # 3. For extension, use a flow-based feasibility check.

    # Flow-based feasibility check for a set B:
    # We create a network where we want to route flow from each branch to all other branches.
    # Actually, let me use a simpler heuristic:
    # For a set B, check if for every pair (u,v) in B, there exists a path from u to v in G \ (B \ {u,v}).
    # This is necessary but not sufficient. The sufficient condition requires all paths to be internally vertex-disjoint.
    # For small k, we can check the sufficient condition by:
    #   - For each non-branch vertex, compute how many pairs "need" it (i.e., it's on a shortest path for that pair).
    #   - This is approximate.

    # Let me just use the backtracking approach but with better ordering and pruning.
    # The key improvement: use a flow-based check to prune early.

    # Actually, let me try a completely different approach:
    # Use the fact that the problem is small (n=24).
    # Try all subsets of size k from largest to smallest, and check feasibility.
    # But C(24, 12) is too large.

    # Better: Start with large cliques and extend.

    # Let me implement a robust backtracking solver with good heuristics:
    def check_clique_subdivision(branches):
        """
        Check if the given set of branches forms a valid clique subdivision.
        Returns (True, paths_list) or (False, None).
        """
        if time.time() > deadline:
            return False, None

        k = len(branches)
        if k < 2:
            return True, []

        branch_set = set(branches)
        pairs = list(combinations(sorted(branches), 2))

        # Sort pairs by difficulty: non-adjacent first, then by fewer common neighbors
        def pair_difficulty(pair):
            u, v = pair
            if v in adj[u]:
                return (0, 0)  # Easiest: direct edge
            common = adj[u] & adj[v]
            common -= branch_set
            return (1, -len(common))  # Harder: no direct edge, fewer common neighbors = harder

        pairs.sort(key=pair_difficulty)

        # Backtracking solver
        def solve(idx, used_interior, current_paths):
            if time.time() > deadline:
                return False, None

            if idx == len(pairs):
                return True, current_paths

            u, v = pairs[idx]

            # If u and v are adjacent, path is [u, v]
            if v in adj[u]:
                path = [u, v]
                return solve(idx + 1, used_interior, current_paths + [path])

            # Find candidate paths from u to v avoiding forbidden vertices
            forbidden = used_interior | (branch_set - {u, v})

            # BFS to find shortest path length
            dist = {u: 0}
            parent = {u: None}
            q = deque([u])
            while q:
                curr = q.popleft()
                if curr == v:
                    break
                for nxt in adj[curr]:
                    if nxt in forbidden:
                        continue
                    if nxt not in dist:
                        dist[nxt] = dist[curr] + 1
                        parent[nxt] = curr
                        q.append(nxt)

            if v not in dist:
                return False, None

            # Generate candidate paths using DFS, preferring shorter paths
            candidate_paths = []

            def find_paths_dfs(curr, target, path, visited, max_paths):
                if time.time() > deadline:
                    return []
                if curr == target:
                    return [list(path)]
                if len(candidate_paths) >= max_paths:
                    return []
                results = []
                for nxt in adj[curr]:
                    if nxt in forbidden:
                        continue
                    if nxt in visited:
                        continue
                    path.append(nxt)
                    visited.add(nxt)
                    res = find_paths_dfs(nxt, target, path, visited, max_paths - len(results))
                    results.extend(res)
                    if len(results) >= max_paths - len(candidate_paths):
                        break
                    visited.remove(nxt)
                    path.pop()
                return results

            # Find up to 20 candidate paths
            candidates = find_paths_dfs(u, v, [u], {u}, 20)
            candidates.sort(key=len)

            # Try each candidate path
            for path in candidates:
                internal = set(path[1:-1])
                if not (internal & used_interior):
                    new_used = used_interior | internal
                    res, res_paths = solve(idx + 1, new_used, current_paths + [path])
                    if res:
                        return True, res_paths

            return False, None

        valid, paths = solve(0, set(), [])
        if valid:
            return True, paths
        return False, None

    best_k = 0
    best_branches = None
    best_paths = None

    # Strategy: Start with largest cliques, try to extend
    for clique in cliques:
        if time.time() > deadline - 5:
            break

        current_branches = set(clique)
        current_k = len(current_branches)

        if current_k <= best_k:
            continue

        # Check if the clique itself is valid
        valid, paths = check_clique_subdivision(list(current_branches))
        if valid:
            if current_k > best_k:
                best_k = current_k
                best_branches = list(current_branches)
                best_paths = paths

        # Try to extend greedily
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            candidates = [v for v in range(n) if v not in current_branches]
            # Sort by degree (high degree first)
            candidates.sort(key=lambda x: len(adj[x]), reverse=True)

            for v in candidates:
                if time.time() > deadline - 3:
                    break
                candidate_set = current_branches | {v}
                valid, paths = check_clique_subdivision(list(candidate_set))
                if valid:
                    current_branches = candidate_set
                    current_k += 1
                    if current_k > best_k:
                        best_k = current_k
                        best_branches = list(current_branches)
                        best_paths = paths
                    improved = True
                    break

    # Fallback
    if best_branches is None or len(best_branches) < 2:
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = [[u, v]]

    # Ensure types are correct
    return {'branches': list(best_branches), 'paths': [list(p) for p in best_paths]}

# EVOLVE-BLOCK-END

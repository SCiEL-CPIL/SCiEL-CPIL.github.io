"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random
from itertools import combinations
from collections import deque

sys.setrecursionlimit(10000)

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # 15s margin before 90s kill

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    adj_list = [sorted(adj[i]) for i in range(n)]

    # Max-flow based feasibility oracle
    # For a set of branches B, we check if there exist internally vertex-disjoint paths
    # connecting every pair. We model this as a flow problem on a split graph.
    
    def max_flow_feasibility(branches):
        """
        Check if branches form a valid clique subdivision using max-flow.
        Returns (True, paths_dict) or (False, None).
        paths_dict maps (u,v) -> path for each pair.
        """
        if time.time() > deadline:
            return False, None
        
        k = len(branches)
        if k < 2:
            return True, {}
        
        branch_set = set(branches)
        pairs = list(combinations(sorted(branches), 2))
        num_pairs = len(pairs)
        
        if num_pairs == 0:
            return True, {}
        
        # Build split graph for vertex-disjoint paths
        # Node indexing: v_in = 2*v, v_out = 2*v+1
        # Plus source and sink
        total_nodes = 2 * n + 2
        source = 2 * n
        sink = 2 * n + 1
        
        INF = num_pairs + 10  # Large enough capacity
        
        # Build residual graph as adjacency list with (to, capacity, reverse_edge_index)
        graph = [[] for _ in range(total_nodes)]
        
        def add_edge(u, v, cap):
            graph[u].append([v, cap, len(graph[v])])
            graph[v].append([u, 0, len(graph[u]) - 1])
        
        # Add vertex split edges
        for v in range(n):
            v_in = 2 * v
            v_out = 2 * v + 1
            if v in branch_set:
                add_edge(v_in, v_out, INF)  # Branch vertices can be used multiple times
            else:
                add_edge(v_in, v_out, 1)  # Non-branch vertices can be used at most once
        
        # Add graph edges (bidirectional)
        for u, v in edges:
            u_out = 2 * u + 1
            v_in = 2 * v
            v_out = 2 * v + 1
            u_in = 2 * u
            add_edge(u_out, v_in, INF)
            add_edge(v_out, u_in, INF)
        
        # For each pair (a,b), we need a path from a to b.
        # We'll use a multi-source, multi-sink approach:
        # Source connects to each branch's out-node with capacity 1 per pair involving that branch
        # Each branch's in-node connects to sink with capacity 1 per pair involving that branch
        
        # Actually, let's use a different formulation:
        # For each pair (a,b), create a separate "commodity" but that's multi-commodity flow (NP-hard).
        # Instead, use the fact that we need vertex-disjoint paths for ALL pairs simultaneously.
        
        # Alternative: For each non-branch vertex, it can be in at most one path.
        # We can check this by: remove all non-branch vertices and check connectivity,
        # but that's not quite right either.
        
        # Let's use a simpler but effective approach:
        # For each pair (u,v), find a path avoiding other branches and already-used vertices.
        # Use greedy with backtracking, but order pairs by difficulty.
        
        # First, check if all pairs are connected in G \ (B \ {u,v})
        for u, v in pairs:
            # BFS from u to v avoiding other branches
            visited = set()
            queue = deque([u])
            visited.add(u)
            found = False
            while queue:
                curr = queue.popleft()
                if curr == v:
                    found = True
                    break
                for nxt in adj[curr]:
                    if nxt in branch_set and nxt not in (u, v):
                        continue
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
            if not found:
                return False, None
        
        # Now try to find actual paths using backtracking
        # Order pairs: adjacent pairs first (trivial), then by fewest available interior vertices
        def pair_order(pair):
            u, v = pair
            if v in adj[u]:
                return (0, 0)
            # Count available interior vertices (not in branch_set, not u or v)
            avail = len(adj[u] & adj[v] - branch_set)
            return (1, avail)
        
        pairs.sort(key=pair_order)
        
        # Backtracking to find paths
        def find_paths_backtrack(idx, used_vertices, paths_found):
            if time.time() > deadline:
                return None
            
            if idx == len(pairs):
                return paths_found
            
            u, v = pairs[idx]
            
            # If u and v are adjacent, path is just [u, v]
            if v in adj[u]:
                paths_found.append([u, v])
                result = find_paths_backtrack(idx + 1, used_vertices, paths_found)
                if result is not None:
                    return result
                paths_found.pop()
                return None
            
            # Find paths from u to v avoiding used_vertices and other branches
            forbidden = used_vertices | (branch_set - {u, v})
            
            # DFS to find candidate paths
            candidates = []
            
            def dfs_path(curr, target, path, visited):
                if time.time() > deadline:
                    return
                if len(candidates) >= 15:
                    return
                if curr == target:
                    candidates.append(list(path))
                    return
                for nxt in adj[curr]:
                    if nxt in forbidden or nxt in visited:
                        continue
                    path.append(nxt)
                    visited.add(nxt)
                    dfs_path(nxt, target, path, visited)
                    visited.remove(nxt)
                    path.pop()
            
            dfs_path(u, v, [u], {u})
            
            # Sort by length (prefer shorter paths)
            candidates.sort(key=len)
            
            for path in candidates:
                interior = set(path[1:-1])
                if not (interior & used_vertices):
                    new_used = used_vertices | interior
                    paths_found.append(path)
                    result = find_paths_backtrack(idx + 1, new_used, paths_found)
                    if result is not None:
                        return result
                    paths_found.pop()
            
            return None
        
        paths = find_paths_backtrack(0, set(), [])
        if paths is not None:
            return True, paths
        return False, None

    # Bron-Kerbosch for maximal cliques
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

    best_k = 0
    best_branches = None
    best_paths = None

    # Strategy: Try cliques and their extensions
    # First pass: check all cliques
    for clique in cliques:
        if time.time() > deadline - 10:
            break
        
        current_branches = set(clique)
        current_k = len(current_branches)
        
        if current_k <= best_k:
            continue
        
        valid, paths = max_flow_feasibility(list(current_branches))
        if valid:
            if current_k > best_k:
                best_k = current_k
                best_branches = list(current_branches)
                best_paths = paths

    # Second pass: try to extend the best found so far and other large cliques
    for clique in cliques:
        if time.time() > deadline - 5:
            break
        
        current_branches = set(clique)
        current_k = len(current_branches)
        
        if current_k + 1 <= best_k:
            continue
        
        # Greedy extension
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            candidates = [v for v in range(n) if v not in current_branches]
            # Sort by: number of connections to current branches (higher is better)
            candidates.sort(key=lambda x: len(adj[x] & current_branches), reverse=True)
            
            for v in candidates:
                if time.time() > deadline - 3:
                    break
                candidate_set = current_branches | {v}
                valid, paths = max_flow_feasibility(list(candidate_set))
                if valid:
                    current_branches = candidate_set
                    current_k += 1
                    if current_k > best_k:
                        best_k = current_k
                        best_branches = list(current_branches)
                        best_paths = paths
                    improved = True
                    break

    # Third pass: try random subsets for diversity
    if time.time() < deadline - 5:
        random.seed(42)
        for _ in range(50):
            if time.time() > deadline - 3:
                break
            # Try random subsets of size best_k + 1
            target_k = min(best_k + 1, n)
            subset = random.sample(range(n), target_k)
            valid, paths = max_flow_feasibility(subset)
            if valid:
                if target_k > best_k:
                    best_k = target_k
                    best_branches = subset
                    best_paths = paths

    # Fallback
    if best_branches is None or len(best_branches) < 2:
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = [[u, v]]

    # Ensure types are correct: convert to Python int and list
    branches_out = [int(b) for b in best_branches]
    paths_out = [[int(v) for v in p] for p in best_paths]

    return {'branches': branches_out, 'paths': paths_out}

# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import sys
from collections import defaultdict

def construct():
    # Instance generator copied from verifier
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def solve_graph(edges):
        if not edges:
            return []
        
        # Build adjacency list
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Calculate max degree
        max_deg = max(len(nei) for nei in adj.values())
        target_k = (max_deg + 1) // 2
        
        # Backtracking search for path cover of size <= target_k
        # If not found, try target_k + 1, etc., up to a limit or fall back to greedy
        # Given the small size, we can afford a limited backtracking.
        
        # State: unused_edges (set), paths (list of tuples of vertices)
        # To speed up, we always pick an edge from unused_edges.
        
        def backtrack(unused, paths, k_limit, node_limit=100000):
            if not unused:
                return paths[:]
            if len(paths) > k_limit:
                return None
            
            # Heuristic: pick edge with highest degree sum of endpoints in unused graph?
            # For simplicity, pick the first one, or one with highest degree in original graph
            # Let's pick an edge arbitrarily but deterministic for consistency in backtracking logic
            # Actually, picking an edge that is "hard" to place is better.
            # Let's pick the edge whose endpoints have the highest degree in the *remaining* graph.
            
            # Build remaining adjacency counts
            rem_deg = defaultdict(int)
            for u, v in unused:
                rem_deg[u] += 1
                rem_deg[v] += 1
                
            # Pick edge with max (rem_deg[u] + rem_deg[v])
            best_edge = None
            max_score = -1
            for e in unused:
                u, v = e
                score = rem_deg[u] + rem_deg[v]
                if score > max_score:
                    max_score = score
                    best_edge = e
                    
            u, v = best_edge
            new_unused = set(unused)
            new_unused.remove(best_edge)
            
            # Option 1: Start a new path
            if len(paths) < k_limit:
                res = backtrack(new_unused, paths + [tuple(sorted((u, v)))], k_limit, node_limit)
                if res:
                    return res
            
            # Option 2: Extend an existing path
            # For each path, check if u or v is an endpoint
            for i, path in enumerate(paths):
                # Path is a tuple of vertices. Endpoints are path[0] and path[-1]
                start, end = path[0], path[-1]
                
                # Can we attach (u,v) to start?
                # If u == start, we can prepend v if v is not in path.
                # If v == start, we can prepend u if u is not in path.
                # Similarly for end.
                
                # Case 2a: Extend at start with u
                if u == start and v not in path:
                    new_path = (v,) + path
                    new_paths = paths.copy()
                    new_paths[i] = new_path
                    res = backtrack(new_unused, new_paths, k_limit, node_limit)
                    if res:
                        return res
                
                # Case 2b: Extend at start with v
                if v == start and u not in path:
                    new_path = (u,) + path
                    new_paths = paths.copy()
                    new_paths[i] = new_path
                    res = backtrack(new_unused, new_paths, k_limit, node_limit)
                    if res:
                        return res
                
                # Case 2c: Extend at end with u
                if u == end and v not in path:
                    new_path = path + (v,)
                    new_paths = paths.copy()
                    new_paths[i] = new_path
                    res = backtrack(new_unused, new_paths, k_limit, node_limit)
                    if res:
                        return res
                
                # Case 2d: Extend at end with v
                if v == end and u not in path:
                    new_path = path + (u,)
                    new_paths = paths.copy()
                    new_paths[i] = new_path
                    res = backtrack(new_unused, new_paths, k_limit, node_limit)
                    if res:
                        return res
            
            return None

        # Try increasing k until we find a solution or hit a limit
        # Gallai's conjecture suggests k = ceil(Delta/2) is enough.
        # We try up to k = max_deg (trivial bound)
        
        best_sol = None
        for k in range(target_k, max_deg + 1):
            # Add a simple node limit to prevent infinite loops on hard instances
            # For n=30, 100k nodes should be fast enough in Python if pruned well.
            # However, Python recursion is slow. Let's use a simpler greedy if backtracking is too slow.
            # We'll run backtracking with a tight limit.
            sol = backtrack(set(edges), [], k, node_limit=50000)
            if sol:
                best_sol = sol
                break
        
        if not best_sol:
            # Fallback: Greedy Longest Path
            # This ensures we always return a valid answer
            unused = set(edges)
            paths = []
            while unused:
                # Find longest path in remaining graph
                # Simple DFS for longest path
                # Since graph is small, we can just do a greedy walk
                # Start from a random vertex, follow unvisited edges
                # To get "longest", we try multiple starts
                
                best_path = []
                # Try starting from each vertex
                # Optimization: only try vertices with remaining degree > 0
                start_candidates = [v for v in adj if any(u in adj[v] for u,v in unused) or any(v in adj[u] for u,v in unused)]
                # Actually, just iterate all vertices
                for start in range(len(adj)):
                    if not any(start in e for e in unused):
                        continue
                    # DFS to find longest path starting at 'start'
                    # Path is a list of vertices
                    stack = [(start, [start])]
                    max_len = 0
                    max_path = []
                    # Simple DFS
                    def dfs_greedy(curr, path, visited_edges):
                        nonlocal max_len, max_path
                        if len(path) > max_len:
                            max_len = len(path)
                            max_path = path[:]
                        for nei in adj[curr]:
                            e = tuple(sorted((curr, nei)))
                            if e in unused and e not in visited_edges:
                                dfs_greedy(nei, path + [nei], visited_edges | {e})
                    
                    # To avoid exponential blowup, limit depth or use greedy
                    # Greedy walk:
                    curr = start
                    curr_path = [curr]
                    visited_e = set()
                    while True:
                        nexts = []
                        for nei in adj[curr]:
                            e = tuple(sorted((curr, nei)))
                            if e in unused and e not in visited_e:
                                nexts.append(nei)
                        if not nexts:
                            break
                        # Pick neighbor with highest degree in remaining graph
                        # For simplicity, pick first
                        next_v = nexts[0]
                        e = tuple(sorted((curr, next_v)))
                        visited_e.add(e)
                        curr_path.append(next_v)
                        curr = next_v
                    
                    if len(curr_path) > len(best_path):
                        best_path = curr_path
                
                # Convert best_path to edges
                path_edges = set()
                for i in range(len(best_path) - 1):
                    path_edges.add(tuple(sorted((best_path[i], best_path[i+1]))))
                
                if not path_edges:
                    # Should not happen if unused is not empty, but safety
                    break
                    
                paths.append(best_path)
                unused -= path_edges
                
            best_sol = paths
            
        return best_sol

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = solve_graph(edges)
        results.append(paths)
        
    return results
# EVOLVE-BLOCK-END

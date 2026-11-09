"""Improved approach: Backtracking Disjoint Path Packing."""
# EVOLVE-BLOCK-START
import itertools
import random
import sys
from collections import deque

# Increase recursion limit for backtracking
sys.setrecursionlimit(10000)

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    n = int(P['n'])
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute degrees for sorting
    degrees = [len(adj[i]) for i in range(n)]
    sorted_vertices = sorted(range(n), key=lambda x: degrees[x], reverse=True)

    # Helper to find multiple simple paths between u and v avoiding blocked vertices
    # Returns a list of paths, sorted by length
    def find_paths(u, v, blocked, max_paths=5, max_len=10):
        if u == v:
            return [[u]]
        if not adj[u] or not adj[v]:
            return []
        
        paths = []
        # BFS to find shortest paths first
        # We use a standard BFS but we want to collect a few paths, not just the first one.
        # To ensure we get diverse paths, we can do a DFS with depth limit if BFS is too restrictive,
        # or just collect all paths of length <= max_len.
        # Given n=24, max_len=10 is safe.
        
        # Optimization: If u and v are adjacent, [u, v] is always a valid path if v not in blocked (v is dest, so ok)
        # Wait, blocked contains interiors. u and v are endpoints.
        
        # Let's do a DFS that finds up to max_paths simple paths with length <= max_len
        def dfs(current, target, path, visited, blocked):
            if len(paths) >= max_paths:
                return
            
            if current == target:
                if len(path) <= max_len:
                    paths.append(list(path))
                return
            
            # Heuristic: if current path length is already >= max_len, stop
            if len(path) > max_len:
                return
                
            for neighbor in adj[current]:
                if neighbor in visited:
                    continue
                if neighbor in blocked and neighbor != target:
                    continue
                
                visited.add(neighbor)
                path.append(neighbor)
                dfs(neighbor, target, path, visited, blocked)
                path.pop()
                visited.remove(neighbor)
                
                if len(paths) >= max_paths:
                    return
        
        # Start DFS
        # To get shorter paths first, we can sort neighbors by distance to target?
        # Or just rely on DFS order. To be safe, let's sort neighbors by degree or just leave it.
        # Actually, BFS is better for shortest. Let's do BFS to get shortest path, then DFS for others?
        # Simpler: Just DFS with a limit on depth. The first few found might not be shortest.
        # Let's use BFS to find the shortest path distance, then DFS to find paths of that length and slightly longer.
        
        # Even simpler: Just collect paths via DFS, then sort by length.
        # To bias towards shorter paths, we can order neighbors.
        
        # Let's try a modified BFS that collects paths.
        # But BFS only gives shortest.
        
        # Let's stick to DFS with a small depth limit and max_paths limit.
        # To make it find shorter paths first, we can do a BFS to get dist, then DFS only exploring nodes that can still reach target within max_len.
        
        # Precompute dist from v to all nodes? No, blocked changes.
        
        # Let's just use DFS and sort results.
        # To improve speed, we prune if len(path) > max_len.
        
        visited = {u}
        path = [u]
        dfs(u, v, path, visited, blocked)
        
        # Sort by length
        paths.sort(key=len)
        return paths

    # Global state for backtracking
    best_k = 0
    best_result = None

    # Try to find largest k
    # Max possible k is limited by n.
    # We try k from min(n, 10) down to 2. Why 10? C(24, 10) is huge.
    # We need to be smart about which combinations to try.
    
    # Strategy:
    # 1. Try top-degree vertices first.
    # 2. If we find a solution for k, we can stop if we are looking for max? 
    #    No, we want the MAX k. So we iterate k from high to low.
    #    For a given k, we try combinations. If we find ANY valid configuration for k, 
    #    we can return it immediately because we are iterating k downwards.
    
    max_k_to_try = min(n, 12) # Limit to 12 to keep search space manageable with backtracking
    
    for k in range(max_k_to_try, 1, -1):
        # Generate candidate sets of size k
        # We focus on high-degree vertices.
        # Pool: top k+5 vertices?
        # If k is large, pool must be large.
        # Let's use the top min(n, k+10) vertices as a pool.
        pool_size = min(n, k + 10)
        pool = sorted_vertices[:pool_size]
        
        # If k is close to n, we have few combinations.
        # If k is small, we have many.
        # We will iterate through combinations of the pool.
        # To avoid explosion, we might need to limit the number of combinations tried per k.
        # But if we find a solution, we break.
        
        # Shuffle pool? No, high degree is a good heuristic.
        
        # We can also add some random combinations if the pool is too small or deterministic search fails?
        # For now, let's just iterate combinations of the pool.
        
        # Optimization: If k is large, C(pool_size, k) is small.
        # If k is small, C(pool_size, k) is large.
        # Let's limit the number of combinations we try for small k if it's too big?
        # No, we need to find the max k. If max k is small (e.g. 5), we might have to search a lot.
        # But usually, clique subdivisions in such graphs are decently sized.
        
        # Let's add a time/iteration guard? No, just efficient backtracking.
        
        for comb in itertools.combinations(pool, k):
            branches = list(comb)
            branches_set = set(branches)
            pairs = list(itertools.combinations(branches, 2))
            
            # Heuristic for pair ordering:
            # Pairs with fewer available paths (more constrained) should be solved first.
            # Approximation: Sort by shortest path length in the original graph?
            # Or just fixed order?
            # Let's sort by the sum of degrees of endpoints (high degree = more options? No, low degree = more constrained?)
            # Actually, pairs that are "far" apart might be easier to separate?
            # Let's try sorting by distance in the original graph (shortest path).
            # Shortest path = tightest constraint?
            
            # Precompute distances for heuristic
            # We already have adj. Let's just use a simple metric: number of common neighbors?
            # Or just fixed order for now to save time in precomputation per combination.
            
            # Let's sort pairs by (degree[u] + degree[v]) descending? 
            # High degree vertices have more options, so maybe they are "easier" to route?
            # Or low degree are "harder"?
            # Let's try sorting by min(degree[u], degree[v]) ascending (hardest first).
            
            pairs_sorted = sorted(pairs, key=lambda p: min(degrees[p[0]], degrees[p[1]]))
            
            # Backtracking Search
            # State:
            # - idx: index of next pair to assign
            # - used_interiors: set of vertices used as interiors
            # - paths: dict mapping (u, v) to path
            
            def solve(idx, used_interiors, paths):
                if idx == len(pairs_sorted):
                    return True
                
                u, v = pairs_sorted[idx]
                
                # Blocked vertices: all branches except u, v + used_interiors
                blocked = branches_set - {u, v}
                blocked.update(used_interiors)
                
                # Find candidate paths
                # We want paths that are disjoint from used_interiors.
                # find_paths handles this via blocked.
                
                # To speed up, we can filter candidates more aggressively.
                candidates = find_paths(u, v, blocked, max_paths=10, max_len=8)
                
                # If no candidates, fail
                if not candidates:
                    return False
                
                # Try each candidate path
                # Heuristic: Try shorter paths first? Or more "central" ones?
                # Shorter paths use fewer vertices, leaving more room for others.
                # So shorter is generally better.
                
                for path in candidates:
                    interiors = path[1:-1]
                    
                    # Check if interiors are disjoint from used_interiors
                    # find_paths already ensures this by passing blocked which includes used_interiors.
                    # But wait, blocked is a set. find_paths checks `if neighbor in blocked`.
                    # So interiors are guaranteed not to be in blocked.
                    
                    # However, we must ensure that the interiors don't contain any branches?
                    # find_paths blocks branches (except u, v). So interiors are non-branches.
                    
                    # Add to used
                    new_used = used_interiors | set(interiors)
                    new_paths = dict(paths)
                    new_paths[(u, v)] = path
                    new_paths[(v, u)] = path
                    
                    if solve(idx + 1, new_used, new_paths):
                        return True
                    
                return False

            # Run solver
            # To handle potential exponential blowup, we might need a limit on recursion or nodes explored?
            # With k <= 12, number of pairs is ~66.
            # This might be too slow if backtracking is deep.
            # Let's add a simple node limit or just trust it for n=24.
            # Actually, 66 pairs is a lot.
            # We should only try this for smaller k if the search space is small.
            # Or optimize the path finding.
            
            # If k is large, the number of pairs is large.
            # Maybe we should only do full backtracking for k <= 8?
            # For k > 8, we can use a greedy heuristic?
            # But the task asks to maximize.
            
            # Let's try. If it's too slow, we can reduce max_k_to_try.
            
            if solve(0, set(), {}):
                # Success
                result_paths = []
                for u, v in pairs:
                    result_paths.append(paths[(u, v)])
                
                best_result = {
                    'branches': [int(b) for b in branches],
                    'paths': result_paths
                }
                best_k = k
                # Since we are iterating k downwards, this is the max k for this pool.
                # But what if a different pool with same k has a solution?
                # We found one, so k is achievable.
                # We can break out of the k loop and return.
                return best_result

        # If no combination in this pool worked for k, try smaller k?
        # Or try a larger pool?
        # If the pool was too small, maybe the branches are outside the top-degree set.
        # Let's try a random pool for the same k if deterministic pool fails?
        # For simplicity, we just drop k.
        # But this might miss a solution with a slightly lower degree vertex.
        
        # Let's add a random sampling step for the same k if the deterministic pool fails.
        # Generate 100 random combinations of size k from the whole graph.
        
        for _ in range(100):
            branches = random.sample(range(n), k)
            branches_set = set(branches)
            pairs = list(itertools.combinations(branches, 2))
            pairs_sorted = sorted(pairs, key=lambda p: min(degrees[p[0]], degrees[p[1]]))
            
            def solve_random(idx, used_interiors, paths):
                if idx == len(pairs_sorted):
                    return True
                
                u, v = pairs_sorted[idx]
                blocked = branches_set - {u, v}
                blocked.update(used_interiors)
                
                candidates = find_paths(u, v, blocked, max_paths=5, max_len=8)
                if not candidates:
                    return False
                
                for path in candidates:
                    interiors = path[1:-1]
                    new_used = used_interiors | set(interiors)
                    new_paths = dict(paths)
                    new_paths[(u, v)] = path
                    new_paths[(v, u)] = path
                    
                    if solve_random(idx + 1, new_used, new_paths):
                        return True
                return False

            if solve_random(0, set(), {}):
                result_paths = []
                for u, v in pairs:
                    result_paths.append(paths[(u, v)])
                
                best_result = {
                    'branches': [int(b) for b in branches],
                    'paths': result_paths
                }
                best_k = k
                return best_result

    # Fallback if no solution found for k >= 2
    # This shouldn't happen if the graph is connected and has edges.
    if edges:
        u, v = edges[0]
        return {
            'branches': [int(u), int(v)],
            'paths': [[int(u), int(v)]]
        }
    else:
        return {
            'branches': [0, 1],
            'paths': [[0, 1]]
        }
# EVOLVE-BLOCK-END

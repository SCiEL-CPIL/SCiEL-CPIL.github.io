"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Calculate degrees
    degrees = [len(adj[i]) for i in range(n)]
    
    # Candidate vertices: top by degree
    candidate_indices = sorted(range(n), key=lambda i: degrees[i], reverse=True)
    
    # Global deadline: 85 seconds (safety margin from 90s)
    global_deadline = time.time() + 85.0
    
    def find_paths_for_subset(branches, time_budget):
        """
        Check if the given set of branches supports a clique subdivision.
        Returns list of paths if found, else None.
        """
        k = len(branches)
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        
        # Order pairs by how constrained they are: pairs where both endpoints have
        # fewer available neighbors (excluding branch_set and considering other pairs)
        # Simpler heuristic: order by sum of degrees of endpoints (ascending = more constrained first)
        pairs_sorted = sorted(pairs, key=lambda p: degrees[p[0]] + degrees[p[1]])
        
        def get_candidate_paths(u, v, forbidden, max_paths=50):
            """
            Find simple paths from u to v avoiding forbidden internal nodes.
            Returns a list of paths (list of vertices), sorted by length.
            """
            paths = []
            
            def dfs(current, target, path, visited):
                if len(paths) >= max_paths:
                    return
                if current == target:
                    if len(path) > 1:
                        paths.append(list(path))
                    return
                
                for neighbor in adj[current]:
                    if neighbor == target:
                        dfs(neighbor, target, path + [neighbor], visited | {neighbor})
                    elif neighbor not in visited and neighbor not in forbidden and neighbor not in branch_set:
                        dfs(neighbor, target, path + [neighbor], visited | {neighbor})

            dfs(u, v, [u], {u})
            paths.sort(key=len)
            return paths

        best_result = None
        deadline = time.time() + time_budget
        
        def solve(idx, used_internal, current_paths):
            nonlocal best_result
            if time.time() > deadline:
                return False
            
            if idx == len(pairs_sorted):
                best_result = current_paths
                return True
            
            u, v = pairs_sorted[idx]
            forbidden = used_internal | (branch_set - {u, v})
            
            candidates = get_candidate_paths(u, v, forbidden, max_paths=50)
            
            if not candidates:
                return False
            
            for path in candidates:
                internal_nodes = set(path[1:-1])
                new_used_internal = used_internal | internal_nodes
                new_paths = current_paths + [(u, v, path)]
                
                if solve(idx + 1, new_used_internal, new_paths):
                    return True
            
            return False
        
        if solve(0, set(), []):
            return [path for u, v, path in best_result]
        
        return None

    # Try k from 8 down to 3
    # Use larger pools for smaller k, tighter for larger k
    best_solution = None
    
    for k in [8, 7, 6, 5, 4, 3]:
        # Allocate time budget based on remaining time and number of subsets to try
        remaining_time = global_deadline - time.time()
        if remaining_time < 1.0:
            break
        
        # Pool sizes: for k=8 use top 14, k=7 top 14, k=6 top 15, k=5 top 18, k=4 top 22, k=3 top 24
        pool_size = {8: 14, 7: 14, 6: 15, 5: 18, 4: 22, 3: 24}[k]
        candidates = candidate_indices[:pool_size]
        
        # Estimate number of subsets
        from math import comb
        num_subsets = comb(pool_size, k)
        
        # Time per subset: allocate remaining time across subsets, with a cap
        time_per_subset = min(remaining_time / max(num_subsets, 1), 2.0)
        # Ensure at least 0.1s per subset
        time_per_subset = max(time_per_subset, 0.1)
        
        found = False
        for subset in combinations(candidates, k):
            if time.time() > global_deadline - 1.0:
                break
            branches = list(subset)
            paths = find_paths_for_subset(branches, time_per_subset)
            if paths is not None:
                best_solution = {
                    'branches': branches,
                    'paths': paths
                }
                found = True
                break
        
        if found:
            break
            
    if best_solution is None:
        # Fallback: find any 2 branches with a path
        u, v = edges[0]
        best_solution = {
            'branches': [u, v],
            'paths': [[u, v]]
        }
    
    return best_solution

# EVOLVE-BLOCK-END

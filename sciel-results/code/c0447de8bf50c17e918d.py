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
    # We want to try larger k first.
    # Max possible branches is limited by n.
    # Let's try k from 6 down to 3.
    
    # To speed up, restrict candidates to top 15 vertices by degree.
    # If a solution exists with a low-degree vertex, it's less likely to be part of a large clique subdivision.
    candidate_indices = sorted(range(n), key=lambda i: degrees[i], reverse=True)
    
    def find_paths_for_subset(branches):
        """
        Check if the given set of branches supports a clique subdivision.
        Returns (paths_dict) if found, else None.
        """
        k = len(branches)
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        
        # State for backtracking
        # We will assign paths to pairs one by one.
        # used_internal: set of vertices used as internal nodes in assigned paths.
        
        # Heuristic: order pairs. Maybe pairs involving high-degree vertices first?
        # Or just fixed order.
        
        # To find a path between u and v avoiding forbidden nodes:
        # We use BFS to find shortest paths.
        # We might need to find multiple paths if backtracking fails.
        # For simplicity, let's find all simple paths up to a certain length?
        # Or just BFS to find one, and if it fails, try to find another?
        # Given small n, we can find all simple paths of length <= 4 or 5.
        
        def get_candidate_paths(u, v, forbidden):
            """
            Find simple paths from u to v avoiding forbidden internal nodes.
            Returns a list of paths (list of vertices).
            """
            paths = []
            # BFS/DFS
            # Since we want disjointness, shorter paths are generally better as they use fewer resources.
            # Let's do a BFS that expands layer by layer, but we need to avoid cycles and forbidden nodes.
            
            # Simple DFS with pruning
            def dfs(current, target, path, visited):
                if len(paths) >= 20: # Limit number of paths to consider
                    return
                if current == target:
                    if len(path) > 1: # Must have at least one edge
                        paths.append(list(path))
                    return
                
                for neighbor in adj[current]:
                    if neighbor == target:
                        # Can go to target directly
                        dfs(neighbor, target, path + [neighbor], visited | {neighbor})
                    elif neighbor not in visited and neighbor not in forbidden and neighbor not in branch_set:
                        # Note: neighbor must not be a branch unless it is the target
                        # But we already handled target.
                        # Also, we must not visit other branches.
                        dfs(neighbor, target, path + [neighbor], visited | {neighbor})

            dfs(u, v, [u], {u})
            
            # Sort by length
            paths.sort(key=len)
            return paths

        # Backtracking search
        # We maintain a list of assigned paths.
        # We pick a pair, find candidate paths, and try them.
        
        # To improve efficiency, we can use a constraint propagation or simply order the pairs.
        # Let's try a simple recursive backtracking.
        
        best_result = None
        deadline = time.time() + 5.0 # 5 seconds per subset check? 
        # Actually, we have 90s total. 
        # Number of subsets: 
        # k=6: C(15,6) = 5005. 5005 * 5s is too much.
        # We need a faster check or fewer subsets.
        
        # Let's restrict candidates to top 10 for k=6. C(10,6)=210.
        # Top 12 for k=5. C(12,5)=792.
        # Top 15 for k=4. C(15,4)=1365.
        
        # Let's adjust candidate pool based on k.
        
        # The function is called with a specific branch set.
        # We need to be fast.
        
        # Optimization: Pre-compute all simple paths between all pairs of vertices up to length 4?
        # There are 24*23/2 = 276 pairs.
        # Paths of length 1, 2, 3, 4.
        # This might be feasible.
        
        # Let's stick to on-the-fly but efficient.
        
        # Recursive solver
        def solve(idx, used_internal, current_paths):
            nonlocal best_result
            if time.time() > deadline:
                return False
            
            if idx == len(pairs):
                best_result = current_paths
                return True
            
            u, v = pairs[idx]
            
            # Get candidate paths for this pair
            # Forbidden: used_internal + (branch_set - {u, v})
            forbidden = used_internal | (branch_set - {u, v})
            
            candidates = get_candidate_paths(u, v, forbidden)
            
            # Try candidates
            for path in candidates:
                # Check if path is valid (it should be by construction)
                # Internal nodes of path
                internal_nodes = set(path[1:-1])
                
                # Check disjointness (already ensured by forbidden set in get_candidate_paths? 
                # Yes, we passed forbidden which includes used_internal.
                # But we need to ensure internal nodes don't overlap with EACH OTHER in current_paths?
                # No, `used_internal` accumulates all used internal nodes from previous steps.
                # So if we pass `used_internal` to get_candidate_paths, the new path will avoid them.
                
                new_used_internal = used_internal | internal_nodes
                new_paths = current_paths + [(u, v, path)]
                
                if solve(idx + 1, new_used_internal, new_paths):
                    return True
            
            return False
        
        # Order pairs to find conflicts early?
        # Pairs sharing a vertex might be more constrained?
        # Let's just use the default order from combinations.
        
        if solve(0, set(), []):
            # Convert best_result to the required format
            # best_result is a list of (u, v, path)
            paths_dict = {}
            for u, v, path in best_result:
                # Ensure consistent key order? The task says "unordered pair".
                # The output format is 'paths': list of paths.
                # "one simple path for each unordered pair of branches"
                # So we just return the list of paths.
                pass
            
            # The return value of construct is a dict with 'branches' and 'paths'.
            # 'paths' is a list of lists.
            return [path for u, v, path in best_result]
        
        return None

    # Try k from 6 down to 3
    # We will iterate through subsets of candidates.
    
    # Define candidate pools
    # k=6: top 12
    # k=5: top 15
    # k=4: top 20
    # k=3: top 24
    
    best_solution = None
    
    for k in [6, 5, 4, 3]:
        pool_size = 12 if k == 6 else (15 if k == 5 else (20 if k == 4 else 24))
        candidates = candidate_indices[:pool_size]
        
        for subset in combinations(candidates, k):
            branches = list(subset)
            paths = find_paths_for_subset(branches)
            if paths is not None:
                best_solution = {
                    'branches': branches,
                    'paths': paths
                }
                break # Found a solution for this k, move to next (smaller) k? No, we want MAX k.
                # Since we iterate k from 6 down, the first found is the max.
        
        if best_solution is not None:
            break
            
    if best_solution is None:
        # Fallback: find any 2 branches with a path
        # Or find a triangle
        # The seed found a triangle.
        # Let's just return the first edge as 2 branches.
        u, v = edges[0]
        best_solution = {
            'branches': [u, v],
            'paths': [[u, v]]
        }
        
    return best_solution

# EVOLVE-BLOCK-END

import time
import random
from collections import defaultdict

# Pinned graph data
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find connected components
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if not visited[nei]:
                        visited[nei] = True
                        stack.append(nei)
            components.append(comp)
    
    best_path = []
    
    # Process each component
    for comp in components:
        if time.time() > deadline:
            break
            
        k = len(comp)
        if k == 0:
            continue
        
        # If component is small, use Exact DP
        if k <= 25:
            # Map comp nodes to 0..k-1
            node_map = {node: i for i, node in enumerate(comp)}
            inv_map = [0] * k
            for node, idx in node_map.items():
                inv_map[idx] = node
                
            # Build local adjacency
            local_adj = [set() for _ in range(k)]
            for u in comp:
                u_idx = node_map[u]
                for v in adj[u]:
                    if v in node_map:
                        local_adj[u_idx].add(node_map[v])
                        
            # DP: dp[mask][last] = 1 if path exists, 0 otherwise
            # To save memory and time, we can use a dict or just bitsets if k is small.
            # k <= 25, so 2^25 * 25 is too big for standard array. 
            # But we can do DFS with memoization on (mask, last).
            # Or, since k is small, we can just do recursive backtracking with pruning.
            
            memo = {}
            def dfs_exact(u, mask):
                key = (u, mask)
                if key in memo:
                    return memo[key]
                
                best_len = 1
                for v in local_adj[u]:
                    if not (mask & (1 << v)):
                        res = dfs_exact(v, mask | (1 << v))
                        if res + 1 > best_len:
                            best_len = res + 1
                memo[key] = best_len
                return best_len
            
            # We need the actual path, not just length. 
            # Re-run to reconstruct or store path in memo.
            # Given time constraints and k<=25, a simple DFS with backtracking is fast enough.
            best_local = []
            def backtrack(u, mask, path):
                nonlocal best_local
                if len(path) > len(best_local):
                    best_local = path[:]
                
                for v in local_adj[u]:
                    if not (mask & (1 << v)):
                        path.append(v)
                        backtrack(v, mask | (1 << v), path)
                        path.pop()
                        
            for i in range(k):
                backtrack(i, 1 << i, [i])
                
            # Convert back to original node IDs
            path_original = [inv_map[idx] for idx in best_local]
            if len(path_original) > len(best_path):
                best_path = path_original
                
        else:
            # Large component: Use Beam Search
            # Map comp nodes to 0..k-1
            node_map = {node: i for i, node in enumerate(comp)}
            inv_map = [0] * k
            for node, idx in node_map.items():
                inv_map[idx] = node
                
            local_adj = [set() for _ in range(k)]
            for u in comp:
                u_idx = node_map[u]
                for v in adj[u]:
                    if v in node_map:
                        local_adj[u_idx].add(node_map[v])
            
            # Beam Search Parameters
            beam_width = 2000
            # State: (end_node, visited_mask_as_tuple_or_set)
            # Using frozenset for visited is safe but might be slow.
            # Since k can be ~80, we can't use int mask.
            # We will store (end_node, frozenset(visited_indices))
            
            # Initial beam: all single nodes
            beam = []
            for i in range(k):
                beam.append((i, frozenset([i]), [i]))
                
            best_beam_path = []
            
            # Iterate until no more extensions or time limit
            # Max depth is k
            for step in range(k):
                if time.time() > deadline:
                    break
                    
                candidates = []
                for end, visited, path in beam:
                    for nei in local_adj[end]:
                        if nei not in visited:
                            new_visited = visited | {nei}
                            new_path = path + [nei]
                            # Score: length is primary. 
                            # Tie-breaker: prefer nodes with more unvisited neighbors to keep options open?
                            # Or prefer nodes with fewer to close off?
                            # Let's just keep all and select top W by length.
                            candidates.append((len(new_path), end, new_visited, new_path))
                
                if not candidates:
                    break
                
                # Sort by length descending
                candidates.sort(key=lambda x: x[0], reverse=True)
                
                # Select top W unique states (end, visited)
                seen_states = set()
                new_beam = []
                for score, end, visited, path in candidates:
                    state_key = (end, visited)
                    if state_key not in seen_states:
                        seen_states.add(state_key)
                        new_beam.append((end, visited, path))
                        if len(new_beam) >= beam_width:
                            break
                        
                if not new_beam:
                    break
                    
                # Update best path found so far
                # The candidates are sorted by length, so the first one is the longest
                # But we should check the best in the new beam
                # Actually, candidates[0] is the longest path generated in this step
                if len(candidates[0][3]) > len(best_beam_path):
                    best_beam_path = candidates[0][3]
                    
                beam = new_beam
                
            if len(best_beam_path) > len(best_path):
                path_original = [inv_map[idx] for idx in best_beam_path]
                if len(path_original) > len(best_path):
                    best_path = path_original
                    
    # Fallback: if best_path is empty (should not happen), return a single node
    if not best_path:
        best_path = [0]
        
    return best_path

# EVOLVE-BLOCK-END

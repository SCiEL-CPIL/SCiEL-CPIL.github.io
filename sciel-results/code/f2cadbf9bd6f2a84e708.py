import time
import sys
import random
from collections import deque

# Set recursion limit to be safe, though we manage depth manually via stack or recursion
sys.setrecursionlimit(20000)

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Find connected components
    visited_comp = [False] * n
    components = []
    for i in range(n):
        if not visited_comp[i]:
            comp = []
            stack = [i]
            visited_comp[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nei in adj[node]:
                    if not visited_comp[nei]:
                        visited_comp[nei] = True
                        stack.append(nei)
            components.append(comp)
            
    # Select largest component
    if not components:
        return [0] # Fallback, shouldn't happen
        
    largest_comp = max(components, key=len)
    comp_nodes = set(largest_comp)
    comp_size = len(largest_comp)
    
    best_path = []
    best_len = 0
    start_time = time.time()
    time_limit = 85.0  # Keep well under 90s with safety margin
    
    # Build component-specific adjacency for faster lookups
    comp_adj = {node: [x for x in adj[node] if x in comp_nodes] for node in largest_comp}
    
    # Initial greedy pass to get a decent lower bound
    for start in largest_comp:
        path = [start]
        seen = {start}
        while True:
            last = path[-1]
            neighbors = [x for x in comp_adj[last] if x not in seen]
            if not neighbors:
                break
            # Warnsdorff: pick neighbor with fewest unvisited neighbors
            def unvisited_deg(node):
                return len([x for x in comp_adj[node] if x not in seen])
            
            next_node = min(neighbors, key=lambda x: (unvisited_deg(x), x))
            path.append(next_node)
            seen.add(next_node)
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # Check time
    if time.time() - start_time > time_limit:
        return best_path
    
    # DFS with improved pruning
    # Key insight: use articulation point detection to prune branches that would isolate unvisited nodes
    
    def get_biconnected_components(nodes_in_subgraph):
        """Find biconnected components of the induced subgraph on nodes_in_subgraph"""
        if not nodes_in_subgraph:
            return []
        
        nodes = set(nodes_in_subgraph)
        # Build subgraph adjacency
        sub_adj = {node: [x for x in comp_adj[node] if x in nodes] for node in nodes}
        
        # Tarjan's algorithm for articulation points
        disc = {}
        low = {}
        parent = {}
        ap = set()  # articulation points
        time_counter = [0]
        
        def tarjan(u):
            disc[u] = time_counter[0]
            low[u] = time_counter[0]
            time_counter[0] += 1
            children = 0
            for v in sub_adj[u]:
                if v not in disc:
                    parent[v] = u
                    children += 1
                    tarjan(v)
                    low[u] = min(low[u], low[v])
                    # u is articulation point if:
                    # 1. u is root and has 2+ children
                    # 2. u is not root and low[v] >= disc[u]
                    if parent.get(u) is None and children > 1:
                        ap.add(u)
                    if parent.get(u) is not None and low[v] >= disc[u]:
                        ap.add(u)
                elif v != parent.get(u):
                    low[u] = min(low[u], disc[v])
        
        # Run Tarjan from each unvisited node
        for node in nodes:
            if node not in disc:
                parent[node] = None
                tarjan(node)
        
        return ap
    
    def count_reachable_from(start, blocked, seen_set):
        """Count how many unvisited nodes are reachable from start without going through blocked nodes"""
        if start in blocked or start in seen_set:
            return 0
        visited = set()
        stack = [start]
        visited.add(start)
        while stack:
            node = stack.pop()
            for nei in comp_adj[node]:
                if nei not in seen_set and nei not in blocked and nei not in visited:
                    visited.add(nei)
                    stack.append(nei)
        return len(visited)
    
    def dfs(path, seen):
        nonlocal best_path, best_len
        
        # Pruning: if current path + remaining nodes in component <= best_len, stop
        if len(path) + (comp_size - len(path)) <= best_len:
            return
            
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
            
        # Check time
        if time.time() - start_time > time_limit:
            return
            
        last = path[-1]
        # Get unvisited neighbors in the component
        candidates = [x for x in comp_adj[last] if x not in seen]
        
        if not candidates:
            return
        
        # Advanced pruning: for each candidate, check if taking it would isolate
        # unvisited nodes from the rest
        # This is expensive, so only do it when we're close to optimal
        if comp_size - len(path) < 15:
            filtered_candidates = []
            for cand in candidates:
                # If we go to cand, can we still reach most unvisited nodes?
                # Simple check: after going to cand, how many unvisited nodes remain reachable from cand?
                # If it's significantly less than total unvisited, this branch is likely bad
                unvisited_after = comp_size - (len(path) + 1)
                # Quick check: is cand a cut vertex that would isolate nodes?
                # For speed, just check if cand has edges to other unvisited nodes
                cand_unvisited_neighbors = [x for x in comp_adj[cand] if x not in seen and x != cand]
                if unvisited_after > 0 and len(cand_unvisited_neighbors) == 0:
                    # Dead end, but might still be valid if it's the last node
                    if unvisited_after == 1:
                        filtered_candidates.append(cand)
                    # else skip (will be handled by natural termination)
                else:
                    filtered_candidates.append(cand)
            candidates = filtered_candidates if filtered_candidates else candidates
        
        # Heuristic: Sort candidates by Warnsdorff (fewest unvisited neighbors first)
        def get_unvisited_deg(node):
            return len([x for x in comp_adj[node] if x not in seen])
        
        candidates.sort(key=lambda x: (get_unvisited_deg(x), x))
        
        for next_node in candidates:
            path.append(next_node)
            seen.add(next_node)
            
            dfs(path, seen)
            
            path.pop()
            seen.remove(next_node)
            
            # Check time after each branch
            if time.time() - start_time > time_limit:
                return

    # Try DFS from multiple promising starting points
    # Sort by degree in component
    degs = [(len(comp_adj[node]), node) for node in largest_comp]
    degs.sort(reverse=True)
    
    # Try DFS from top 10 nodes by degree
    for _, start_node in degs[:10]:
        if time.time() - start_time > time_limit:
            break
        dfs([start_node], {start_node})
    
    # If we still have time, try more starts
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 10:
        # Try all remaining nodes
        for start_node in largest_comp:
            if time.time() - start_time > time_limit:
                break
            dfs([start_node], {start_node})
    
    # Final polish: try to extend the best path from both ends
    if best_path and time.time() - start_time < time_limit - 2:
        best_path = list(best_path)
        best_seen = set(best_path)
        improved = True
        while improved:
            improved = False
            # Try extending from start
            if len(best_path) > 0:
                first = best_path[0]
                for nei in comp_adj[first]:
                    if nei not in best_seen:
                        # Check if we can extend
                        new_path = [nei] + best_path
                        new_seen = best_seen | {nei}
                        # Verify it's a valid path (it should be since nei-first is an edge)
                        best_path = new_path
                        best_seen = new_seen
                        best_len = len(best_path)
                        improved = True
                        break
            # Try extending from end
            if len(best_path) > 0:
                last = best_path[-1]
                for nei in comp_adj[last]:
                    if nei not in best_seen:
                        new_path = best_path + [nei]
                        new_seen = best_seen | {nei}
                        best_path = new_path
                        best_seen = new_seen
                        best_len = len(best_path)
                        improved = True
                        break
    
    return best_path

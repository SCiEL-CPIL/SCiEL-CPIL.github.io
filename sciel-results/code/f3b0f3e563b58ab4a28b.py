"""Rewritten program: structural change to a local-search / path-merging approach."""
import time
import random
from collections import defaultdict, deque

DEADLINE = 135.0  # safety margin before 150s kill

def construct():
    start_time = time.time()
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    graphs = [instance(*spec) for spec in specs]
    
    # Build adjacency for each graph
    adj_list = []
    for edges in graphs:
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        adj_list.append(adj)
    
    # Initial decomposition: each edge as its own path
    # Then apply local search to merge paths
    
    def decompose_paths(edges, adj):
        """Decompose edges into paths using a greedy + local search approach."""
        # Start with each edge as a path
        paths = [[u, v] for u, v in edges]
        
        # Track which edges are used
        edge_set = set(edges)
        
        # Greedy: try to extend paths by merging
        # We'll use a local search approach: repeatedly try to merge two paths
        # into one longer simple path
        
        # For efficiency, we'll use a simpler approach:
        # 1. Find longest simple paths greedily
        # 2. Then try to merge remaining short paths
        
        # Actually, let's use a different structural approach:
        # Model as a flow problem or use a matching-based approach
        
        # Simpler structural change: use a heuristic that finds
        # edge-disjoint paths by repeatedly finding longest simple paths
        
        remaining_edges = set(edges)
        result_paths = []
        
        # Build adjacency from remaining edges
        def build_adj_from_edges(edge_set):
            adj = defaultdict(set)
            for u, v in edge_set:
                adj[u].add(v)
                adj[v].add(u)
            return adj
        
        # Find longest simple path in a graph using DFS with pruning
        def find_longest_simple_path(adj, edge_set):
            """Find a long simple path using greedy DFS."""
            if not edge_set:
                return None
            
            # Pick a starting node - prefer nodes with higher degree
            degrees = {u: len(adj[u]) for u in adj}
            # Start from a node with max degree
            start = max(degrees.keys(), key=lambda x: degrees[x])
            
            best_path = None
            best_len = 0
            
            # Try DFS from start
            def dfs(path, visited_nodes, current_node):
                nonlocal best_path, best_len
                if len(path) > best_len:
                    best_path = list(path)
                    best_len = len(path)
                
                # Try extending
                for neighbor in sorted(adj[current_node]):
                    if neighbor not in visited_nodes:
                        # Check if edge (current_node, neighbor) is in edge_set
                        e = tuple(sorted((current_node, neighbor)))
                        if e in edge_set:
                            visited_nodes.add(neighbor)
                            path.append(neighbor)
                            dfs(path, visited_nodes, neighbor)
                            path.pop()
                            visited_nodes.discard(neighbor)
            
            dfs([start], {start}, start)
            
            if best_path is None:
                # Just take one edge
                u, v = next(iter(edge_set))
                return [u, v]
            return best_path
        
        # Greedy extraction of long paths
        remaining = set(edges)
        result_paths = []
        
        while remaining:
            adj_rem = build_adj_from_edges(remaining)
            path = find_longest_simple_path(adj_rem, remaining)
            
            if path is None:
                # Should not happen if remaining is non-empty
                break
            
            # Remove edges used in this path
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                remaining.discard(e)
            
            result_paths.append(path)
        
        return result_paths
    
    # Apply decomposition to each graph
    all_paths = []
    for i, edges in enumerate(graphs):
        if time.time() > DEADLINE:
            # Fallback: return trivial decomposition
            all_paths.append([[u, v] for u, v in edges])
            continue
        paths = decompose_paths(edges, adj_list[i])
        all_paths.append(paths)
    
    # Local search: try to improve by merging paths
    # A merge is possible if two paths can be concatenated into a simple path
    # with all edges present in the graph
    
    def can_merge(path1, path2, adj, edge_set):
        """Check if path1 and path2 can be merged into one simple path."""
        # Try all combinations of endpoints
        # path1: a -> b, path2: c -> d
        # Merge possibilities:
        # 1. b-c edge exists, no vertex overlap
        # 2. b-d edge exists, no vertex overlap (reverse path2)
        # 3. a-c edge exists, no vertex overlap (reverse path1)
        # 4. a-d edge exists, no vertex overlap (reverse both)
        
        v1 = set(path1)
        v2 = set(path2)
        if v1 & v2:
            return None
        
        endpoints1 = [path1[0], path1[-1]]
        endpoints2 = [path2[0], path2[-1]]
        
        for e1 in endpoints1:
            for e2 in endpoints2:
                e = tuple(sorted((e1, e2)))
                if e in edge_set:
                    # Check if the merge is valid
                    # Determine the merged path
                    if e1 == path1[0] and e2 == path2[0]:
                        merged = list(reversed(path1)) + list(reversed(path2))[1:]
                    elif e1 == path1[0] and e2 == path2[-1]:
                        merged = list(reversed(path1)) + path2
                    elif e1 == path1[-1] and e2 == path2[0]:
                        merged = path1 + list(reversed(path2))[1:]
                    elif e1 == path1[-1] and e2 == path2[-1]:
                        merged = path1 + path2
                    else:
                        continue
                    
                    # Verify all edges in merged path are in edge_set
                    valid = True
                    for i in range(len(merged) - 1):
                        me = tuple(sorted((merged[i], merged[i+1])))
                        if me not in edge_set:
                            valid = False
                            break
                    if valid:
                        return merged
        
        return None
    
    # Apply local search for each graph
    for i, paths in enumerate(all_paths):
        if time.time() > DEADLINE - 10:
            break
        
        edge_set = set(graphs[i])
        adj = adj_list[i]
        
        improved = True
        while improved and time.time() < DEADLINE - 5:
            improved = False
            n = len(paths)
            for i in range(n):
                if time.time() > DEADLINE - 5:
                    break
                for j in range(i + 1, n):
                    merged = can_merge(paths[i], paths[j], adj, edge_set)
                    if merged is not None:
                        paths[i] = merged
                        paths.pop(j)
                        improved = True
                        break
                if improved:
                    break
    
    return all_paths

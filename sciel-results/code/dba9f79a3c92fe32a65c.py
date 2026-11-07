"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict, deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    deadline = time.time() + 80.0  # Safety margin before 90s limit
    
    # Build adjacency set for quick lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((u, v))
        edge_set.add((v, u))
    
    def find_paths_between_pairs(branches):
        """Find vertex-disjoint paths between all pairs of branches.
        Returns list of paths (one per unordered pair) or None if impossible."""
        if len(branches) < 2:
            return None
        
        k = len(branches)
        pairs = list(combinations(range(k), 2))
        paths = {}
        used_interior = set()
        
        # Try to find paths greedily, prioritizing pairs with fewer options
        # Sort pairs by "difficulty" - pairs whose endpoints have fewer common neighbors first
        def pair_difficulty(i, j):
            bi, bj = branches[i], branches[j]
            # Count available interior vertices (not branches, not already used)
            common = (adj[bi] & adj[bj]) - set(branches) - used_interior
            return len(common)
        
        # Iteratively add paths, trying easier pairs first
        remaining_pairs = list(pairs)
        max_attempts = 50
        attempt = 0
        
        while remaining_pairs and attempt < max_attempts:
            if time.time() > deadline - 5:
                return None
            
            # Find the "hardest" remaining pair (fewest options)
            remaining_pairs.sort(key=lambda p: pair_difficulty(p[0], p[1]))
            i, j = remaining_pairs[0]
            bi, bj = branches[i], branches[j]
            
            # Try to find a path from bi to bj avoiding used_interior and other branches
            path = find_shortest_path(bi, bj, set(branches) - {bi, bj}, used_interior)
            
            if path is not None:
                # Extract interior vertices
                interior = set(path[1:-1])
                if interior & used_interior:
                    continue  # Shouldn't happen if path finding is correct
                paths[(i, j)] = path
                used_interior.update(interior)
                remaining_pairs.remove((i, j))
            else:
                # Can't find a path for this pair, try next
                remaining_pairs.pop(0)
            
            attempt += 1
        
        if len(paths) != len(pairs):
            return None
        
        # Convert to ordered list matching combinations order
        result = []
        for i, j in pairs:
            result.append(paths[(i, j)])
        return result
    
    def find_shortest_path(src, dst, forbidden, used):
        """BFS for shortest path from src to dst avoiding forbidden and used vertices."""
        if src == dst:
            return [src]
        
        # Combine forbidden and used
        blocked = forbidden | used
        
        visited = {src}
        queue = deque([(src, [src])])
        
        while queue:
            node, path = queue.popleft()
            if node == dst:
                return path
            
            for neighbor in adj[node]:
                if neighbor in blocked or neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def is_valid_clique(branches):
        """Check if all pairs of branches have direct edges (for quick filter)."""
        for i, j in combinations(range(len(branches)), 2):
            if (branches[i], branches[j]) not in edge_set:
                return False
        return True
    
    # Strategy: Start with high-degree vertices, try to build a large clique subdivision
    
    # First, find a good initial set of candidate branches
    # Use vertices with high degree as they're more likely to be in large subdivisions
    degree = {v: len(adj[v]) for v in range(n)}
    sorted_vertices = sorted(range(n), key=lambda v: degree[v], reverse=True)
    
    best_branches = None
    best_paths = None
    best_size = 2  # At least 2 branches required
    
    # Try to build a large set of branches
    # Start with the top few high-degree vertices
    candidate_pool = sorted_vertices[:12]  # Consider top 12 vertices
    
    # Greedy extension: start with a small set, try to add vertices
    # First, find an initial clique (all pairs connected)
    initial_clique = []
    for v in candidate_pool:
        if all((v, u) in edge_set for u in initial_clique):
            initial_clique.append(v)
    
    if len(initial_clique) < 2:
        # Fallback: just take two connected vertices
        for u, v in edges:
            initial_clique = [u, v]
            break
    
    # Try to extend the clique
    current_branches = list(initial_clique)
    
    # Try adding vertices from candidate_pool
    for v in candidate_pool:
        if v in current_branches:
            continue
        if time.time() > deadline - 10:
            break
        
        # Check if v connects to all current branches
        if all((v, u) in edge_set for u in current_branches):
            # Try to find valid paths with v added
            new_branches = current_branches + [v]
            paths = find_paths_between_pairs(new_branches)
            if paths is not None:
                current_branches = new_branches
                best_branches = list(current_branches)
                best_paths = paths
                best_size = len(current_branches)
    
    # If we didn't find anything better, use the initial clique
    if best_branches is None and len(initial_clique) >= 2:
        paths = find_paths_between_pairs(initial_clique)
        if paths is not None:
            best_branches = list(initial_clique)
            best_paths = paths
            best_size = len(initial_clique)
    
    # Fallback: just return two connected vertices
    if best_branches is None:
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = [[u, v]]
        best_size = 2
    
    # Verify and return
    result = {'branches': list(best_branches), 'paths': [list(p) for p in best_paths]}
    return result

# EVOLVE-BLOCK-END

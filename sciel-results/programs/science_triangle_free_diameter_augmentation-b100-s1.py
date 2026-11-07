"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_base_adj(n):
    """Build adjacency list and set for the base path graph."""
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
    return adj, edge_set

def _is_triangle_free(adj, n, added_edges):
    """Check if adding edges to base graph keeps it triangle-free."""
    # For each added edge (u,v), check no common neighbor exists in the full graph
    full_adj = [set(adj[i]) for i in range(n)]
    for u, v in added_edges:
        full_adj[u].add(v)
        full_adj[v].add(u)
    
    for u, v in added_edges:
        # Check if u and v share a common neighbor
        common = full_adj[u] & full_adj[v]
        if common:
            return False
    return True

def _diameter(adj, n):
    """Compute graph diameter using BFS from each node."""
    max_d = 0
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = deque([src])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        for d in dist:
            if d > max_d:
                max_d = d
        if max_d > 4:
            return max_d
    return max_d

def _valid_solution(added_edges, n, base_adj, base_edge_set):
    """Check if a set of added edges is valid."""
    # Check no duplicates with base edges
    for u, v in added_edges:
        key = (min(u, v), max(u, v))
        if key in base_edge_set:
            return False
    
    # Check no duplicate added edges
    seen = set()
    for u, v in added_edges:
        key = (min(u, v), max(u, v))
        if key in seen:
            return False
        seen.add(key)
    
    # Check triangle-free
    if not _is_triangle_free(base_adj, n, added_edges):
        return False
    
    # Check diameter
    adj = [list(base_adj[i]) for i in range(n)]
    for u, v in added_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    if _diameter(adj, n) > 4:
        return False
    
    return True

def _find_all_candidate_edges(n, base_edge_set):
    """Find all possible edges that can be added (not in base, not creating immediate triangle with base)."""
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) in base_edge_set:
                continue
            # Check if adding (u,v) creates a triangle with base edges
            # u and v must not share a common neighbor in the base graph
            # Base graph is a path, so u and v share a neighbor only if |u-v| == 2
            if abs(u - v) == 2:
                continue
            candidates.append((u, v))
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s margin for 90s limit
    
    n = P['n']
    base_adj, base_edge_set = _build_base_adj(n)
    candidates = _find_all_candidate_edges(n, base_edge_set)
    
    # Start with known good solution of 10 edges
    best_solution = [
        (0, 12), (2, 12), (4, 12), (6, 12), (8, 12),
        (15, 12), (17, 12), (19, 12), (21, 12), (23, 12)
    ]
    
    # Try to find solutions with fewer edges
    # For k edges, try combinations from candidates
    # Focus on promising candidates: those connecting distant vertices
    
    # Score candidates by "usefulness" - prefer edges that shorten long paths
    def candidate_score(edge):
        u, v = edge
        return abs(u - v)
    
    candidates_sorted = sorted(candidates, key=candidate_score, reverse=True)
    
    # Try k = 5, 6, 7, 8, 9 edges
    for k in range(5, 10):
        if time.time() > deadline:
            break
        
        # For each k, try combinations of top candidates
        # Use a heuristic: focus on edges involving central vertices
        # Central vertices are around 11-12
        
        # Generate promising subsets: edges that connect to central region
        central = set(range(8, 16))
        promising = [e for e in candidates_sorted if e[0] in central or e[1] in central]
        
        if len(promising) < k:
            promising = candidates_sorted
        
        # Try random subsets from promising candidates
        import random
        random.seed(42)
        tried = set()
        
        max_tries = 50000
        tries = 0
        while tries < max_tries and time.time() < deadline:
            subset = random.sample(promising, k)
            subset_tuple = tuple(sorted(subset))
            if subset_tuple in tried:
                tries += 1
                continue
            tried.add(subset_tuple)
            
            if _valid_solution(subset, n, base_adj, base_edge_set):
                best_solution = subset
                break
            tries += 1
        
        if len(best_solution) <= k:
            break
    
    # Also try a structured approach: star-like solutions
    # Pick a center and connect to multiple vertices
    for center in range(4, 20):
        if time.time() > deadline:
            break
        
        # Find vertices that can connect to center without creating triangles
        # Center c: cannot connect to c-1, c+1 (already adjacent), c-2, c+2 (would form triangle)
        valid_targets = []
        for v in range(n):
            if v == center:
                continue
            if abs(v - center) <= 2:
                continue
            # Check if (center, v) is in base
            if (min(center, v), max(center, v)) in base_edge_set:
                continue
            valid_targets.append(v)
        
        # Try different numbers of connections from center
        for num_conn in range(1, min(10, len(valid_targets) + 1)):
            if time.time() > deadline:
                break
            
            # Try random subsets of valid_targets
            import random
            random.seed(center * 1000 + num_conn)
            
            max_inner_tries = 10000
            for _ in range(max_inner_tries):
                if time.time() > deadline:
                    break
                targets = random.sample(valid_targets, num_conn)
                subset = [(center, t) for t in targets]
                subset_tuple = tuple(sorted(subset))
                
                if _valid_solution(subset, n, base_adj, base_edge_set):
                    if len(subset) < len(best_solution):
                        best_solution = subset
                    break
    
    # Final verification and return
    if _valid_solution(best_solution, n, base_adj, base_edge_set):
        return [[u, v] for u, v in best_solution]
    else:
        # Fallback to known good solution
        fallback = [
            [0, 12], [2, 12], [4, 12], [6, 12], [8, 12],
            [15, 12], [17, 12], [19, 12], [21, 12], [23, 12]
        ]
        if _valid_solution(fallback, n, base_adj, base_edge_set):
            return fallback
        return []

# EVOLVE-BLOCK-END

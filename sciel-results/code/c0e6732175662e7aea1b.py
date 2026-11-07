"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from collections import deque
import itertools

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edge_set = set()
    for e in P['edges']:
        base_edge_set.add(frozenset(e))
    
    # Build base adjacency
    base_adj = [set() for _ in range(n)]
    for e in P['edges']:
        u, v = e
        base_adj[u].add(v)
        base_adj[v].add(u)
    
    # Generate all non-base candidate edges
    all_candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if frozenset((u, v)) not in base_edge_set:
                all_candidates.append((u, v))
    
    def build_adj(edge_list):
        adj = [set() for _ in range(n)]
        for e in P['edges']:
            u, v = e
            adj[u].add(v)
            adj[v].add(u)
        for e in edge_list:
            u, v = e
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def is_triangle_free(edge_list):
        adj = build_adj(edge_list)
        for u in range(n):
            for v in adj[u]:
                if v > u:
                    common = adj[u] & adj[v]
                    if common:
                        return False
        return True
    
    def compute_diameter(edge_list):
        adj = build_adj(edge_list)
        max_d = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        if dist[nb] > max_d:
                            max_d = dist[nb]
                        queue.append(nb)
            if any(d == -1 for d in dist):
                return n + 1  # disconnected
        return max_d
    
    def check_valid(edge_list):
        if not is_triangle_free(edge_list):
            return False
        d = compute_diameter(edge_list)
        return d <= 4
    
    # Greedy set-cover approach:
    # Step 1: Find which vertex pairs have distance > 4 in base graph
    base_diam = compute_diameter([])
    
    # For each candidate edge, compute which pairs it "helps" by reducing distance
    # Use a greedy approach: at each step, pick the edge that reduces the number of pairs with dist > 4 the most
    
    def get_pairs_over_4(edge_list):
        adj = build_adj(edge_list)
        pairs = []
        for i in range(n):
            dist = [-1] * n
            dist[i] = 0
            queue = deque([i])
            while queue:
                node = queue.popleft()
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        queue.append(nb)
            for j in range(i + 1, n):
                if dist[j] > 4:
                    pairs.append((i, j))
        return set(pairs)
    
    current_edges = []
    current_pairs = get_pairs_over_4(current_edges)
    
    # Pre-filter: only consider edges that are individually triangle-free with base
    valid_candidates = []
    for u, v in all_candidates:
        # Check if adding (u,v) to base creates a triangle
        common = base_adj[u] & base_adj[v]
        if not common:
            valid_candidates.append((u, v))
    
    # Greedy: repeatedly pick edge that covers most uncovered pairs
    while current_pairs:
        best_edge = None
        best_covered = set()
        for u, v in valid_candidates:
            if (u, v) in current_edges or (v, u) in current_edges:
                continue
            # Check if adding this edge maintains triangle-freeness
            test_edges = current_edges + [(u, v)]
            if not is_triangle_free(test_edges):
                continue
            # Compute which pairs are now within distance 4
            new_pairs = get_pairs_over_4(test_edges)
            covered = current_pairs - new_pairs
            if len(covered) > len(best_covered):
                best_covered = covered
                best_edge = (u, v)
        
        if best_edge is None:
            break
        
        current_edges.append(best_edge)
        current_pairs = get_pairs_over_4(current_edges)
    
    # Verify
    if check_valid(current_edges):
        # Try to remove redundant edges
        final_edges = list(current_edges)
        for i in range(len(final_edges)):
            for j in range(i):
                # Try removing edge j
                test = [e for k, e in enumerate(final_edges) if k != j]
                if check_valid(test):
                    final_edges.pop(j)
                    # Need to re-check since indices changed, so restart
                    break
            else:
                continue
            break
        
        # More careful removal: iterate until no more removals possible
        improved = True
        while improved:
            improved = False
            for i in range(len(final_edges)):
                test = [e for k, e in enumerate(final_edges) if k != i]
                if check_valid(test):
                    final_edges.pop(i)
                    improved = True
                    break
        
        return [[u, v] for u, v in final_edges]
    
    # Fallback: use a known-good construction
    # Try hub-based approach
    for hub in range(n):
        for targets in itertools.combinations([i for i in range(n) if i != hub], 4):
            edges = [(hub, t) for t in targets]
            if check_valid(edges):
                return [[u, v] for u, v in edges]
    
    # Last resort: return empty (should not happen)
    return []
# EVOLVE-BLOCK-END

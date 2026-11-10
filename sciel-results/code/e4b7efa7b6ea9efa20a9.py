import time
from itertools import combinations
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s margin before 90s kill
    
    n = P['n']
    # Build adjacency sets for O(1) lookup
    adj_sets = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj_sets[u].add(v)
        adj_sets[v].add(u)
    
    # Precompute common neighbors for all pairs
    common = [None] * (n * n)
    for u in range(n):
        for v in range(u + 1, n):
            common[u * n + v] = adj_sets[u] & adj_sets[v]
    
    def are_adjacent(u, v):
        return v in adj_sets[u]
    
    def hopcroft_karp(left_nodes, right_nodes, adj_list, deadline):
        """
        Hopcroft-Karp bipartite matching.
        left_nodes: list of left side nodes
        right_nodes: list of right side nodes  
        adj_list: dict mapping left node -> list of right nodes
        Returns matching dict {left_node: right_node} or None if no perfect matching
        """
        if not left_nodes:
            return {}
        if len(left_nodes) > len(right_nodes):
            return None
        
        # Map to indices for efficiency
        right_set = set(right_nodes)
        pair_u = {}  # left node -> matched right node
        pair_v = {}  # right node -> matched left node
        dist = {}
        
        def bfs():
            nonlocal dist
            dist = {}
            queue = deque()
            for u in left_nodes:
                if u not in pair_u:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            found = False
            while queue:
                u = queue.popleft()
                if time.time() > deadline:
                    return False
                for v in adj_list.get(u, []):
                    if v in right_set:
                        pu = pair_v.get(v)
                        if pu is None:
                            found = True
                        elif dist.get(pu, float('inf')) == float('inf'):
                            dist[pu] = dist[u] + 1
                            queue.append(pu)
            return found
        
        def dfs(u, visited):
            nonlocal pair_u, pair_v
            if time.time() > deadline:
                return False
            for v in adj_list.get(u, []):
                if v not in right_set:
                    continue
                if v in visited:
                    continue
                visited.add(v)
                pu = pair_v.get(v)
                if pu is None or (dist.get(pu, float('inf')) == dist.get(u, 0) + 1 and dfs(pu, visited)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = float('inf')
            return False
        
        matching_size = 0
        while bfs():
            if time.time() > deadline:
                return None
            visited = set()
            for u in left_nodes:
                if u not in pair_u:
                    if dfs(u, visited):
                        matching_size += 1
            visited.clear()
        
        if matching_size == len(left_nodes):
            return pair_u
        return None
    
    def check_and_build(branch_combo, deadline):
        """Check if branch_combo is feasible and build paths if so."""
        branch_set = set(branch_combo)
        internal_candidates = [v for v in range(n) if v not in branch_set]
        pairs = list(combinations(branch_combo, 2))
        
        # For each pair, determine if adjacent or needs internal
        non_adj_pairs = []
        adj_pairs = []
        for u, v in pairs:
            if are_adjacent(u, v):
                adj_pairs.append((u, v))
            else:
                common_v = common[min(u,v) * n + max(u,v)]
                # Filter to internal candidates only
                valid_common = [w for w in common_v if w not in branch_set]
                if not valid_common:
                    return None
                non_adj_pairs.append((u, v, valid_common))
        
        if len(non_adj_pairs) > len(internal_candidates):
            return None
        
        # Build adjacency list for matching
        # Left side: non_adj_pairs (indexed), Right side: internal vertices
        adj_list = {}
        for i, (u, v, vc) in enumerate(non_adj_pairs):
            adj_list[i] = vc
        
        # Run Hopcroft-Karp
        left_nodes = list(range(len(non_adj_pairs)))
        result = hopcroft_karp(left_nodes, internal_candidates, adj_list, deadline)
        
        if result is None:
            return None
        
        # Build paths
        paths = []
        for u, v in adj_pairs:
            paths.append([u, v])
        for i, (u, v, vc) in enumerate(non_adj_pairs):
            w = result[i]
            paths.append([u, w, v])
        
        return {'branches': list(branch_combo), 'paths': paths}
    
    # Try k from high to low
    best_result = None
    best_k = 0
    
    # Sort vertices by degree descending - prefer high-degree vertices as branches
    degree_sorted = sorted(range(n), key=lambda v: len(adj_sets[v]), reverse=True)
    
    for k in range(9, 1, -1):
        if time.time() > deadline:
            break
        
        # Try combinations, prioritizing high-degree vertices
        # Use degree-sorted ordering to find good combos faster
        candidates = degree_sorted[:min(n, 20)]  # Focus on top candidates
        
        # Generate combinations from candidates first
        found = False
        for branch_combo in combinations(candidates, k):
            if time.time() > deadline:
                break
            result = check_and_build(branch_combo, deadline)
            if result is not None:
                best_result = result
                best_k = k
                found = True
                break
        
        if found:
            break
        
        # If not found in top candidates, try all
        if not found:
            for branch_combo in combinations(range(n), k):
                if time.time() > deadline:
                    break
                result = check_and_build(branch_combo, deadline)
                if result is not None:
                    best_result = result
                    best_k = k
                    found = True
                    break
        
        if found:
            break
    
    # Fallback
    if best_result is not None:
        return best_result
    
    # Try k=3 with any combo
    for combo in combinations(range(n), 3):
        if time.time() > deadline:
            break
        result = check_and_build(combo, deadline)
        if result is not None:
            return result
    
    # Final fallback: single edge
    u, v = P['edges'][0]
    return {'branches': [u, v], 'paths': [[u, v]]}
# EVOLVE-BLOCK-END

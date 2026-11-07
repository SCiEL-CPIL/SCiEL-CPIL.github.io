"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets for fast lookups
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Convert to frozenset for faster hashing if needed, but sets are fine for membership
    
    # Helper: check if a tuple/list of vertices is a clique
    def is_clique(vertices):
        verts = list(vertices)
        for i in range(len(verts)):
            for j in range(i + 1, len(verts)):
                if verts[j] not in adj[verts[i]]:
                    return False
        return True
    
    # Helper: find a simple path between u and v that avoids a set of forbidden vertices
    # Returns list of vertices [u, ..., v] or None
    # Uses BFS
    def find_path_avoiding(u, v, forbidden):
        if u == v:
            return [u]
        if v in forbidden:
            return None
        if u in forbidden:
            return None
            
        from collections import deque
        # BFS
        queue = deque()
        queue.append((u, [u]))
        visited = {u}
        
        while queue:
            node, path = queue.popleft()
            if node == v:
                return path
            for neighbor in adj[node]:
                if neighbor not in visited and neighbor not in forbidden:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    # Helper: Validate a full clique subdivision solution
    # branches: list of branch vertices
    # paths: dict mapping (u, v) tuple to path list
    def validate_solution(branches, paths_dict):
        # Check branches are distinct
        if len(branches) != len(set(branches)):
            return False
        
        # Check all pairs exist in paths_dict
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                pair = (branches[i], branches[j])
                if pair not in paths_dict and (branches[j], branches[i]) not in paths_dict:
                    return False
        
        # Collect all interior vertices
        all_interiors = set()
        for pair, path in paths_dict.items():
            # Path must start and end at branches
            u, v = pair
            if path[0] != u and path[0] != v:
                return False
            if path[-1] != u and path[-1] != v:
                return False
            
            # Check path is simple (no repeated vertices)
            if len(path) != len(set(path)):
                return False
            
            # Check adjacency
            for k in range(len(path) - 1):
                if path[k+1] not in adj[path[k]]:
                    return False
            
            # Extract interiors (exclude endpoints)
            interiors = set(path[1:-1])
            
            # Interiors must not contain any branch
            for b in branches:
                if b in interiors:
                    return False
            
            # Interiors must be disjoint from other paths' interiors
            if all_interiors & interiors:
                return False
            
            all_interiors.update(interiors)
            
        return True

    # Strategy:
    # 1. Try to find large cliques (k=6 down to 3). If found, return immediately (paths are edges).
    # 2. If no large clique, try to find K4 subdivisions (4 branches).
    #    - Try all 4-tuples.
    #    - For each 4-tuple, check if they form a clique (already handled in step 1 if k>=4, but maybe we missed some due to time? No, step 1 is exhaustive for k).
    #    - If not a clique, try to find disjoint paths for the 6 pairs.
    #    - This is NP-hard in general, but for small n=24 and specific structure, we can use a greedy/heuristic approach.
    #    - Heuristic: Order pairs by shortest path length (excluding branches). Try to assign paths in that order, updating forbidden set.
    
    best_branches = None
    best_paths = None
    
    # Step 1: Find max clique
    max_clique_size = 0
    max_clique_vertices = []
    
    # We only need to check k down to 4. If we find a clique of size >= 4, it's likely optimal or close.
    # Actually, we should check up to a reasonable limit. C(24, 8) is large. Let's check k=6 first, then 5, then 4.
    # If we find a clique of size 6, that's 6 branches.
    # If we find a clique of size 5, that's 5 branches.
    # If we find a clique of size 4, that's 4 branches.
    
    # To save time, we can use a branching algorithm or just iterate combinations for smaller k.
    # Let's iterate k from 6 down to 4.
    for k in range(6, 3, -1):
        found = False
        for comb in itertools.combinations(range(n), k):
            if is_clique(comb):
                max_clique_size = k
                max_clique_vertices = list(comb)
                found = True
                break
        if found:
            break
        
        # Time check
        if time.time() - start_time > DEADLINE - 5:
            break
    
    if max_clique_size >= 4:
        branches = max_clique_vertices
        paths_dict = {}
        for i in range(len(branches)):
            for j in range(i + 1, len(branches)):
                u, v = branches[i], branches[j]
                paths_dict[(u, v)] = [u, v]
        
        # Construct return format
        paths_list = []
        for pair in itertools.combinations(branches, 2):
            p = paths_dict.get(pair) or paths_dict.get((pair[1], pair[0]))
            paths_list.append(p)
            
        return {'branches': branches, 'paths': paths_list}
    
    # Step 2: Try to find a K4 subdivision (4 branches)
    # If we didn't find a clique of size >= 4, we might still find a subdivision of K4.
    # This allows 4 branches even if no K4 clique exists.
    # We iterate over all 4-tuples of vertices.
    # For each 4-tuple (a, b, c, d), we need to find disjoint paths:
    # a-b, a-c, a-d, b-c, b-d, c-d
    # avoiding {a, b, c, d} in the interiors.
    
    # To make this feasible, we can prioritize 4-tuples that are "close" to being a clique.
    # Or just iterate. C(24, 4) = 10,626. For each, we run a search.
    # The search involves finding 6 disjoint paths.
    # We can use a simple backtracking or greedy approach.
    
    # Greedy approach:
    # 1. Find shortest path for each pair avoiding the 4 branches.
    # 2. Sort pairs by path length.
    # 3. Greedily assign paths, marking used vertices as forbidden.
    # 4. If all 6 assigned successfully, we have a solution.
    
    def try_find_k4_subdivision(branches_4):
        a, b, c, d = branches_4
        branch_set = {a, b, c, d}
        pairs = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
        
        # Find shortest paths for each pair avoiding branch_set
        paths_found = {}
        for u, v in pairs:
            p = find_path_avoiding(u, v, branch_set)
            if p is None:
                return None
            paths_found[(u, v)] = p
        
        # Sort pairs by path length (shortest first)
        sorted_pairs = sorted(pairs, key=lambda x: len(paths_found[x]))
        
        forbidden_interiors = set()
        assigned_paths = {}
        
        for u, v in sorted_pairs:
            p = paths_found[(u, v)]
            interiors = set(p[1:-1])
            
            # Check if interiors are disjoint from already forbidden
            if interiors & forbidden_interiors:
                return None
            
            # Also ensure interiors don't contain branches (already checked in find_path_avoiding)
            assigned_paths[(u, v)] = p
            forbidden_interiors.update(interiors)
        
        # If we get here, we have a valid solution
        return assigned_paths
    
    # Iterate over all 4-tuples
    # To improve efficiency, we can first check if the 4-tuple has many edges between them.
    # A clique of size 4 is already handled. 
    # We can skip 4-tuples that have very few edges between them, as they are less likely to have disjoint paths.
    # But let's just try them, with a time limit.
    
    count = 0
    for comb in itertools.combinations(range(n), 4):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Quick check: if they form a clique, we would have found it in step 1.
        # But step 1 only went up to k=6. If k=4 clique exists, it was found.
        # So we only need to check non-cliques here?
        # Actually, if is_clique(comb) is True, the paths are just edges, which is valid.
        # But we already returned if we found a clique of size >= 4.
        # So any 4-tuple we process here is NOT a clique.
        
        # Heuristic: Count edges between the 4 vertices.
        # If there are very few edges, it's unlikely to have disjoint paths.
        # Minimum edges for K4 subdivision? 
        # Each pair needs a path. If u-v is an edge, path is [u,v].
        # If not, path must go through other vertices.
        # Let's just try the greedy search.
        
        result = try_find_k4_subdivision(comb)
        if result is not None:
            branches = list(comb)
            paths_list = []
            for pair in itertools.combinations(branches, 2):
                p = result.get(pair) or result.get((pair[1], pair[0]))
                paths_list.append(p)
            
            # Final validation just in case
            if validate_solution(branches, result):
                return {'branches': branches, 'paths': paths_list}
        
        count += 1
    
    # Step 3: Fallback to K3 (triangle) if no K4 subdivision found
    # Find any triangle
    for comb in itertools.combinations(range(n), 3):
        if is_clique(comb):
            branches = list(comb)
            paths_list = []
            for pair in itertools.combinations(branches, 2):
                paths_list.append(list(pair))
            return {'branches': branches, 'paths': paths_list}
    
    # Step 4: Fallback to edge
    if edges:
        u, v = edges[0]
        return {'branches': [u, v], 'paths': [[u, v]]}
    else:
        return {'branches': [0, 1], 'paths': [[0, 1]]}

# EVOLVE-BLOCK-END

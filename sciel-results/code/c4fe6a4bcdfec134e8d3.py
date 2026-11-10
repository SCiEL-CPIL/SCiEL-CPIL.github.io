"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
from itertools import combinations
from collections import deque
import random
import sys

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = _time.time()
    deadline = start_time + 85  # 90s limit, 5s margin
    
    n = P['n']
    edges = P['edges']
    
    # Bitmask adjacency
    adj_mask = [0] * n
    adj_list = [[] for _ in range(n)]
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        adj_list[u].append(v)
        adj_list[v].append(u)
        
    all_mask = (1 << n) - 1

    def time_check():
        return _time.time() > deadline

    def find_path_bfs(src, dst, forbidden_mask):
        """Find shortest path from src to dst avoiding vertices in forbidden_mask.
        src and dst are allowed even if in forbidden_mask (but they shouldn't be).
        Returns list of vertices or None.
        """
        if src == dst:
            return [src]
        
        # If src or dst is forbidden, we can't use them as interior, but they are endpoints.
        # The problem says "avoid every branch" for interiors.
        # So forbidden_mask should contain all branches except src and dst?
        # Actually, the caller should pass a mask of all vertices that cannot be used as interiors.
        
        parent = [-1] * n
        parent[src] = src
        q = deque([src])
        
        while q:
            u = q.popleft()
            if u == dst:
                path = []
                cur = dst
                while cur != src:
                    path.append(cur)
                    cur = parent[cur]
                path.append(src)
                return path[::-1]
            
            # Neighbors
            neighbors_mask = adj_mask[u]
            # Exclude forbidden vertices, but allow dst
            # We want to visit v if v is not in forbidden_mask OR v == dst
            # Also v should not be visited
            # visited is tracked by parent != -1
            
            # Iterate through set bits in neighbors_mask
            m = neighbors_mask
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                m ^= lsb
                
                if parent[v] != -1:
                    continue
                if v == dst:
                    parent[v] = u
                    # We can return immediately or let the loop handle it
                    # Let's let the loop handle it for simplicity, or break
                    # Actually, BFS guarantees shortest path, so we can return
                    path = []
                    cur = dst
                    while cur != src:
                        path.append(cur)
                        cur = parent[cur]
                    path.append(src)
                    return path[::-1]
                
                if (forbidden_mask >> v) & 1:
                    continue
                    
                parent[v] = u
                q.append(v)
                
        return None

    def check_routable(branches):
        """
        Check if the set of branches has disjoint paths.
        Returns dict pair -> path, or None.
        """
        bset = set(branches)
        branch_mask = 0
        for b in branches:
            branch_mask |= (1 << b)
            
        pairs = list(combinations(branches, 2))
        if not pairs:
            return {}
            
        # Separate direct and non-direct pairs
        direct_pairs = []
        need_pairs = [] # (u, v, common_mask)
        
        for u, v in pairs:
            if (adj_mask[u] >> v) & 1:
                direct_pairs.append((u, v))
            else:
                common = adj_mask[u] & adj_mask[v] & ~branch_mask
                if common == 0:
                    return None
                need_pairs.append((u, v, common))
                
        paths = {}
        
        # Handle direct pairs
        for u, v in direct_pairs:
            paths[(u, v)] = [u, v]
            
        # Handle need_pairs with matching
        if need_pairs:
            # Bipartite matching: Left = need_pairs, Right = vertices
            # We want to assign a unique vertex to each pair.
            
            num_left = len(need_pairs)
            # left_adj[i] is the set of vertices that can serve as intermediate for pair i
            left_adj = [0] * num_left # Use bitmask for right side too? 
            # Actually, let's use a list of sets or bitmasks for the right side.
            # Since n=24, we can use bitmasks for the right side as well.
            
            for i, (u, v, common) in enumerate(need_pairs):
                left_adj[i] = common
                
            # Hopcroft-Karp or simple augmenting path
            # Simple augmenting path is fine for small N
            match_right = [-1] * n # match_right[v] = index of left node
            match_left = [-1] * num_left # match_left[i] = vertex v
            
            def try_augment(li, visited):
                # visited is a bitmask of right nodes visited
                m = left_adj[li]
                while m:
                    lsb = m & (-m)
                    v = lsb.bit_length() - 1
                    m ^= lsb
                    if (visited >> v) & 1:
                        continue
                    visited |= (1 << v)
                    
                    if match_right[v] == -1:
                        match_right[v] = li
                        match_left[li] = v
                        return True
                    else:
                        if try_augment(match_right[v], visited):
                            match_right[v] = li
                            match_left[li] = v
                            return True
                return False
                
            for i in range(num_left):
                visited = 0
                if not try_augment(i, visited):
                    return None # Matching failed
            
            # Assign paths
            for i, (u, v, common) in enumerate(need_pairs):
                w = match_left[i]
                paths[(u, v)] = [u, w, v]
                
        return paths

    def check_routable_general(branches):
        """
        Fallback for when length-2 matching fails.
        Uses backtracking with BFS.
        """
        bset = set(branches)
        branch_mask = 0
        for b in branches:
            branch_mask |= (1 << b)
            
        pairs = list(combinations(branches, 2))
        if not pairs:
            return {}
            
        # Sort pairs by number of common neighbors (ascending) to constrain early
        def pair_key(p):
            u, v = p
            common = adj_mask[u] & adj_mask[v] & ~branch_mask
            return common.bit_count()
            
        pairs_sorted = sorted(pairs, key=pair_key)
        
        paths = {}
        used_mask = 0 # Mask of used interior vertices
        
        def backtrack(idx):
            if time_check():
                return False
            if idx == len(pairs_sorted):
                return True
                
            u, v = pairs_sorted[idx]
            
            # If u and v are adjacent, path is [u, v]
            if (adj_mask[u] >> v) & 1:
                paths[(u, v)] = [u, v]
                if backtrack(idx + 1):
                    return True
                del paths[(u, v)]
                return False
                
            # Otherwise, find a path
            # Forbidden mask: all branches except u, v, plus used interiors
            forbidden = (branch_mask ^ (1 << u) ^ (1 << v)) | used_mask
            
            path = find_path_bfs(u, v, forbidden)
            if path is None:
                return False
                
            interior_mask = 0
            for w in path[1:-1]:
                interior_mask |= (1 << w)
                
            # Check if interior is disjoint from used
            if interior_mask & used_mask:
                return False
                
            # Check if interior contains any branch (should be handled by forbidden, but just in case)
            if interior_mask & branch_mask:
                return False
                
            paths[(u, v)] = path
            used_mask_new = used_mask | interior_mask
            
            if backtrack(idx + 1):
                return True
                
            del paths[(u, v)]
            return False
            
        if backtrack(0):
            return paths
        return None

    def find_max_branches():
        best_branches = None
        best_paths = None
        
        # 1. Max Clique
        clique = max_clique()
        if len(clique) >= 2:
            paths = check_routable(clique)
            if paths is None:
                paths = check_routable_general(clique)
            if paths is not None:
                best_branches = list(clique)
                best_paths = paths
                
        # 2. Greedy Expansion
        if best_branches:
            current = list(best_branches)
            improved = True
            while improved and not time_check():
                improved = False
                for candidate in range(n):
                    if candidate in current:
                        continue
                    new_set = current + [candidate]
                    paths = check_routable(new_set)
                    if paths is None:
                        paths = check_routable_general(new_set)
                    if paths is not None:
                        current = new_set
                        best_branches = current
                        best_paths = paths
                        improved = True
                        break
                        
        # 3. Top Degree Vertices
        deg = [adj_mask[i].bit_count() for i in range(n)]
        top = sorted(range(n), key=lambda x: -deg[x])[:12]
        for size in range(min(10, len(top)), 1, -1):
            if time_check():
                break
            found = False
            for combo in combinations(top, size):
                if time_check():
                    break
                paths = check_routable(list(combo))
                if paths is None:
                    paths = check_routable_general(list(combo))
                if paths is not None and (best_branches is None or size > len(best_branches)):
                    best_branches = list(combo)
                    best_paths = paths
                    found = True
                    break
            if found:
                break
                
        # 4. Random Search
        random.seed(42)
        for _ in range(200):
            if time_check():
                break
            size = random.randint(4, 14)
            combo = random.sample(range(n), size)
            paths = check_routable(combo)
            if paths is None:
                paths = check_routable_general(combo)
            if paths is not None and (best_branches is None or size > len(best_branches)):
                best_branches = combo
                best_paths = paths
                
        if best_branches is None or len(best_branches) < 2:
            u, v = edges[0]
            best_branches = [u, v]
            best_paths = {(u, v): [u, v]}
            
        branches = [int(b) for b in best_branches]
        paths_list = []
        for (u, v) in combinations(branches, 2):
            path = best_paths.get((u, v)) or best_paths.get((v, u))
            if path is None:
                path = [u, v]
            paths_list.append([int(x) for x in path])
            
        return {'branches': branches, 'paths': paths_list}

    def max_clique():
        best = []
        
        def expand(clique, candidates):
            nonlocal best
            if time_check():
                return
            if not candidates:
                if len(clique) > len(best):
                    best = list(clique)
                return
                
            # Color bound
            colors = {}
            for v in candidates:
                used_colors = set()
                for u in candidates:
                    if u in adj[v] and u in colors:
                        used_colors.add(colors[u])
                c = 0
                while c in used_colors:
                    c += 1
                colors[v] = c
                
            max_color = max(colors.values()) + 1
            if len(clique) + max_color <= len(best):
                return
                
            ordered = sorted(candidates, key=lambda v: -colors[v])
            candidates_set = set(candidates)
            
            for v in ordered:
                if v not in candidates_set:
                    continue
                new_clique = clique + [v]
                new_candidates = [u for u in candidates_set if u in adj[v] and u != v]
                expand(new_clique, new_candidates)
                candidates_set.discard(v)
                remaining_max = 0
                for u in candidates_set:
                    if all(u in adj[c] for c in clique):
                        remaining_max = max(remaining_max, colors.get(u, 0) + 1)
                if len(clique) + remaining_max <= len(best):
                    break
                    
        expand([], list(range(n)))
        return best

    return find_max_branches()
# EVOLVE-BLOCK-END

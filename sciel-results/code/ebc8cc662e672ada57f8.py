import time
import random
from itertools import combinations
from collections import deque

def construct():
    t_start = time.time()
    DEADLINE = 85.0

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    adj_set = adj

    def time_up():
        return time.time() - t_start > DEADLINE

    def find_shortest_path(u, v, banned):
        """
        Find shortest path from u to v avoiding 'banned' vertices.
        u and v are assumed not to be in 'banned' (or we handle it).
        Returns list of vertices or None.
        """
        if u == v:
            return None
        if v in adj_set[u]:
            return [u, v]
        
        parent = {u: None}
        q = deque([u])
        while q:
            cur = q.popleft()
            for nxt in adj_set[cur]:
                if nxt == v:
                    # Reconstruct
                    path = [v]
                    node = cur
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                if nxt in parent or nxt in banned:
                    continue
                parent[nxt] = cur
                q.append(nxt)
        return None

    def try_find_subdivision(branches, rng=None):
        """
        Try to find disjoint interior paths for all pairs in 'branches'.
        Returns dict of pair->path or None.
        """
        if rng is None:
            rng = random.Random()
            
        bset = set(branches)
        pairs = list(combinations(branches, 2))
        # Shuffle pairs to try different orders if needed?
        # Actually, a good heuristic is to process pairs where the endpoints have low connectivity first?
        # Or just shuffle.
        
        attempts = 10 # Try 10 random orderings
        for _ in range(attempts):
            random.shuffle(pairs)
            used_interior = set()
            paths = {}
            success = True
            for a, b in pairs:
                # Banned: all branches + used interior
                # But a and b are allowed as endpoints
                banned = used_interior | bset
                banned.discard(a)
                banned.discard(b)
                
                path = find_shortest_path(a, b, banned)
                if path is None:
                    success = False
                    break
                
                interior = set(path[1:-1])
                # Check if interior intersects with used_interior (should be handled by banned)
                # Check if interior intersects with branches (should be handled by banned)
                if interior & bset:
                    success = False
                    break
                if interior & used_interior:
                    success = False
                    break
                
                used_interior |= interior
                paths[frozenset((a, b))] = path
            
            if success:
                return paths
        return None

    def validate(branches, paths):
        if not branches or len(branches) < 2:
            return 0
        bset = set(branches)
        if len(bset) != len(branches):
            return 0
        expected_pairs = set(frozenset((a, b)) for a, b in combinations(branches, 2))
        if len(paths) != len(expected_pairs):
            return 0
        used_interior = set()
        for fkey, path in paths.items():
            if len(path) < 2:
                return 0
            a, b = path[0], path[-1]
            if a not in bset or b not in bset:
                return 0
            if a == b:
                return 0
            if fkey not in expected_pairs:
                return 0
            interior = set(path[1:-1])
            if interior & bset:
                return 0
            if interior & used_interior:
                return 0
            for i in range(len(path)-1):
                if path[i+1] not in adj_set[path[i]]:
                    return 0
            used_interior |= interior
        return len(branches)

    # Strategy: Try k from high to low
    # Candidate sets: Top k by degree
    degrees = [len(adj[v]) for v in range(n)]
    sorted_verts = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    best_count = 0
    best_result = None
    rng = random.Random(42)

    # Try k from 12 down to 2
    for k in range(12, 1, -1):
        if time_up():
            break
            
        # Generate a few candidate sets of size k
        candidates = []
        
        # 1. Top k degrees
        candidates.append(sorted_verts[:k])
        
        # 2. Random samples (if k is small enough to be fast)
        for _ in range(5):
            candidates.append(rng.sample(range(n), k))
        
        for cand in candidates:
            if time_up():
                break
            paths = try_find_subdivision(cand, rng)
            if paths is not None:
                cnt = validate(cand, paths)
                if cnt > best_count:
                    best_count = cnt
                    best_result = (cand, paths)
                    # If we found a valid subdivision for this k, we can stop looking for smaller k?
                    # No, we are iterating k downwards, so the first valid one is the max?
                    # Not necessarily, a random set of size k might fail while another set of size k succeeds.
                    # But if we find *any* set of size k, that's the best for this k.
                    # Since we iterate k downwards, the first k that yields a valid set is the max possible?
                    # Yes, because we want to maximize the number of branches.
                    # So if k=10 works, we don't need to check k=9.
                    # So we can return immediately?
                    # Wait, we should check if a larger k was missed.
                    # We iterate 12, 11, 10...
                    # So if 12 fails, we try 11. If 11 succeeds, we return 11.
                    # This is correct.
                    break
        
        if best_result is not None:
            break

    # Fallback if no result found (should not happen for k>=2)
    if best_result is None:
        u, v = edges[0]
        branches = [u, v]
        paths = {frozenset((u, v)): [u, v]}
        best_result = (branches, paths)

    branches, paths = best_result
    
    # Build final dict
    branch_list = [int(b) for b in sorted(branches)]
    path_list = []
    for a, b in combinations(sorted(branches), 2):
        fkey = frozenset((a, b))
        p = paths.get(fkey)
        if p is None:
            p = [a, b] # Should not happen if valid
        path_list.append([int(x) for x in p])

    return {'branches': branch_list, 'paths': path_list}

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    t_start = time.time()
    DEADLINE = 85.0

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    adj_set = adj

    def time_up():
        return time.time() - t_start > DEADLINE

    def find_path(u, v, banned, strategy='shortest'):
        """Find a path from u to v avoiding banned vertices.
        strategy: 'shortest' (BFS), 'avoid_dense' (prefer low-degree interiors)
        """
        if u == v:
            return None
        if v in adj_set[u]:
            return [u, v]
        
        if strategy == 'shortest':
            parent = {u: None}
            q = deque([u])
            while q:
                cur = q.popleft()
                for nxt in sorted(adj_set[cur]):
                    if nxt == v:
                        path = [v]
                        node = cur
                        while node is not None:
                            path.append(node)
                            node = parent[node]
                        path.reverse()
                        return path
                    if nxt in parent or nxt in banned:
                        continue
                    parent[nxt] = cur
                    q.append(nxt)
            return None
        elif strategy == 'avoid_dense':
            # BFS but prefer neighbors with lower degree
            parent = {u: None}
            q = deque([u])
            while q:
                cur = q.popleft()
                neighbors = [nxt for nxt in adj_set[cur] if nxt != v and nxt not in parent and nxt not in banned]
                neighbors.sort(key=lambda x: len(adj_set[x]))
                for nxt in neighbors:
                    parent[nxt] = cur
                    q.append(nxt)
                if v in parent:
                    path = [v]
                    node = v
                    while node != u:
                        node = parent[node]
                        path.append(node)
                    path.reverse()
                    return path
            return None

    def try_add_branch(branches, paths, new_branch, used_interior, bset):
        """Try to add new_branch to existing subdivision. Returns (new_paths, new_used_interior) or None."""
        new_bset = bset | {new_branch}
        new_pairs = list(combinations(sorted(new_bset), 2))
        
        # We need to find paths for all pairs involving new_branch
        # Existing paths stay the same; we just need paths from new_branch to each existing branch
        new_paths = dict(paths)
        new_used = set(used_interior)
        
        for b in branches:
            banned = new_used | new_bset
            banned.discard(new_branch)
            banned.discard(b)
            
            # Try multiple strategies
            found = False
            for strat in ['shortest', 'avoid_dense']:
                path = find_path(new_branch, b, banned, strat)
                if path is not None:
                    interior = set(path[1:-1])
                    if not (interior & new_bset) and not (interior & new_used):
                        new_paths[frozenset((new_branch, b))] = path
                        new_used |= interior
                        found = True
                        break
            
            if not found:
                return None
        
        return (new_paths, new_used)

    def validate(branches, paths):
        if not branches or len(branches) < 2:
            return 0
        bset = set(branches)
        if len(bset) != len(branches):
            return 0
        expected_pairs = set(frozenset((a, b)) for a, b in combinations(branches, 2))
        if len(paths) != len(expected_pairs):
            return 0
        used_interior = set()
        for fkey, path in paths.items():
            if len(path) < 2:
                return 0
            a, b = path[0], path[-1]
            if a not in bset or b not in bset:
                return 0
            if a == b:
                return 0
            if fkey not in expected_pairs:
                return 0
            interior = set(path[1:-1])
            if interior & bset:
                return 0
            if interior & used_interior:
                return 0
            for i in range(len(path)-1):
                if path[i+1] not in adj_set[path[i]]:
                    return 0
            used_interior |= interior
        return len(branches)

    degrees = [len(adj[v]) for v in range(n)]
    sorted_verts = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    best_count = 0
    best_result = None
    rng = random.Random(42)

    # Greedy incremental: start with 2 branches, try to grow
    # Try multiple starting pairs and growth orders
    for start_seed in range(20):
        if time_up():
            break
        
        # Choose starting pair
        if start_seed == 0:
            start_pair = (sorted_verts[0], sorted_verts[1])
        else:
            start_pair = tuple(rng.sample(range(n), 2))
        
        u, v = start_pair
        if u == v:
            continue
        
        # Initialize with just this pair
        if v in adj_set[u]:
            init_paths = {frozenset((u, v)): [u, v]}
            init_used = set()
        else:
            # Find a path between them
            path = find_path(u, v, set(), 'shortest')
            if path is None:
                continue
            init_paths = {frozenset((u, v)): path}
            init_used = set(path[1:-1])
        
        current_branches = [u, v]
        current_paths = init_paths
        current_used = init_used
        current_bset = set(current_branches)
        
        # Try to grow
        remaining = [x for x in range(n) if x not in current_bset]
        remaining.sort(key=lambda x: degrees[x], reverse=True)
        
        for attempt in range(len(remaining)):
            if time_up():
                break
            
            # Try adding the next candidate
            added = False
            for candidate in remaining[:10]:  # Try top 10 candidates
                if time_up():
                    break
                
                result = try_add_branch(current_branches, current_paths, candidate, current_used, current_bset)
                if result is not None:
                    new_paths, new_used = result
                    cnt = validate(current_branches + [candidate], new_paths)
                    if cnt > best_count:
                        best_count = cnt
                        best_result = (current_branches + [candidate], new_paths)
                    
                    # Accept the addition and continue
                    current_branches.append(candidate)
                    current_paths = new_paths
                    current_used = new_used
                    current_bset.add(candidate)
                    remaining.remove(candidate)
                    added = True
                    break
            
            if not added:
                break
    
    # Also try the original top-down approach as backup
    for k in range(12, 1, -1):
        if time_up():
            break
        if best_count >= k:
            break
            
        candidates = [sorted_verts[:k]]
        for _ in range(5):
            candidates.append(rng.sample(range(n), k))
            
        for cand in candidates:
            if time_up():
                break
            bset = set(cand)
            pairs = list(combinations(cand, 2))
            attempts = 10
            for _ in range(attempts):
                if time_up():
                    break
                rng.shuffle(pairs)
                used_interior = set()
                paths = {}
                success = True
                for a, b in pairs:
                    banned = used_interior | bset
                    banned.discard(a)
                    banned.discard(b)
                    path = find_path(a, b, banned, 'shortest')
                    if path is None:
                        success = False
                        break
                    interior = set(path[1:-1])
                    if interior & bset or interior & used_interior:
                        success = False
                        break
                    used_interior |= interior
                    paths[frozenset((a, b))] = path
                if success:
                    cnt = validate(cand, paths)
                    if cnt > best_count:
                        best_count = cnt
                        best_result = (cand, paths)
                    break
        
        if best_result is not None and best_count >= k:
            break

    if best_result is None:
        u, v = edges[0]
        branches = [u, v]
        paths = {frozenset((u, v)): [u, v]}
        best_result = (branches, paths)

    branches, paths = best_result
    
    branch_list = [int(b) for b in sorted(branches)]
    path_list = []
    for a, b in combinations(sorted(branches), 2):
        fkey = frozenset((a, b))
        p = paths.get(fkey)
        if p is None:
            p = [a, b]
        path_list.append([int(x) for x in p])

    return {'branches': branch_list, 'paths': path_list}
# EVOLVE-BLOCK-END

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    deadline = time.time() + 55.0
    best_branches = None
    best_paths = None
    
    def find_max_clique():
        # Bron-Kerbosch with pivot, optimized for speed
        best = []
        best_size = 0
        adj_list = [sorted(adj[i]) for i in range(n)]
        
        def bk(r, p, x):
            nonlocal best, best_size
            if time.time() > deadline:
                return
            if not p and not x:
                if len(r) > best_size:
                    best_size = len(r)
                    best = list(r)
                return
            # Upper bound pruning
            if len(r) + len(p) <= best_size:
                return
            # Pivot
            if p:
                candidates = p | x
            else:
                candidates = x
            if candidates:
                pivot = max(candidates, key=lambda v: len(adj[v] & p))
                p_iter = [v for v in p if pivot not in adj[v]]
            else:
                p_iter = list(p)
            for v in p_iter:
                bk(r | {v}, p & adj[v], x & adj[v])
                p.remove(v)
                x.add(v)
        bk(set(), set(range(n)), set())
        return best
    
    def bfs_path(src, dst, forbidden):
        """Find shortest path from src to dst avoiding forbidden interior vertices."""
        if src == dst:
            return None
        if dst in forbidden and dst != src and dst != dst:
            pass
        parent = {src: None}
        queue = deque([src])
        while queue:
            u = queue.popleft()
            if u == dst:
                # Reconstruct path
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            for w in adj[u]:
                if w not in parent and w != dst and w not in forbidden:
                    parent[w] = u
                    queue.append(w)
        return None
    
    def validate_config(branches, paths):
        """Validate that the configuration is correct."""
        bset = set(branches)
        if len(bset) != len(branches):
            return False
        if len(branches) < 2:
            return False
        # Check all pairs have a path
        pairs = set()
        for path in paths:
            if len(path) < 2:
                return False
            s, d = path[0], path[-1]
            if s not in bset or d not in bset:
                return False
            if s == d:
                return False
            pair = frozenset((s, d))
            if pair in pairs:
                return False
            pairs.add(pair)
            # Check path is simple
            seen = set()
            for v in path:
                if v in seen:
                    return False
                seen.add(v)
            # Check consecutive vertices are adjacent
            for i in range(len(path) - 1):
                if path[i+1] not in adj[path[i]]:
                    return False
            # Check interior vertices are not branches
            for v in path[1:-1]:
                if v in bset:
                    return False
        # Check all pairs covered
        for a, b in combinations(branches, 2):
            if frozenset((a, b)) not in pairs:
                return False
        # Check interiors are pairwise disjoint
        interior_sets = []
        for path in paths:
            interior_sets.append(set(path[1:-1]))
        for i in range(len(interior_sets)):
            for j in range(i+1, len(interior_sets)):
                if interior_sets[i] & interior_sets[j]:
                    return False
        return True
    
    def try_add_vertex(branches, paths, new_v):
        """Try to add new_v to the branch set. Returns (new_branches, new_paths) or None."""
        bset = set(branches)
        if new_v in bset:
            return None
        # Find paths from new_v to each existing branch, avoiding all branch vertices and previously used interiors
        forbidden = set(bset) - {new_v}
        used_interiors = set()
        for path in paths:
            used_interiors.update(path[1:-1])
        forbidden |= used_interiors
        
        new_paths = list(paths)
        new_interiors_set = set()
        
        # Sort branches by distance from new_v to find easier paths first
        def dist_to_branch(b):
            if b in adj[new_v]:
                return 0
            return 1
        
        sorted_branches = sorted(branches, key=dist_to_branch)
        
        for b in sorted_branches:
            path = bfs_path(new_v, b, forbidden)
            if path is None:
                return None
            # Check this path's interiors don't conflict with other new paths
            new_interiors = set(path[1:-1])
            if new_interiors & forbidden:
                return None
            # Check against other new paths added in this round
            if new_interiors & new_interiors_set:
                return None
            new_paths.append(path)
            forbidden |= new_interiors
            new_interiors_set |= new_interiors
        
        return (branches + [new_v], new_paths)
    
    def greedy_grow(start_clique):
        """Greedy: start from a clique, try to add vertices one by one."""
        branches = list(start_clique)
        paths = []
        for a, b in combinations(branches, 2):
            paths.append([a, b])  # Direct edge since it's a clique
        
        # Try to add remaining vertices
        remaining = [v for v in range(n) if v not in set(branches)]
        random.shuffle(remaining)
        
        # Multiple passes to keep trying to add vertices
        for _pass in range(3):
            if time.time() > deadline:
                break
            added = False
            for v in remaining:
                if time.time() > deadline:
                    break
                if v in set(branches):
                    continue
                result = try_add_vertex(branches, paths, v)
                if result is not None:
                    branches, paths = result
                    remaining.remove(v)
                    added = True
            if not added:
                break
        
        return branches, paths
    
    def local_search(branches, paths):
        """Try to swap out vertices for better ones."""
        bset = set(branches)
        remaining = [v for v in range(n) if v not in bset]
        random.shuffle(remaining)
        
        for v in remaining:
            if time.time() > deadline:
                break
            result = try_add_vertex(branches, paths, v)
            if result is not None:
                branches, paths = result
        return branches, paths
    
    # Phase 1: Find max clique
    max_clique = find_max_clique()
    if not max_clique:
        max_clique = [0, 1]
    
    # Phase 2: Multiple randomized attempts
    best_size = 0
    num_attempts = 500
    for attempt in range(num_attempts):
        if time.time() > deadline:
            break
        
        # Randomize the order of trying to add vertices
        start = list(max_clique)
        random.shuffle(start)
        
        branches, paths = greedy_grow(start)
        
        if len(branches) > best_size:
            best_size = len(branches)
            best_branches = list(branches)
            best_paths = [list(p) for p in paths]
        
        # Try local search
        branches2, paths2 = local_search(branches, paths)
        if len(branches2) > best_size:
            best_size = len(branches2)
            best_branches = list(branches2)
            best_paths = [list(p) for p in paths2]
        
        # Also try starting from smaller cliques with different orderings
        if attempt % 5 == 0:
            # Try a random subset of size 2 or 3
            size = random.choice([2, 3, 4])
            subset = random.sample(range(n), min(size, n))
            # Check if it forms a valid starting config (all pairs need paths)
            valid_start = True
            temp_paths = []
            used = set()
            for a, b in combinations(subset, 2):
                p = bfs_path(a, b, set(subset) - {a, b})
                if p is None:
                    valid_start = False
                    break
                new_int = set(p[1:-1])
                if new_int & used:
                    valid_start = False
                    break
                used |= new_int
                temp_paths.append(p)
            if valid_start and len(subset) >= 2:
                branches, paths = greedy_grow(subset)
                if len(branches) > best_size:
                    best_size = len(branches)
                    best_branches = list(branches)
                    best_paths = [list(p) for p in paths]
    
    # Final validation
    if best_branches is not None and len(best_branches) >= 2:
        if validate_config(best_branches, best_paths):
            # Convert to proper types
            branches_out = [int(v) for v in best_branches]
            paths_out = [[int(v) for v in p] for p in best_paths]
            return {'branches': branches_out, 'paths': paths_out}
    
    # Fallback: return the max clique
    if len(max_clique) >= 2:
        branches_out = [int(v) for v in max_clique]
        paths_out = [[int(a), int(b)] for a, b in combinations(max_clique, 2)]
        return {'branches': branches_out, 'paths': paths_out}
    
    # Last resort
    u, v = P['edges'][0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

# EVOLVE-BLOCK-END

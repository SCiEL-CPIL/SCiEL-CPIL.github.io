"""Randomized search with diverse heuristics to find long simple path in pinned graph."""
import random
import time
from collections import deque

N = 100
EDGES = [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]]

ADJ = [set() for _ in range(N)]
DEG = [0] * N
for u, v in EDGES:
    ADJ[u].add(v)
    ADJ[v].add(u)
    DEG[u] += 1
    DEG[v] += 1

# Find connected components
def find_components():
    visited = set()
    comps = []
    for s in range(N):
        if s not in visited:
            comp = set()
            stack = [s]
            while stack:
                u = stack.pop()
                if u in visited:
                    continue
                visited.add(u)
                comp.add(u)
                for v in ADJ[u]:
                    if v not in visited:
                        stack.append(v)
            comps.append(comp)
    return comps

def greedy_warnsdorff(start, rng):
    """Warnsdorff's rule: pick neighbor with fewest unvisited neighbors."""
    seen = {start}
    path = [start]
    while True:
        cand = ADJ[path[-1]] - seen
        if not cand:
            break
        v = min(cand, key=lambda w: (len(ADJ[w] - seen), rng.random()))
        path.append(v)
        seen.add(v)
    return path

def greedy_max_degree(start, rng):
    """Pick neighbor with highest degree."""
    seen = {start}
    path = [start]
    while True:
        cand = ADJ[path[-1]] - seen
        if not cand:
            break
        v = max(cand, key=lambda w: (DEG[w], rng.random()))
        path.append(v)
        seen.add(v)
    return path

def greedy_random(start, rng):
    """Random neighbor choice."""
    seen = {start}
    path = [start]
    while True:
        cand = list(ADJ[path[-1]] - seen)
        if not cand:
            break
        v = rng.choice(cand)
        path.append(v)
        seen.add(v)
    return path

def greedy_mixed(start, rng):
    """Mixed heuristic: combine Warnsdorff and degree."""
    seen = {start}
    path = [start]
    while True:
        cand = ADJ[path[-1]] - seen
        if not cand:
            break
        weights = rng.random() * 2
        v = max(cand, key=lambda w: (-len(ADJ[w] - seen), DEG[w] * weights, rng.random()))
        path.append(v)
        seen.add(v)
    return path

def extend_both_ends(path, seen):
    """Extend path from both ends using Warnsdorff."""
    path = list(path)
    seen = set(seen)
    improved = True
    while improved:
        improved = False
        # Extend front
        while True:
            cand = ADJ[path[0]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: (len(ADJ[w] - seen), w))
            path.insert(0, v)
            seen.add(v)
            improved = True
        # Extend back
        while True:
            cand = ADJ[path[-1]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: (len(ADJ[w] - seen), w))
            path.append(v)
            seen.add(v)
            improved = True
    return path

def local_improve(path, seen, rng, deadline):
    """Try to insert unvisited vertices into the path using 2-opt style moves."""
    path = list(path)
    seen = set(seen)
    unvisited = set(range(N)) - seen
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try inserting each unvisited vertex
        for u in list(unvisited):
            for i in range(len(path) - 1):
                if path[i] in ADJ[u] and path[i+1] in ADJ[u]:
                    path.insert(i + 1, u)
                    seen.add(u)
                    unvisited.remove(u)
                    improved = True
                    break
            if improved:
                break
        # Also try extending ends
        while True:
            cand = ADJ[path[0]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: (len(ADJ[w] - seen), w))
            path.insert(0, v)
            seen.add(v)
            improved = True
        while True:
            cand = ADJ[path[-1]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: (len(ADJ[w] - seen), w))
            path.append(v)
            seen.add(v)
            improved = True
    return path

def two_opt_improve(path, seen, rng, deadline):
    """2-opt style: try reversing segments and inserting unvisited vertices."""
    path = list(path)
    seen = set(seen)
    unvisited = set(range(N)) - seen
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try inserting each unvisited vertex at any position
        for u in list(unvisited):
            best_pos = -1
            for i in range(len(path) - 1):
                if path[i] in ADJ[u] and path[i+1] in ADJ[u]:
                    best_pos = i
                    break
            if best_pos >= 0:
                path.insert(best_pos + 1, u)
                seen.add(u)
                unvisited.remove(u)
                improved = True
                continue
            # Try extending ends
            if path[0] in ADJ[u]:
                path.insert(0, u)
                seen.add(u)
                unvisited.remove(u)
                improved = True
                continue
            if path[-1] in ADJ[u]:
                path.append(u)
                seen.add(u)
                unvisited.remove(u)
                improved = True
                continue
        # Try 2-opt: reverse a segment
        n = len(path)
        for i in range(n - 1):
            for j in range(i + 2, n):
                # Check if reversing path[i+1:j+1] creates valid edges
                # Original: ... path[i] - path[i+1] ... path[j] - path[j+1] ...
                # After reverse: ... path[i] - path[j] ... path[i+1] - path[j+1] ...
                if j + 1 < n:
                    if path[i] in ADJ[path[j]] and path[i+1] in ADJ[path[j+1]]:
                        # Check internal edges of reversed segment are valid (they are, since undirected)
                        path[i+1:j+1] = path[i+1:j+1][::-1]
                        improved = True
                        break
                else:
                    if path[i] in ADJ[path[j]]:
                        path[i+1:j+1] = path[i+1:j+1][::-1]
                        improved = True
                        break
            if improved:
                break
    return path

def dfs_search(start, seen, path, best, deadline, max_depth):
    """DFS to find longer path from current state with pruning."""
    if time.time() > deadline:
        return
    if len(path) > len(best):
        best.clear()
        best.extend(path)
    if len(path) >= max_depth:
        return
    # Pruning: if we can't possibly beat best, return
    remaining = max_depth - len(path)
    if remaining <= 0:
        return
    cand = ADJ[path[-1]] - seen
    # Warnsdorff ordering
    cand = sorted(cand, key=lambda w: (len(ADJ[w] - seen), w))
    for v in cand:
        path.append(v)
        seen.add(v)
        dfs_search(start, seen, path, best, deadline, max_depth)
        path.pop()
        seen.remove(v)
        if time.time() > deadline:
            return
        if len(best) >= max_depth:
            return

def matching_seed(comp_vertices, deadline):
    """Use vertex-splitting to find a matching-based path seed."""
    # Create bipartite graph: for each vertex v, create v_in and v_out
    # Edge u_out -> v_in if (u,v) is an edge in original graph
    # Find maximum matching, then extract paths
    vertices = sorted(comp_vertices)
    n = len(vertices)
    idx = {v: i for i, v in enumerate(vertices)}
    
    # Build adjacency for matching
    # Left side: v_out for each v, Right side: v_in for each v
    # Edge from u_out to v_in if u-v is an edge
    left_adj = [set() for _ in range(n)]
    for u in vertices:
        for v in ADJ[u]:
            if v in idx:
                left_adj[idx[u]].add(idx[v])
    
    # Hopcroft-Karp
    pair_u = [-1] * n
    pair_v = [-1] * n
    dist = [0] * n
    
    def bfs():
        from collections import deque
        q = deque()
        for u in range(n):
            if pair_u[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = float('inf')
        found = False
        while q:
            u = q.popleft()
            for v in left_adj[u]:
                pu = pair_v[v]
                if pu == -1:
                    found = True
                elif dist[pu] == float('inf'):
                    dist[pu] = dist[u] + 1
                    q.append(pu)
        return found
    
    def dfs(u):
        for v in left_adj[u]:
            pu = pair_v[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = float('inf')
        return False
    
    matching_size = 0
    while time.time() < deadline:
        if not bfs():
            break
        for u in range(n):
            if pair_u[u] == -1:
                if dfs(u):
                    matching_size += 1
    
    # Extract paths from matching
    # pair_u[u] = v means edge u->v in matching (u_out to v_in)
    # Build directed graph from matching
    next_node = {}
    for u in range(n):
        if pair_u[u] != -1:
            next_node[vertices[u]] = vertices[pair_u[u]]
    
    # Find paths
    visited = set()
    paths = []
    for start in vertices:
        if start not in visited and start not in next_node:
            # This is a path start (no incoming edge in matching)
            path = [start]
            visited.add(start)
            cur = start
            while cur in next_node:
                cur = next_node[cur]
                if cur in visited:
                    break
                path.append(cur)
                visited.add(cur)
            if len(path) > 1:
                paths.append(path)
    
    # Return longest path
    if not paths:
        return list(vertices)
    paths.sort(key=len, reverse=True)
    return paths[0]

def construct():
    t0 = time.time()
    deadline = t0 + 85  # Safety margin before 90s kill
    
    comps = find_components()
    comps.sort(key=len, reverse=True)
    
    best_path = []
    rng = random.Random(42)
    
    for comp in comps:
        if time.time() > deadline:
            break
        comp_size = len(comp)
        comp_list = sorted(comp)
        target = comp_size  # We want to find a Hamiltonian path in the component
        
        # Phase 0: Matching-based seed
        if time.time() < deadline - 5:
            match_seed = matching_seed(comp_list, time.time() + 2)
            match_seed = extend_both_ends(match_seed, set(match_seed))
            if len(match_seed) > len(best_path):
                best_path = match_seed[:]
        
        # Phase 1: Diverse greedy starts
        for start in comp_list:
            if time.time() > deadline - 10:
                break
            for func in [greedy_warnsdorff, greedy_max_degree, greedy_random, greedy_mixed]:
                if time.time() > deadline - 10:
                    break
                p = func(start, rng)
                p = extend_both_ends(p, set(p))
                if len(p) > len(best_path):
                    best_path = p[:]
        
        # Phase 2: Local improvement on best paths found
        if len(best_path) > 0 and len(best_path) < target:
            if time.time() < deadline - 5:
                p = local_improve(best_path, set(best_path), rng, deadline - 5)
                if len(p) > len(best_path):
                    best_path = p
        
        # Phase 2.5: 2-opt improvement
        if len(best_path) > 0 and len(best_path) < target:
            if time.time() < deadline - 5:
                p = two_opt_improve(best_path, set(best_path), rng, deadline - 5)
                if len(p) > len(best_path):
                    best_path = p
        
        # Phase 3: If not Hamiltonian, try DFS from promising starts
        if len(best_path) < target:
            # Try DFS from endpoints of current best and from low-degree vertices
            promising_starts = set()
            if best_path:
                promising_starts.add(best_path[0])
                promising_starts.add(best_path[-1])
            # Add some low-degree vertices in component
            for v in comp_list:
                if v in ADJ:
                    pass
            # Just try all vertices with limited time
            for start in comp_list:
                if time.time() > deadline - 2:
                    break
                budget = min(1.5, (deadline - time.time()) * 0.3)
                dl = time.time() + budget
                local_best = list(best_path)
                dfs_search(start, {start}, [start], local_best, dl, target)
                if len(local_best) > len(best_path):
                    best_path = local_best
                    if len(best_path) >= target:
                        break
    
    # Final safety check
    if not best_path:
        best_path = [0]
    
    # Verify path validity
    for i in range(len(best_path) - 1):
        if best_path[i+1] not in ADJ[best_path[i]]:
            return [0]
    if len(set(best_path)) != len(best_path):
        seen = set()
        fixed = []
        for v in best_path:
            if v not in seen:
                fixed.append(v)
                seen.add(v)
        return fixed
    
    return best_path

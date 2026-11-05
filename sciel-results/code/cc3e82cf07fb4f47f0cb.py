import time
import random
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_components(adj, n):
    visited = [False] * n
    components = []
    for i in range(n):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                node = stack.pop()
                comp.append(node)
                for nb in adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            components.append(comp)
    return components

def _is_tree(adj, comp):
    """Check if component is a tree (edges = vertices - 1)."""
    comp_set = set(comp)
    edge_count = sum(len(adj[u] & comp_set) for u in comp) // 2
    return edge_count == len(comp) - 1

def _tree_diameter_path(adj, comp):
    """Find the longest path in a tree component using 2-BFS."""
    if len(comp) == 1:
        return list(comp)
    comp_set = set(comp)
    start = comp[0]
    # First BFS to find farthest node
    dist = {start: 0}
    parent = {start: None}
    q = deque([start])
    farthest = start
    while q:
        u = q.popleft()
        if dist[u] > dist[farthest]:
            farthest = u
        for v in adj[u]:
            if v in comp_set and v not in dist:
                dist[v] = dist[u] + 1
                parent[v] = u
                q.append(v)
    # Second BFS from farthest
    dist2 = {farthest: 0}
    parent2 = {farthest: None}
    q = deque([farthest])
    farthest2 = farthest
    while q:
        u = q.popleft()
        if dist2[u] > dist2[farthest2]:
            farthest2 = u
        for v in adj[u]:
            if v in comp_set and v not in dist2:
                dist2[v] = dist2[u] + 1
                parent2[v] = u
                q.append(v)
    # Reconstruct path
    path = []
    cur = farthest2
    while cur is not None:
        path.append(cur)
        cur = parent2[cur]
    path.reverse()
    return path

def _greedy_path(adj, start, n, max_steps=200):
    """Greedy path: at each step go to neighbor with fewest unvisited neighbors."""
    path = [start]
    visited = {start}
    steps = 0
    while steps < max_steps:
        cur = path[-1]
        cands = [nb for nb in adj[cur] if nb not in visited]
        if not cands:
            break
        cands.sort(key=lambda x: (len(adj[x] - visited), x))
        nxt = cands[0]
        path.append(nxt)
        visited.add(nxt)
        steps += 1
    return path

def _try_extend_path(adj, path, n, deadline):
    """Try to extend the path from both ends. Returns improved path."""
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        path_set = set(path)
        # Try extending the end
        end = path[-1]
        cands = [nb for nb in adj[end] if nb not in path_set]
        if cands:
            cands.sort(key=lambda x: len(adj[x] & path_set))
            path.append(cands[0])
            improved = True
            continue
        # Try extending the beginning
        start = path[0]
        cands = [nb for nb in adj[start] if nb not in path_set]
        if cands:
            cands.sort(key=lambda x: len(adj[x] & path_set))
            path.insert(0, cands[0])
            improved = True
            continue
    return path

def _insert_vertex(adj, path, vertex, n):
    """Try to insert vertex into the path. Returns new path or None."""
    if vertex in set(path):
        return None
    # Check if vertex can be inserted at the start
    if vertex in adj[path[0]]:
        return [vertex] + path
    # Check if vertex can be inserted at the end
    if vertex in adj[path[-1]]:
        return path + [vertex]
    # Check internal insertion
    for i in range(len(path) - 1):
        a, b = path[i], path[i+1]
        if vertex in adj[a] and vertex in adj[b]:
            new_path = path[:i+1] + [vertex] + path[i+1:]
            return new_path
    return None

def _2opt_improve(adj, path, n, deadline):
    """Try 2-opt style reversals to find a longer path."""
    best_path = path[:]
    best_len = len(best_path)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best_path) - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(best_path)):
                if time.time() > deadline:
                    break
                if i > 0:
                    if best_path[j] not in adj[best_path[i-1]]:
                        continue
                if j + 1 < len(best_path):
                    if best_path[i] not in adj[best_path[j+1]]:
                        continue
                
                new_path = best_path[:i] + best_path[i:j+1][::-1] + best_path[j+1:]
                
                valid = True
                for k in range(len(new_path) - 1):
                    if new_path[k+1] not in adj[new_path[k]]:
                        valid = False
                        break
                if not valid:
                    continue
                
                new_path = _try_extend_path(adj, new_path, n, deadline)
                
                if len(new_path) > best_len:
                    best_path = new_path
                    best_len = len(best_path)
                    improved = True
                    break
            if improved:
                break
    return best_path

def _limited_dfs(adj, start, n, deadline, max_depth=30):
    """Limited DFS to find a long path from start."""
    best_path = [start]
    best_len = 1
    
    def dfs(path, visited):
        nonlocal best_path, best_len
        if time.time() > deadline:
            return
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
        if len(path) >= max_depth:
            return
        cur = path[-1]
        cands = [nb for nb in adj[cur] if nb not in visited]
        cands.sort(key=lambda x: (len(adj[x] - visited), x))
        for nxt in cands:
            if time.time() > deadline:
                return
            path.append(nxt)
            visited.add(nxt)
            dfs(path, visited)
            path.pop()
            visited.remove(nxt)
    
    dfs([start], {start})
    return best_path

def _multi_start_search(adj, n, deadline):
    """Multiple random starts with greedy + local search."""
    best_path = []
    best_len = 0
    rng = random.Random(748)
    
    for start in range(n):
        if time.time() > deadline:
            break
        path = _greedy_path(adj, start, n)
        path = _try_extend_path(adj, path, n, deadline)
        if len(path) > best_len:
            best_path = path
            best_len = len(path)
    
    starts = list(range(n))
    rng.shuffle(starts)
    for start in starts[:50]:
        if time.time() > deadline:
            break
        path = _greedy_path(adj, start, n)
        path = _try_extend_path(adj, path, n, deadline)
        if len(path) > best_len:
            best_path = path
            best_len = len(path)
    
    return best_path

def _dfs_enhanced(adj, n, deadline):
    """DFS from multiple starts with limited depth."""
    best_path = []
    best_len = 0
    for start in range(n):
        if time.time() > deadline:
            break
        path = _limited_dfs(adj, start, n, deadline, max_depth=25)
        path = _try_extend_path(adj, path, n, deadline)
        if len(path) > best_len:
            best_path = path
            best_len = len(path)
    return best_path

def _branch_and_bound(adj, comp, deadline):
    """Exact longest path in a small component using DFS with branch and bound."""
    comp_set = set(comp)
    n_comp = len(comp)
    if n_comp <= 1:
        return list(comp)
    
    best_path = [comp[0]]
    best_len = 1
    
    # Try from each vertex as start
    for start in comp:
        if time.time() > deadline:
            break
        
        def dfs(path, visited):
            nonlocal best_path, best_len
            if time.time() > deadline:
                return
            if len(path) > best_len:
                best_len = len(path)
                best_path = path[:]
            if len(path) == n_comp:
                return
            cur = path[-1]
            cands = [nb for nb in adj[cur] if nb not in visited and nb in comp_set]
            # Sort by degree to explore more constrained nodes first
            cands.sort(key=lambda x: (len(adj[x] - visited), x))
            for nxt in cands:
                if time.time() > deadline:
                    return
                path.append(nxt)
                visited.add(nxt)
                dfs(path, visited)
                path.pop()
                visited.remove(nxt)
        
        dfs([start], {start})
    
    return best_path

def _solve_component(adj, comp, deadline):
    """Solve a single component for its longest path."""
    n_comp = len(comp)
    if n_comp == 1:
        return list(comp)
    
    if _is_tree(adj, comp):
        return _tree_diameter_path(adj, comp)
    
    # For small cyclic components, try exact DFS
    if n_comp <= 20:
        exact = _branch_and_bound(adj, comp, deadline)
        if len(exact) == n_comp:
            return exact
        # Fall through to heuristic if exact didn't find Hamiltonian path
    
    # For larger or unsolved components, use heuristic
    best_path = []
    for start in comp:
        if time.time() > deadline:
            break
        path = _greedy_path(adj, start, P['n'])
        path = _try_extend_path(adj, path, P['n'], deadline)
        if len(path) > len(best_path):
            best_path = path
    
    return best_path

def construct():
    n = P['n']
    adj = _build_adj()
    
    # Use 85s deadline to be safe (evaluator kills at 90s)
    deadline = time.time() + 85.0
    
    # Find components
    components = _find_components(adj, n)
    components.sort(key=len, reverse=True)
    
    # Solve each component
    best_path = []
    for comp in components:
        if time.time() > deadline:
            break
        comp_path = _solve_component(adj, comp, deadline)
        if len(comp_path) > len(best_path):
            best_path = comp_path
    
    # Optimize: extend, 2-opt, insert
    best_path = _try_extend_path(adj, best_path, n, deadline)
    best_path = _2opt_improve(adj, best_path, n, deadline)
    
    # Iterative improvement loop
    iteration = 0
    while time.time() < deadline - 3.0:
        iteration += 1
        path_set = set(best_path)
        inserted = False
        for v in range(n):
            if v not in path_set:
                new_path = _insert_vertex(adj, best_path, v, n)
                if new_path is not None:
                    best_path = new_path
                    path_set = set(best_path)
                    inserted = True
                    break
        if not inserted:
            new_best = _2opt_improve(adj, best_path, n, deadline)
            if len(new_best) > len(best_path):
                best_path = new_best
            else:
                # Try fresh multi-start search
                fresh = _multi_start_search(adj, n, deadline)
                fresh = _try_extend_path(adj, fresh, n, deadline)
                fresh = _2opt_improve(adj, fresh, n, deadline)
                if len(fresh) > len(best_path):
                    best_path = fresh
                else:
                    # Try DFS-enhanced
                    dfs_best = _dfs_enhanced(adj, n, deadline)
                    dfs_best = _try_extend_path(adj, dfs_best, n, deadline)
                    dfs_best = _2opt_improve(adj, dfs_best, n, deadline)
                    if len(dfs_best) > len(best_path):
                        best_path = dfs_best
                    else:
                        break
        else:
            best_path = _try_extend_path(adj, best_path, n, deadline)
            best_path = _2opt_improve(adj, best_path, n, deadline)
    
    # Final verification
    if len(best_path) == 0:
        best_path = [0]
    
    for i in range(len(best_path) - 1):
        assert best_path[i+1] in adj[best_path[i]], f"Invalid edge at position {i}"
    assert len(set(best_path)) == len(best_path), "Path has repeated vertices"
    
    return best_path
# EVOLVE-BLOCK-END

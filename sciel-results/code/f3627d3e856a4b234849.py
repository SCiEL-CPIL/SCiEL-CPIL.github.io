"""Longest path in pinned G(100, 1.8/100) via bitmask B&B + greedy + local repair."""
import time
import random
from collections import deque

# Pinned graph data
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
adj = [set() for _ in range(N)]
for u, v in P['edges']:
    adj[u].add(v)
    adj[v].add(u)
adj_list = [sorted(adj[i]) for i in range(N)]
adj_set = adj

def _find_components():
    visited = [False] * N
    comps = []
    for i in range(N):
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
            comps.append(comp)
    return comps

def _greedy_extend(path, seen, rng):
    while True:
        cur = path[-1]
        cands = [v for v in adj_list[cur] if v not in seen]
        if not cands:
            break
        cands.sort(key=lambda w: (len([x for x in adj_list[w] if x not in seen]), rng.random()))
        v = cands[0]
        path.append(v)
        seen.add(v)
    return path

def _greedy_extend_det(path, seen):
    while True:
        cur = path[-1]
        cands = [v for v in adj_list[cur] if v not in seen]
        if not cands:
            break
        cands.sort(key=lambda w: (len([x for x in adj_list[w] if x not in seen]), w))
        v = cands[0]
        path.append(v)
        seen.add(v)
    return path

def _extend_both_ends(path):
    """Try to extend path from both ends greedily."""
    visited = set(path)
    changed = True
    while changed:
        changed = False
        # Extend from start
        cur = path[0]
        cands = [v for v in adj_list[cur] if v not in visited]
        if cands:
            cands.sort(key=lambda w: (len([x for x in adj_list[w] if x not in visited]), w))
            v = cands[0]
            path = [v] + path
            visited.add(v)
            changed = True
        # Extend from end
        cur = path[-1]
        cands = [v for v in adj_list[cur] if v not in visited]
        if cands:
            cands.sort(key=lambda w: (len([x for x in adj_list[w] if x not in visited]), w))
            v = cands[0]
            path = path + [v]
            visited.add(v)
            changed = True
    return path

def _local_repair(path, max_restarts=50):
    """Try to improve path by removing a vertex and re-inserting it elsewhere,
    or by swapping endpoints to find longer extensions."""
    best = path[:]
    best_len = len(best)
    
    for restart in range(max_restarts):
        cur_path = best[:]
        visited = set(cur_path)
        
        # Try removing a random internal vertex and re-extending
        if len(cur_path) > 3:
            idx = random.Random(restart * 1000 + 7).randint(1, len(cur_path) - 2)
            removed = cur_path[idx]
            cur_path = cur_path[:idx] + cur_path[idx+1:]
            visited.discard(removed)
            
            # Re-extend from both ends
            cur_path = _extend_both_ends(cur_path)
            
            # Try to insert removed vertex somewhere
            inserted = False
            for i in range(len(cur_path)):
                if i > 0 and cur_path[i-1] in adj_set[removed] and removed in adj_set[cur_path[i+1]] if i+1 < len(cur_path) else False:
                    cur_path = cur_path[:i] + [removed] + cur_path[i:]
                    inserted = True
                    break
            # Also try at ends
            if not inserted:
                if removed in adj_set[cur_path[0]]:
                    cur_path = [removed] + cur_path
                    inserted = True
                elif removed in adj_set[cur_path[-1]]:
                    cur_path = cur_path + [removed]
                    inserted = True
            
            # Extend again
            cur_path = _extend_both_ends(cur_path)
            
            if len(cur_path) > best_len:
                best = cur_path
                best_len = len(cur_path)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Find components
    components = _find_components()
    largest_comp = max(components, key=len)
    comp_size = len(largest_comp)
    comp_set = set(largest_comp)
    
    # If largest component is small, just do exhaustive
    if comp_size <= 1:
        return [largest_comp[0]] if largest_comp else [0]
    
    # Map to local indices
    idx_map = {v: i for i, v in enumerate(largest_comp)}
    
    # Build local adjacency
    local_adj = [set() for _ in range(comp_size)]
    for u in largest_comp:
        ui = idx_map[u]
        for v in adj[u]:
            if v in comp_set:
                vi = idx_map[v]
                local_adj[ui].add(vi)
    
    local_adj_list = [sorted(local_adj[i]) for i in range(comp_size)]
    local_adj_set = local_adj
    
    # Bitmask adjacency
    adj_bits = [0] * comp_size
    for i in range(comp_size):
        for j in local_adj[i]:
            adj_bits[i] |= (1 << j)
    
    full_mask = (1 << comp_size) - 1
    
    best_path = []
    best_len = 0
    
    def update_best(path_local):
        nonlocal best_path, best_len
        if len(path_local) > best_len:
            best_len = len(path_local)
            best_path = list(path_local)
    
    def time_up():
        return time.time() > deadline
    
    # Phase 1: Greedy from all starts
    for start in range(comp_size):
        if time_up():
            break
        # Deterministic greedy
        path = [start]
        seen = {start}
        while True:
            cur = path[-1]
            cands = [v for v in local_adj_list[cur] if v not in seen]
            if not cands:
                break
            cands.sort(key=lambda w: (len([x for x in local_adj_list[w] if x not in seen]), w))
            v = cands[0]
            path.append(v)
            seen.add(v)
        # Extend both ends
        visited = set(path)
        changed = True
        while changed:
            changed = False
            cur = path[0]
            cands = [v for v in local_adj_list[cur] if v not in visited]
            if cands:
                cands.sort(key=lambda w: (len([x for x in local_adj_list[w] if x not in visited]), w))
                v = cands[0]
                path = [v] + path
                visited.add(v)
                changed = True
            cur = path[-1]
            cands = [v for v in local_adj_list[cur] if v not in visited]
            if cands:
                cands.sort(key=lambda w: (len([x for x in local_adj_list[w] if x not in visited]), w))
                v = cands[0]
                path = path + [v]
                visited.add(v)
                changed = True
        update_best(path)
    
    if best_len == comp_size:
        return [largest_comp[i] for i in best_path]
    
    # Phase 2: Randomized greedy with many trials
    rng = random.Random(748)
    for trial in range(500):
        if time_up():
            break
        start = rng.randint(0, comp_size - 1)
        path = [start]
        seen = {start}
        while True:
            cur = path[-1]
            cands = [v for v in local_adj_list[cur] if v not in seen]
            if not cands:
                break
            cands.sort(key=lambda w: (len([x for x in local_adj_list[w] if x not in seen]), rng.random()))
            v = cands[0]
            path.append(v)
            seen.add(v)
        # Extend both ends
        visited = set(path)
        changed = True
        while changed:
            changed = False
            cur = path[0]
            cands = [v for v in local_adj_list[cur] if v not in visited]
            if cands:
                v = rng.choice(cands)
                path = [v] + path
                visited.add(v)
                changed = True
            cur = path[-1]
            cands = [v for v in local_adj_list[cur] if v not in visited]
            if cands:
                v = rng.choice(cands)
                path = path + [v]
                visited.add(v)
                changed = True
        update_best(path)
    
    if best_len == comp_size:
        return [largest_comp[i] for i in best_path]
    
    # Phase 3: Bitmask DFS with branch and bound
    # Upper bound: current path length + number of unvisited vertices reachable from either endpoint
    def upper_bound(path, visited_mask):
        unvisited = full_mask & ~visited_mask
        if unvisited == 0:
            return len(path)
        e1 = path[0]
        e2 = path[-1]
        
        # BFS from e1 in unvisited subgraph
        reachable1 = 0
        if adj_bits[e1] & unvisited:
            frontier = adj_bits[e1] & unvisited
            visited1 = 0
            while frontier:
                visited1 |= frontier
                # Expand all nodes in frontier
                nbrs = 0
                f = frontier
                while f:
                    bit = f & (-f)
                    idx = bit.bit_length() - 1
                    nbrs |= adj_bits[idx]
                    f &= f - 1
                frontier = nbrs & unvisited & ~visited1
        
        # BFS from e2 in unvisited subgraph (excluding those reached from e1 to avoid double count)
        remaining = unvisited & ~visited1
        reachable2 = 0
        if adj_bits[e2] & remaining:
            frontier = adj_bits[e2] & remaining
            visited2 = 0
            while frontier:
                visited2 |= frontier
                nbrs = 0
                f = frontier
                while f:
                    bit = f & (-f)
                    idx = bit.bit_length() - 1
                    nbrs |= adj_bits[idx]
                    f &= f - 1
                frontier = nbrs & remaining & ~visited2
        
        return len(path) + visited1.bit_count() + visited2.bit_count()
    
    def dfs_bitmask(path, visited_mask, depth):
        nonlocal best_path, best_len
        if len(path) > best_len:
            best_len = len(path)
            best_path = list(path)
        
        if best_len == comp_size:
            return
        
        if time_up():
            return
        
        # Upper bound check
        if upper_bound(path, visited_mask) <= best_len:
            return
        
        cur = path[-1]
        nbrs = adj_bits[cur] & ~visited_mask
        if not nbrs:
            return
        
        # Get list of candidate neighbors, sorted by fewest unvisited neighbors
        candidates = []
        n = nbrs
        while n:
            bit = n & (-n)
            idx = bit.bit_length() - 1
            unvis_count = (adj_bits[idx] & ~visited_mask).bit_count()
            candidates.append((unvis_count, idx))
            n &= n - 1
        candidates.sort()
        
        for _, v in candidates:
            if time_up():
                return
            path.append(v)
            dfs_bitmask(path, visited_mask | (1 << v), depth + 1)
            path.pop()
            if best_len == comp_size:
                return
    
    # Try DFS from high-degree vertices
    deg_sorted = sorted(range(comp_size), key=lambda v: -len(local_adj_list[v]))
    for s in deg_sorted[:30]:
        if time_up():
            break
        dfs_bitmask([s], 1 << s, 0)
        if best_len == comp_size:
            break
    
    if best_len == comp_size:
        return [largest_comp[i] for i in best_path]
    
    # Phase 4: Local repair on best path
    if time.time() < deadline - 5:
        repaired = _local_repair(best_path, max_restarts=100)
        update_best(repaired)
    
    if best_len == comp_size:
        return [largest_comp[i] for i in best_path]
    
    # Phase 5: More randomized attempts with different strategies
    rng2 = random.Random(12345)
    for trial in range(300):
        if time.time() > deadline - 2:
            break
        start = rng2.randint(0, comp_size - 1)
        path = [start]
        seen = {start}
        # Random greedy
        while True:
            cur = path[-1]
            cands = [v for v in local_adj_list[cur] if v not in seen]
            if not cands:
                break
            v = rng2.choice(cands)
            path.append(v)
            seen.add(v)
        # Extend both ends randomly
        visited = set(path)
        changed = True
        while changed:
            changed = False
            cur = path[0]
            cands = [v for v in local_adj_list[cur] if v not in visited]
            if cands:
                v = rng2.choice(cands)
                path = [v] + path
                visited.add(v)
                changed = True
            cur = path[-1]
            cands = [v for v in local_adj_list[cur] if v not in visited]
            if cands:
                v = rng2.choice(cands)
                path = path + [v]
                visited.add(v)
                changed = True
        update_best(path)
    
    # Final: extend best path
    if best_path:
        best_path = _extend_both_ends(best_path)
        update_best(best_path)
    
    # Convert back to original vertex labels
    if not best_path:
        # Fallback: find any edge or single vertex
        for u, v in P['edges']:
            return [u, v]
        return [0]
    
    return [largest_comp[i] for i in best_path]

"""Longest simple path in pinned G(100, 1.8/100) graph via multi-strategy
search: greedy warm-starts, Warnsdorff DFS with backtracking, and
iterative local refinement (insertions, 2-opt swaps, end extensions)."""
import time
import random
import sys
from collections import deque

sys.setrecursionlimit(10000)

# Pinned graph
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)

DEG = [len(ADJ[i]) for i in range(N)]
NBR_SORTED = [sorted(ADJ[i], key=lambda x: (DEG[x], x)) for i in range(N)]

def find_components():
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
                for nb in ADJ[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            comps.append(comp)
    return comps

def warnsdorff_path(start, seen=None, deadline=None):
    """Greedy Warnsdorff: pick neighbor with fewest unvisited neighbors."""
    if seen is None:
        seen = set()
    seen.add(start)
    path = [start]
    while True:
        if deadline is not None and time.time() > deadline:
            break
        cur = path[-1]
        cands = [v for v in NBR_SORTED[cur] if v not in seen]
        if not cands:
            break
        best = min(cands, key=lambda v: (sum(1 for w in NBR_SORTED[v] if w not in seen), v))
        path.append(best)
        seen.add(best)
    return path

def greedy_path(start, mode='min', seen=None, deadline=None):
    if seen is None:
        seen = set()
    seen.add(start)
    path = [start]
    while True:
        if deadline is not None and time.time() > deadline:
            break
        cur = path[-1]
        cands = [v for v in NBR_SORTED[cur] if v not in seen]
        if not cands:
            break
        if mode == 'min':
            best = min(cands, key=lambda v: DEG[v])
        elif mode == 'max':
            best = max(cands, key=lambda v: DEG[v])
        else:
            best = random.choice(cands)
        path.append(best)
        seen.add(best)
    return path

def extend_both_ends(path, deadline=None):
    seen = set(path)
    changed = True
    while changed:
        if deadline is not None and time.time() > deadline:
            break
        changed = False
        if path:
            cur = path[0]
            cands = [v for v in NBR_SORTED[cur] if v not in seen]
            if cands:
                best = min(cands, key=lambda v: (sum(1 for w in NBR_SORTED[v] if w not in seen), v))
                path.insert(0, best)
                seen.add(best)
                changed = True
        if path:
            cur = path[-1]
            cands = [v for v in NBR_SORTED[cur] if v not in seen]
            if cands:
                best = min(cands, key=lambda v: (sum(1 for w in NBR_SORTED[v] if w not in seen), v))
                path.append(best)
                seen.add(best)
                changed = True
    return path

def try_insert(path, deadline=None):
    """Try to insert an unvisited vertex into the path at any valid position."""
    seen = set(path)
    unvisited = [v for v in range(N) if v not in seen]
    for v in unvisited:
        if deadline is not None and time.time() > deadline:
            break
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            if b in ADJ[v] and a in ADJ[v]:
                return path[:i+1] + [v] + path[i+1:]
    for v in unvisited:
        if path and path[-1] in ADJ[v]:
            return path + [v]
        if path and path[0] in ADJ[v]:
            return [v] + path
    return None

def try_2opt_swap(path, deadline=None):
    """Try to swap two internal vertices to enable extension."""
    seen = set(path)
    unvisited = [v for v in range(N) if v not in seen]
    # Try removing a vertex and reinserting it elsewhere
    for i in range(1, len(path) - 1):
        if deadline is not None and time.time() > deadline:
            break
        v = path[i]
        # Remove v
        trial = path[:i] + path[i+1:]
        # Check if path is still valid
        if len(trial) >= 2:
            valid = True
            for j in range(len(trial) - 1):
                if trial[j+1] not in ADJ[trial[j]]:
                    valid = False
                    break
            if not valid:
                continue
        # Try to insert v back at a different position
        for j in range(len(trial) - 1):
            a, b = trial[j], trial[j+1]
            if b in ADJ[v] and a in ADJ[v]:
                new_path = trial[:j+1] + [v] + trial[j+1:]
                return new_path
    return None

def refine(path, deadline=None):
    """Iterative local refinement: insertions, 2-opt swaps, end extensions."""
    best_path = list(path)
    improved = True
    max_iters = 500
    for _ in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        if not improved:
            break
        improved = False
        
        # Try insertions
        result = try_insert(best_path, deadline)
        if result and len(result) > len(best_path):
            best_path = result
            improved = True
        
        # Try 2-opt swaps
        result2 = try_2opt_swap(best_path, deadline)
        if result2 and len(result2) >= len(best_path):
            # Check if it enables further extension
            extended = extend_both_ends(result2, deadline)
            if len(extended) > len(best_path):
                best_path = extended
                improved = True
        
        # Try end extensions
        extended = extend_both_ends(best_path, deadline)
        if len(extended) > len(best_path):
            best_path = extended
            improved = True
    
    return best_path

def deep_dfs(start, best_len, deadline, max_nodes=200000):
    """
    Depth-first search with strong pruning and Warnsdorff ordering.
    Returns the best path found.
    """
    best_path = [start]
    best_len_local = 1
    visited_arr = [False] * N
    visited_arr[start] = True
    path = [start]
    node_count = [0]
    
    def dfs(cur):
        nonlocal best_path, best_len_local, node_count
        
        if time.time() > deadline:
            return
        node_count[0] += 1
        if node_count[0] > max_nodes:
            return
        
        if len(path) > best_len_local:
            best_len_local = len(path)
            best_path = list(path)
        
        # Pruning
        remaining = N - len(path)
        if len(path) + remaining <= best_len_local:
            return
        
        cands = [v for v in NBR_SORTED[cur] if not visited_arr[v]]
        if not cands:
            return
        
        cands.sort(key=lambda v: (sum(1 for w in NBR_SORTED[v] if not visited_arr[w]), v))
        
        for v in cands:
            if time.time() > deadline:
                return
            if node_count[0] > max_nodes:
                return
            
            path.append(v)
            visited_arr[v] = True
            dfs(v)
            path.pop()
            visited_arr[v] = False
            
            if time.time() > deadline or node_count[0] > max_nodes:
                return
    
    dfs(start)
    return best_path

def local_search_extend(path, deadline):
    """Local search: try to extend the path by finding vertices that can be inserted or appended."""
    best_path = list(path)
    seen = set(best_path)
    
    for _ in range(100):
        if time.time() > deadline:
            break
        
        extended = False
        # Try extending from tail
        cur = best_path[-1]
        cands = [v for v in NBR_SORTED[cur] if v not in seen]
        if cands:
            for v in cands:
                trial = best_path + [v]
                trial = extend_both_ends(trial, deadline)
                if len(trial) > len(best_path):
                    best_path = trial
                    seen = set(best_path)
                    extended = True
                    break
        
        # Try extending from head
        if not extended:
            cur = best_path[0]
            cands = [v for v in NBR_SORTED[cur] if v not in seen]
            if cands:
                for v in cands:
                    trial = [v] + best_path
                    trial = extend_both_ends(trial, deadline)
                    if len(trial) > len(best_path):
                        best_path = trial
                        seen = set(best_path)
                        extended = True
                        break
        
        # Try insertions
        if not extended:
            result = try_insert(best_path, deadline)
            if result and len(result) > len(best_path):
                best_path = result
                seen = set(best_path)
                extended = True
        
        if not extended:
            break
    
    return best_path

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # Safety margin within 90s limit
    
    # Find connected components
    comps = find_components()
    comps.sort(key=len, reverse=True)
    
    best_path = []
    best_len = 0
    
    # Phase 1: Multi-start greedy with various heuristics
    for comp in comps:
        if len(comp) <= best_len:
            break
        for start in comp:
            if time.time() > deadline - 15.0:
                break
            
            # Strategy 1: Warnsdorff
            p = warnsdorff_path(start, deadline=deadline - 10.0)
            p = extend_both_ends(p, deadline - 10.0)
            if len(p) > best_len:
                best_path = list(p)
                best_len = len(p)
            
            # Strategy 2: Min-degree greedy
            p2 = greedy_path(start, 'min', deadline=deadline - 10.0)
            p2 = extend_both_ends(p2, deadline - 10.0)
            if len(p2) > best_len:
                best_path = list(p2)
                best_len = len(p2)
            
            # Strategy 3: Max-degree greedy
            p3 = greedy_path(start, 'max', deadline=deadline - 10.0)
            p3 = extend_both_ends(p3, deadline - 10.0)
            if len(p3) > best_len:
                best_path = list(p3)
                best_len = len(p3)
    
    # Phase 2: Refine the best path found
    if best_path:
        refined = refine(best_path, deadline - 10.0)
        if len(refined) > best_len:
            best_path = refined
            best_len = len(refined)
    
    # Phase 3: Deep DFS from promising starts
    dfs_starts = set()
    if best_path:
        dfs_starts.add(best_path[0])
        dfs_starts.add(best_path[-1])
    if comps:
        for v in sorted(comps[0], key=lambda x: -DEG[x])[:15]:
            dfs_starts.add(v)
    
    for start in dfs_starts:
        if time.time() > deadline - 25.0:
            break
        
        dfs_deadline = min(deadline - 5.0, time.time() + 6.0)
        dfs_result = deep_dfs(start, best_len, dfs_deadline, max_nodes=150000)
        if len(dfs_result) > best_len:
            best_path = list(dfs_result)
            best_len = len(dfs_result)
        
        # Refine DFS result
        if len(dfs_result) > best_len - 5:
            refined = refine(dfs_result, deadline - 5.0)
            if len(refined) > best_len:
                best_path = refined
                best_len = len(refined)
    
    # Phase 4: Local search extension on the best path
    if best_path and time.time() < deadline - 10.0:
        ls_result = local_search_extend(best_path, deadline - 5.0)
        if len(ls_result) > best_len:
            best_path = ls_result
            best_len = len(ls_result)
    
    # Phase 5: Random restarts with refinement
    rng = random.Random(12345)
    for iteration in range(500):
        if time.time() > deadline - 10.0:
            break
        
        if comps:
            start = rng.choice(comps[0])
        else:
            start = rng.randint(0, N - 1)
        
        mode = rng.choice(['min', 'max', 'random'])
        p = greedy_path(start, mode, deadline=deadline - 5.0)
        p = extend_both_ends(p, deadline - 5.0)
        p = refine(p, deadline - 5.0)
        
        if len(p) > best_len:
            best_path = list(p)
            best_len = len(p)
        
        # Try local search on this path
        if len(p) >= best_len - 3 and time.time() < deadline - 5.0:
            ls_p = local_search_extend(p, deadline - 3.0)
            if len(ls_p) > best_len:
                best_path = ls_p
                best_len = len(ls_p)
    
    # Phase 6: Final refinement
    if best_path:
        final = refine(best_path, deadline - 2.0)
        if len(final) > best_len:
            best_path = final
            best_len = len(final)
    
    # Ensure nonempty
    if not best_path:
        best_path = [0]
    
    return best_path

"""Combined approach: structural component analysis + Warnsdorff greedy + bidirectional extension + time-limited backtracking."""
import time
import random
import sys

P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

N = P['n']
ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)

def find_components():
    visited = set()
    comps = []
    for s in range(N):
        if s not in visited:
            comp = set()
            stack = [s]
            visited.add(s)
            while stack:
                u = stack.pop()
                comp.add(u)
                for v in ADJ[u]:
                    if v not in visited:
                        visited.add(v)
                        stack.append(v)
            comps.append(comp)
    return comps

def warnsdorff_extend(path, seen):
    """Extend path from the end using Warnsdorff's rule."""
    while True:
        cand = ADJ[path[-1]] - seen
        if not cand:
            break
        v = min(cand, key=lambda w: (len(ADJ[w] - seen), w))
        path.append(v)
        seen.add(v)
    return path

def extend_both_ends(path):
    """Extend a path from both ends using Warnsdorff."""
    path = list(path)
    seen = set(path)
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

def try_insert_vertex(path, comp):
    """Try to insert a vertex not in path between consecutive pairs."""
    path_set = set(path)
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        candidates = [w for w in comp if w not in path_set and w in ADJ[u] and w in ADJ[v]]
        if candidates:
            # Pick the candidate that minimizes onward degree
            w = min(candidates, key=lambda x: len(ADJ[x] - path_set - {u, v}))
            new_path = path[:i+1] + [w] + path[i+1:]
            # Verify validity
            valid = True
            for j in range(len(new_path) - 1):
                if new_path[j+1] not in ADJ[new_path[j]]:
                    valid = False
                    break
            if valid and len(set(new_path)) == len(new_path):
                return new_path
    return None

def local_search_insert(path, comp, max_iter=30):
    """Repeatedly try to insert vertices into the path."""
    for _ in range(max_iter):
        new_path = try_insert_vertex(path, comp)
        if new_path is None:
            break
        path = new_path
    return path

def time_limited_dfs(start, comp, target_len, deadline, best_path):
    """Time-limited DFS with Warnsdorff ordering to find a longer path."""
    comp_set = set(comp)
    best = best_path[:]
    
    def dfs(path, seen):
        nonlocal best
        if time.time() > deadline:
            return
        if len(path) > len(best):
            best = path[:]
            if len(best) >= target_len:
                return
        cand = [v for v in ADJ[path[-1]] if v in comp_set and v not in seen]
        if not cand:
            return
        # Warnsdorff ordering
        cand.sort(key=lambda w: (len([x for x in ADJ[w] if x in comp_set and x not in seen and x != path[-1]]), w))
        for v in cand:
            if time.time() > deadline:
                return
            path.append(v)
            seen.add(v)
            dfs(path, seen)
            path.pop()
            seen.remove(v)
            if time.time() > deadline:
                return
    
    dfs([start], {start})
    return best

def construct():
    t0 = time.time()
    deadline = t0 + 82  # Safety margin before 90s kill
    
    # Build components
    comps = find_components()
    comps.sort(key=len, reverse=True)
    
    best_path = []
    
    for comp in comps:
        if time.time() > deadline:
            break
        comp_size = len(comp)
        comp_list = sorted(comp)
        
        # Phase 1: Greedy from every vertex with bidirectional extension
        for start in comp_list:
            if time.time() > deadline:
                break
            path = [start]
            seen = {start}
            path = warnsdorff_extend(path, seen)
            path = extend_both_ends(path)
            if len(path) > len(best_path):
                best_path = path[:]
        
        # Phase 2: Local search insertion on best path found so far
        if best_path:
            best_path = local_search_insert(best_path, comp)
        
        # Phase 3: If greedy didn't find full component, try time-limited DFS
        if len(best_path) < comp_size:
            # Try DFS from several promising starts
            # Sort by degree descending for promising starts
            starts = sorted(comp_list, key=lambda v: -len(ADJ[v]))
            # Also include some random starts
            random_starts = random.sample(comp_list, min(10, comp_size))
            all_starts = list(set(starts[:10] + random_starts))
            
            for start in all_starts:
                if time.time() > deadline:
                    break
                remaining = deadline - time.time()
                if remaining < 0.1:
                    break
                # Allocate time proportional to remaining
                budget = min(remaining * 0.3, 2.0)
                dl = time.time() + budget
                
                local_best = time_limited_dfs(start, comp, comp_size, dl, best_path)
                if len(local_best) > len(best_path):
                    best_path = local_best
                    # Try local search on improved path
                    best_path = local_search_insert(best_path, comp)
                    if len(best_path) >= comp_size:
                        break
    
    # Final verification
    if not best_path:
        return [0]
    
    for i in range(len(best_path) - 1):
        if best_path[i+1] not in ADJ[best_path[i]]:
            best_path = best_path[:i+1]
            break
    
    if len(set(best_path)) != len(best_path):
        seen = set()
        fixed = []
        for v in best_path:
            if v not in seen:
                fixed.append(v)
                seen.add(v)
        best_path = fixed
    
    return best_path if best_path else [0]

"""Find longest simple path in pinned graph using multi-strategy greedy + local search + DFS."""
import time
from collections import deque

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

def greedy_warnsdorff(start, comp_set, seen=None):
    """Greedy Warnsdorff: always pick neighbor with fewest unseen neighbors."""
    if seen is None:
        seen = {start}
    path = [start]
    while True:
        cand = ADJ[path[-1]] & comp_set - seen
        if not cand:
            break
        v = min(cand, key=lambda w: (len(ADJ[w] & comp_set - seen), w))
        path.append(v)
        seen.add(v)
    return path

def greedy_forced(start, comp_set, seen=None):
    """Greedy: prefer forced moves (unique unvisited neighbor), else max degree."""
    if seen is None:
        seen = {start}
    path = [start]
    while True:
        cand = ADJ[path[-1]] & comp_set - seen
        if not cand:
            break
        # Check for forced move
        forced = [w for w in cand if len(ADJ[w] & comp_set - seen) == 1]
        if forced:
            v = forced[0]
        else:
            v = max(cand, key=lambda w: (len(ADJ[w] & comp_set - seen), -w))
        path.append(v)
        seen.add(v)
    return path

def greedy_degree(start, comp_set, seen=None):
    """Greedy: pick neighbor with most unseen neighbors (opposite of Warnsdorff)."""
    if seen is None:
        seen = {start}
    path = [start]
    while True:
        cand = ADJ[path[-1]] & comp_set - seen
        if not cand:
            break
        v = max(cand, key=lambda w: (len(ADJ[w] & comp_set - seen), -w))
        path.append(v)
        seen.add(v)
    return path

def extend_both_ends(path, seen, comp_set):
    """Extend path greedily from both ends using Warnsdorff."""
    path = list(path)
    seen = set(seen)
    changed = True
    while changed:
        changed = False
        # Extend front
        while True:
            cand = ADJ[path[0]] & comp_set - seen
            if not cand:
                break
            v = min(cand, key=lambda w: (len(ADJ[w] & comp_set - seen), w))
            path.insert(0, v)
            seen.add(v)
            changed = True
        # Extend back
        while True:
            cand = ADJ[path[-1]] & comp_set - seen
            if not cand:
                break
            v = min(cand, key=lambda w: (len(ADJ[w] & comp_set - seen), w))
            path.append(v)
            seen.add(v)
            changed = True
    return path

def local_search_extend(path, comp_set, deadline):
    """
    Try to improve path by:
    1. Extending from both ends
    2. Inserting vertices between consecutive path vertices
    3. Trying different insertion points
    """
    path = list(path)
    seen = set(path)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to extend from both ends
        new_path = extend_both_ends(path, seen, comp_set)
        if len(new_path) > len(path):
            path = new_path
            seen = set(path)
            improved = True
        
        # Try to find a vertex that can be inserted into the path
        for i in range(len(path) - 1):
            if time.time() > deadline:
                return path
            u, v = path[i], path[i+1]
            # Find common neighbors of u and v not in path
            common = (ADJ[u] & ADJ[v]) & comp_set - seen
            if common:
                # Insert the one with best Warnsdorff score
                w = min(common, key=lambda x: (len(ADJ[x] & comp_set - seen), x))
                path.insert(i+1, w)
                seen.add(w)
                improved = True
                break  # Restart scan after insertion
    
    return path

def time_bounded_dfs(start, comp_set, deadline, best_path, best_len):
    """Time-bounded DFS to find longest path starting from start within comp."""
    best = [best_path[:], best_len]
    
    def dfs(path, seen):
        if time.time() > deadline:
            return
        if len(path) > best[1]:
            best[0] = path[:]
            best[1] = len(path)
        if len(path) == len(comp_set):
            return
        cand = ADJ[path[-1]] & comp_set - seen
        if not cand:
            return
        # Sort by Warnsdorff heuristic
        cand_sorted = sorted(cand, key=lambda w: (len(ADJ[w] & comp_set - seen), w))
        for v in cand_sorted:
            path.append(v)
            seen.add(v)
            dfs(path, seen)
            path.pop()
            seen.remove(v)
            if time.time() > deadline:
                return
    
    dfs([start], {start})
    return best[0]

def construct():
    t0 = time.time()
    deadline = t0 + 85  # Safety margin before 90s kill
    
    # Find components
    comps = find_components()
    comps.sort(key=len, reverse=True)
    
    if not comps:
        return [0]
    
    largest_comp = comps[0]
    comp_size = len(largest_comp)
    comp_list = sorted(largest_comp)
    
    best_path = []
    
    # Phase 1: Multiple greedy strategies from every vertex in largest component
    for start in comp_list:
        if time.time() > deadline:
            break
        # Warnsdorff
        p1 = greedy_warnsdorff(start, largest_comp)
        p1 = extend_both_ends(p1, set(p1), largest_comp)
        if len(p1) > len(best_path):
            best_path = list(p1)
        
        # Forced move
        p2 = greedy_forced(start, largest_comp)
        p2 = extend_both_ends(p2, set(p2), largest_comp)
        if len(p2) > len(best_path):
            best_path = list(p2)
        
        # Max degree
        p3 = greedy_degree(start, largest_comp)
        p3 = extend_both_ends(p3, set(p3), largest_comp)
        if len(p3) > len(best_path):
            best_path = list(p3)
        
        # If we found Hamiltonian path, done
        if len(best_path) >= comp_size:
            for i in range(len(best_path) - 1):
                if best_path[i+1] not in ADJ[best_path[i]]:
                    break
            else:
                return best_path
    
    # Phase 2: Local search to extend best path
    if best_path:
        local_dl = min(deadline, time.time() + 10)
        best_path = local_search_extend(best_path, largest_comp, local_dl)
        
        # If we found Hamiltonian path, done
        if len(best_path) >= comp_size:
            for i in range(len(best_path) - 1):
                if best_path[i+1] not in ADJ[best_path[i]]:
                    break
            else:
                return best_path
    
    # Phase 3: Time-bounded DFS from promising starts
    promising_starts = sorted(comp_list, key=lambda x: (-len(ADJ[x] & largest_comp), x))
    
    if best_path:
        for v in [best_path[0], best_path[-1], best_path[len(best_path)//2]]:
            if v not in promising_starts:
                promising_starts.insert(0, v)
    
    for start in promising_starts:
        if time.time() > deadline:
            break
        if len(best_path) >= comp_size:
            break
        
        remaining = deadline - time.time()
        if remaining < 0.5:
            break
        budget = min(5.0, remaining * 0.3)
        dl = time.time() + budget
        
        local_best = time_bounded_dfs(start, largest_comp, dl, best_path, len(best_path))
        if len(local_best) > len(best_path):
            best_path = local_best
    
    # Phase 4: Final local search
    if best_path:
        final_dl = min(deadline, time.time() + 15)
        best_path = local_search_extend(best_path, largest_comp, final_dl)
    
    # Phase 5: Try to extend the best path found
    if best_path:
        best_path = extend_both_ends(best_path, set(best_path), largest_comp)
        best_path = [v for v in best_path if v in largest_comp]
    
    # Final validation
    if not best_path:
        best_path = [comp_list[0]]
    
    for i in range(len(best_path) - 1):
        if best_path[i+1] not in ADJ[best_path[i]]:
            return [comp_list[0]]
    if len(set(best_path)) != len(best_path):
        seen = set()
        fixed = []
        for v in best_path:
            if v not in seen:
                fixed.append(v)
                seen.add(v)
        return fixed
    
    return best_path

"""Improved: systematic search for longest simple path using Warnsdorff + two-end extension + limited backtracking."""
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

def warnsdorff_score(v, seen):
    """Score for Warnsdorff: prefer vertices with fewer unseen neighbors."""
    return (len(ADJ[v] - seen), v)

def greedy_path(start):
    """Greedy Warnsdorff path from start."""
    seen = {start}
    path = [start]
    while True:
        cand = ADJ[path[-1]] - seen
        if not cand:
            break
        v = min(cand, key=lambda w: warnsdorff_score(w, seen))
        path.append(v)
        seen.add(v)
    return path

def two_ends_extend(path):
    """Extend path from both ends using Warnsdorff."""
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
            v = min(cand, key=lambda w: warnsdorff_score(w, seen))
            path.insert(0, v)
            seen.add(v)
            improved = True
        # Extend back
        while True:
            cand = ADJ[path[-1]] - seen
            if not cand:
                break
            v = min(cand, key=lambda w: warnsdorff_score(w, seen))
            path.append(v)
            seen.add(v)
            improved = True
    return path

def limited_backtracking(start, target_len, deadline, best_path, best_seen):
    """Time-limited DFS trying to find a path of target_len or longer."""
    found_hamiltonian = False
    
    def dfs(path, seen):
        nonlocal found_hamiltonian
        if time.time() > deadline:
            return
        if len(path) > len(best_path):
            best_path.clear()
            best_path.extend(path)
            best_seen.clear()
            best_seen.update(seen)
        if len(path) >= target_len:
            found_hamiltonian = True
            return
        cand = ADJ[path[-1]] - seen
        # Sort by Warnsdorff heuristic
        cand_sorted = sorted(cand, key=lambda w: warnsdorff_score(w, seen))
        for v in cand_sorted:
            path.append(v)
            seen.add(v)
            dfs(path, seen)
            path.pop()
            seen.remove(v)
            if time.time() > deadline:
                return
    
    dfs([start], {start})
    return found_hamiltonian

def construct():
    t0 = time.time()
    deadline = t0 + 85  # Safety margin before 90s limit
    
    # Find components
    comps = find_components()
    comps.sort(key=len, reverse=True)
    
    best_path = []
    
    for comp in comps:
        if time.time() > deadline:
            break
        comp_size = len(comp)
        comp_list = sorted(comp)
        
        # Phase 1: Try greedy + two-end extension from every vertex
        for start in comp_list:
            if time.time() > deadline:
                break
            p = greedy_path(start)
            p = two_ends_extend(p)
            if len(p) > len(best_path):
                best_path = list(p)
        
        # If we found a Hamiltonian path, we're done
        if len(best_path) >= comp_size:
            break
        
        # Phase 2: Limited backtracking from promising starts
        # Focus on vertices with high degree or in the current best path
        promising_starts = set(comp_list)
        # Add endpoints and middle of current best path
        if best_path:
            promising_starts.add(best_path[0])
            promising_starts.add(best_path[-1])
            promising_starts.add(best_path[len(best_path)//2])
        
        # Sort promising starts by degree (higher first)
        promising_starts = sorted(promising_starts, key=lambda x: (-len(ADJ[x]), x))
        
        for start in promising_starts:
            if time.time() > deadline:
                break
            if len(best_path) >= comp_size:
                break
            
            # Give each start a time budget
            remaining = deadline - time.time()
            if remaining < 0.1:
                break
            budget = min(2.0, remaining * 0.3)
            dl = time.time() + budget
            
            local_best = best_path[:]
            local_seen = set(best_path)
            
            found = limited_backtracking(start, comp_size, dl, local_best, local_seen)
            
            if len(local_best) > len(best_path):
                best_path = local_best
                if len(best_path) >= comp_size:
                    break
    
    if not best_path:
        best_path = [0]
    
    # Final validation
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

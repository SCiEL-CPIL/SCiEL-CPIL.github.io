import time
import sys

N = 9
SIZE = 1 << N

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s

    best_path = None
    best_len = 0

    # We'll do a backtracking search.
    # Fix v0 = 0, v1 = 1 (bit 0 flipped) by symmetry.
    
    path = [0, 1]
    used = [False] * SIZE
    used[0] = True
    used[1] = True
    
    # forbidden_count[x] = number of path vertices (excluding last) adjacent to x
    # Initially, last is index 1 (vertex 1). Non-consecutive for future: v0 = 0.
    # So neighbors of 0 are forbidden.
    forbidden_count = [0] * SIZE
    for b in range(N):
        nb = 0 ^ (1 << b)
        forbidden_count[nb] += 1
    
    # path_vertices list for backtracking: we need to know, when we backtrack from position k,
    # which vertex becomes "last again" and thus we decrement its neighbors.
    
    def time_ok():
        return time.time() < deadline
    
    # Recursive backtracking
    # path is the current list of vertices
    # We extend from path[-1]
    
    sys.setrecursionlimit(10000)
    
    def backtrack():
        nonlocal best_path, best_len
        if not time_ok():
            return
        cur_len = len(path)
        if cur_len > best_len:
            best_len = cur_len
            best_path = path.copy()
        
        cur = path[-1]
        # Find candidates: neighbors of cur that are not used and forbidden_count == 0
        cands = []
        for b in range(N):
            w = cur ^ (1 << b)
            if not used[w] and forbidden_count[w] == 0:
                cands.append(w)
        
        if not cands:
            return
        
        # Heuristic: sort candidates by number of available neighbors (more first)
        # For each candidate w, count how many of w's neighbors are not used and forbidden_count==0
        # But that's expensive; let's use a cheaper heuristic: prefer w with higher "openness"
        # Actually, let's just try in a fixed order but maybe randomize or use degree.
        # For now, sort by the number of free neighbors.
        
        def openness(w):
            cnt = 0
            for b in range(N):
                nb = w ^ (1 << b)
                if not used[nb] and forbidden_count[nb] == 0:
                    cnt += 1
            return cnt
        
        cands.sort(key=openness, reverse=True)
        
        for w in cands:
            if not time_ok():
                return
            # Add w
            path.append(w)
            used[w] = True
            # Now the previous last (cur) becomes non-consecutive; increment forbidden_count for its neighbors
            for b in range(N):
                nb = cur ^ (1 << b)
                forbidden_count[nb] += 1
            
            backtrack()
            
            # Backtrack
            for b in range(N):
                nb = cur ^ (1 << b)
                forbidden_count[nb] -= 1
            used[w] = False
            path.pop()
    
    backtrack()
    
    # If best_path is None, return the seed-like greedy
    if best_path is None:
        # fallback
        return [0, 1]
    
    return best_path

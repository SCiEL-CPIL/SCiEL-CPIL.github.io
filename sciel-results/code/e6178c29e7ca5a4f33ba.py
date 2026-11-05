import time
import random

N = 40
TARGET_DIFFS = set(range(1, 39)) # 1 to 38

def _cover_count(S):
    S = sorted(S)
    covered = set()
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            covered.add(S[j] - S[i])
    return len(covered & TARGET_DIFFS)

def _is_valid_basis(S):
    return _cover_count(S) == len(TARGET_DIFFS)

def _find_basis(target_size, time_limit=5.0):
    start = time.time()
    best_S = None
    best_score = 0
    
    # Multiple restarts
    while time.time() - start < time_limit:
        # Random init
        S = set(random.sample(range(1, 40), target_size))
        score = _cover_count(S)
        
        # Local Search
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            S_list = sorted(S)
            for i in range(len(S_list)):
                old = S_list[i]
                # Try all possible new values
                for nv in range(1, 40):
                    if nv in S and nv != old:
                        continue
                    S.remove(old)
                    S.add(nv)
                    new_score = _cover_count(S)
                    if new_score > score:
                        score = new_score
                        improved = True
                        # Don't break immediately, might find better, but for speed:
                        break
                    else:
                        S.remove(nv)
                        S.add(old)
                if score == len(TARGET_DIFFS):
                    return sorted(S)
        
        if score > best_score:
            best_score = score
            best_S = sorted(S)
            
    if best_S and _is_valid_basis(best_S):
        return best_S
    return None

def construct():
    deadline = time.time() + 80.0
    
    # Try to find smallest basis
    for size in [9, 10, 11]:
        # Allocate time based on difficulty
        t_limit = 10.0 if size == 9 else 5.0
        S = _find_basis(size, t_limit)
        if S:
            break
        if time.time() > deadline - 5.0:
            # Fallback to guaranteed valid
            S = list(range(1, 12)) # Size 11, might not be valid, need to check
            # Actually, let's just use a known valid size 11 or generate one
            # The current program's fallback is star+path which is huge.
            # We need a valid basis.
            # A simple valid basis of size 11:
            # We can just pick 1, 2, ..., 11? No, diffs only up to 10.
            # We need to cover up to 38.
            # Greedy construction for size 11:
            S = [1]
            for _ in range(10):
                # Pick next element to maximize coverage
                best_next = None
                best_cov = 0
                for x in range(1, 40):
                    if x in S: continue
                    temp_S = S + [x]
                    cov = _cover_count(temp_S)
                    if cov > best_cov:
                        best_cov = cov
                        best_next = x
                if best_next:
                    S.append(best_next)
                else:
                    break
            S = sorted(set(S))
            if not _is_valid_basis(S):
                 # Last resort: brute force small size 12?
                 # Or just return the huge star+path to ensure validity, though score will be bad.
                 # But we want to avoid that.
                 pass
            
            if _is_valid_basis(S):
                break
            else:
                # If greedy failed, try random search for size 11
                S = _find_basis(11, 10.0)
                if not S:
                    # Absolute fallback: Star + Path
                    edges = [(0, i) for i in range(1, N)] + [(i, i+1) for i in range(1, N-1)]
                    cycles = [list(range(k)) for k in range(3, N+1)]
                    return {'edges': edges, 'cycles': cycles}
                break

    # Construct graph
    S_set = set(S)
    edges = []
    edges_set = set()
    
    def add_edge(a, b):
        if a == b: return
        if a > b: a, b = b, a
        if (a, b) not in edges_set:
            edges_set.add((a, b))
            edges.append((a, b))
            
    # Path 1-39
    for i in range(1, N-1):
        add_edge(i, i+1)
    # Spokes 0-s
    for s in S:
        add_edge(0, s)
        
    cycles = []
    for L in range(3, N+1):
        diff = L - 2
        found = False
        for u in S:
            v = u + diff
            if v in S_set:
                cycle = [0] + list(range(u, v+1))
                # Verify
                ok = True
                for i in range(len(cycle)-1):
                    a, b = cycle[i], cycle[i+1]
                    if a > b: a, b = b, a
                    if (a, b) not in edges_set:
                        ok = False
                        break
                if ok:
                    a, b = cycle[-1], cycle[0]
                    if a > b: a, b = b, a
                    if (a, b) not in edges_set:
                        ok = False
                if ok:
                    cycles.append(cycle)
                    found = True
                    break
        if not found:
            # Should not happen if S is valid basis
            # Fallback
            edges = [(0, i) for i in range(1, N)] + [(i, i+1) for i in range(1, N-1)]
            cycles = [list(range(k)) for k in range(3, N+1)]
            return {'edges': edges, 'cycles': cycles}
            
    return {'edges': edges, 'cycles': cycles}

import time
import random

_N = 256
_DEADLINE = 50.0

def _is_good_pair(a, b):
    return (a * b) % (a + b) != 0

def _build_conflict_graph():
    conflict = [set() for _ in range(_N + 1)]
    for a in range(1, _N + 1):
        for b in range(a + 1, _N + 1):
            if (a * b) % (a + b) == 0:
                conflict[a].add(b)
                conflict[b].add(a)
    return conflict

def _validate(S):
    s = set(S)
    if len(s) != len(S):
        return False
    if any(x < 1 or x > _N for x in s):
        return False
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            a, b = S[i], S[j]
            if (a * b) % (a + b) == 0:
                return False
    return True

def _greedy_and_local_search(conflict, start_time, deadline):
    best_set = set()
    nodes = list(range(1, _N + 1))
    
    while time.time() < deadline:
        # Randomized greedy initialization
        random.shuffle(nodes)
        S = set()
        for x in nodes:
            if not (S & conflict[x]):
                S.add(x)
        
        # Local improvement
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Add any node that has no conflicts with S
            for v in nodes:
                if v not in S:
                    if not (S & conflict[v]):
                        S.add(v)
                        improved = True
            
            # 2. Try 1-for-2 swaps
            # For each u in S, find candidates v in V\S that only conflict with u in S
            # If we can find two such candidates that are compatible with each other, 
            # we can remove u and add them.
            for u in list(S):
                candidates = []
                for v in nodes:
                    if v == u or v in S:
                        continue
                    # Check if v's neighbors in S are exactly {u}
                    # i.e., (conflict[v] & S) == {u}
                    if (conflict[v] & S) == {u}:
                        candidates.append(v)
                
                # Try to find two compatible candidates
                if len(candidates) >= 2:
                    found = False
                    for i in range(len(candidates)):
                        if found: break
                        for j in range(i + 1, len(candidates)):
                            v1 = candidates[i]
                            v2 = candidates[j]
                            if _is_good_pair(v1, v2):
                                # Perform swap
                                S.remove(u)
                                S.add(v1)
                                S.add(v2)
                                improved = True
                                found = True
                                break
                    if found:
                        continue # Restart the while loop as S changed significantly
        
        if len(S) > len(best_set):
            best_set = set(S)
            
    return list(best_set)

def construct():
    start_time = time.time()
    conflict = _build_conflict_graph()
    
    # Initial heuristic: Odd numbers
    best_set = set(range(1, 257, 2))
    
    # Run local search from multiple random starts
    # We can also try to improve the odd set directly
    S = set(best_set)
    # Quick local improvement on the odd set
    improved = True
    while improved and time.time() < _DEADLINE:
        improved = False
        for v in range(1, _N + 1):
            if v not in S:
                if not (S & conflict[v]):
                    S.add(v)
                    improved = True
        for u in list(S):
            candidates = []
            for v in range(1, _N + 1):
                if v == u or v in S:
                    continue
                if (conflict[v] & S) == {u}:
                    candidates.append(v)
            if len(candidates) >= 2:
                for i in range(len(candidates)):
                    for j in range(i + 1, len(candidates)):
                        v1, v2 = candidates[i], candidates[j]
                        if _is_good_pair(v1, v2):
                            S.remove(u)
                            S.add(v1)
                            S.add(v2)
                            improved = True
                            break
                    if improved:
                        break
            if improved:
                break
    if len(S) > len(best_set):
        best_set = set(S)

    # Run randomized restarts
    result = _greedy_and_local_search(conflict, start_time, _DEADLINE)
    
    if len(result) > len(best_set):
        best_set = set(result)
        
    # Final validation and return
    if not _validate(list(best_set)):
        return list(range(1, 257, 2))
        
    return list(best_set)

import time
import random
import itertools

N = 16
_ALL_EDGES = [(u, v) for u in range(N) for v in range(u+1, N)]
_EIGHT_SUBS = list(itertools.combinations(range(N), 8))

def _build_adj(edge_set):
    adj = [0] * N
    for (u, v) in edge_set:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _has_q3(adj):
    for sub in _EIGHT_SUBS:
        # Quick edge count
        ec = 0
        for i in range(8):
            for j in range(i+1, 8):
                if adj[sub[i]] & (1 << sub[j]):
                    ec += 1
        if ec < 12:
            continue
        for even_choice in itertools.combinations(range(8), 4):
            odd_set = set(range(8)) - set(even_choice)
            odd_choice = tuple(odd_set)
            zeros = []
            for ri in range(4):
                for cj in range(4):
                    if not (adj[sub[even_choice[ri]]] & (1 << sub[odd_choice[cj]])):
                        zeros.append((ri, cj))
            if len(zeros) != 4:
                continue
            row_z = [False]*4
            col_z = [False]*4
            ok = True
            for (ri, cj) in zeros:
                if row_z[ri] or col_z[cj]:
                    ok = False
                    break
                row_z[ri] = True
                col_z[cj] = True
            if ok:
                return True
    return False

def construct():
    deadline = time.time() + 110
    random.seed(42)
    
    edge_set = set((a, b) for a in range(3) for b in range(3, 16))
    adj = _build_adj(edge_set)
    best_edges = len(edge_set)
    best_adj = adj
    
    # Greedy add
    for e in _ALL_EDGES:
        if time.time() > deadline - 10:
            break
        if e in edge_set:
            continue
        edge_set.add(e)
        new_adj = _build_adj(edge_set)
        if not _has_q3(new_adj):
            adj = new_adj
            if len(edge_set) > best_edges:
                best_edges = len(edge_set)
                best_adj = adj
        else:
            edge_set.remove(e)
    
    # Limited local search
    steps = 0
    while time.time() < deadline - 3 and steps < 20000:
        steps += 1
        missing = [e for e in _ALL_EDGES if e not in edge_set]
        if not missing:
            break
        e = random.choice(missing)
        edge_set.add(e)
        new_adj = _build_adj(edge_set)
        if not _has_q3(new_adj):
            adj = new_adj
            if len(edge_set) > best_edges:
                best_edges = len(edge_set)
                best_adj = adj
        else:
            edge_set.remove(e)
    
    result = []
    for u in range(N):
        for v in range(u+1, N):
            if best_adj[u] & (1 << v):
                result.append([u, v])
    return result

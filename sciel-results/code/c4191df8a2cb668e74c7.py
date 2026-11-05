import time
import random
import math

_START = time.time()
_DEADLINE = 140

def _elapsed():
    return time.time() - _START

def _has_k4(adj, n):
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                continue
            common = adj[i] & adj[j] & ~((1 << (j + 1)) - 1)
            bits = []
            c = common
            while c:
                lsb = c & -c
                idx = lsb.bit_length() - 1
                bits.append(idx)
                c ^= lsb
            for k_idx in range(len(bits)):
                k = bits[k_idx]
                for l_idx in range(k_idx + 1, len(bits)):
                    l = bits[l_idx]
                    if adj[k] & (1 << l):
                        return [i, j, k, l]
    return None

def _has_ind_6(adj, n):
    def backtrack(candidates, current_set):
        if len(current_set) == 6:
            return current_set[:]
        if candidates.bit_count() < 6 - len(current_set):
            return None
        lsb = candidates & -candidates
        v = lsb.bit_length() - 1
        candidates ^= lsb
        new_candidates = candidates & ~adj[v]
        res = backtrack(new_candidates, current_set + [v])
        if res:
            return res
        res = backtrack(candidates, current_set)
        if res:
            return res
        return None
    return backtrack((1 << n) - 1, [])

def _try_local_search(n, time_limit):
    start = time.time()
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.4:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    while time.time() - start < time_limit:
        k4 = _has_k4(adj, n)
        if k4:
            pairs = [(k4[0], k4[1]), (k4[0], k4[2]), (k4[0], k4[3]),
                     (k4[1], k4[2]), (k4[1], k4[3]), (k4[2], k4[3])]
            u, v = random.choice(pairs)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            continue
        
        ind6 = _has_ind_6(adj, n)
        if ind6:
            pairs = [(ind6[i], ind6[j]) for i in range(6) for j in range(i + 1, 6)]
            u, v = random.choice(pairs)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            continue
        
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges
    
    return None

def construct():
    # Try n=35
    if _elapsed() < _DEADLINE - 10:
        edges = _try_local_search(35, _DEADLINE - _elapsed() - 5)
        if edges is not None:
            return {'n': 35, 'edges': edges}
            
    # Try n=34
    if _elapsed() < _DEADLINE - 5:
        edges = _try_local_search(34, _DEADLINE - _elapsed() - 2)
        if edges is not None:
            return {'n': 34, 'edges': edges}
            
    # Fallback
    p = 17
    squares = {x * x % p for x in range(1, p)}
    best_edges = [[i, j] for i in range(p) for j in range(i + 1, p) if (j - i) % p in squares]
    return {'n': p, 'edges': best_edges}

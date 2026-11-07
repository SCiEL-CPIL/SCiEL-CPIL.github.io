"""Optimize dense odd-order graph avoiding equal-degree distance-3 endpoints."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

_N = 15

def _adjacency(needles):
    adj = [0]*_N
    for u,v in needles:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj

def _degrees(adj):
    return [bin(adj[v]).count('1') for v in range(_N)]

def _is_valid(adj):
    deg = _degrees(adj)
    for u in range(_N):
        for v in range(_N):
            if u < v and (adj[u] >> v) & 1:
                # path s-u-v-t, s != t, s != v, t != u
                s_bits = adj[u] & ~(1 << v)
                t_bits = adj[v] & ~(1 << u)
                # s must also have deg[s] != deg[t] for all s in s_bits, t in t_bits
                while s_bits:
                    s = (s_bits & -s_bits).bit_length() - 1
                    ds = deg[s]
                    tt = t_bits
                    while tt:
                        t = (tt & -tt).bit_length() - 1
                        if ds == deg[t]:
                            return False
                        tt &= tt - 1
                    s_bits &= s_bits - 1
    return True

def _violation_count(adj):
    deg = _degrees(adj)
    cnt = 0
    for u in range(_N):
        for v in range(_N):
            if u < v and (adj[u] >> v) & 1:
                s_bits = adj[u] & ~(1 << v)
                t_bits = adj[v] & ~(1 << u)
                while s_bits:
                    s = (s_bits & -s_bits).bit_length() - 1
                    ds = deg[s]
                    tt = t_bits
                    while tt:
                        t = (tt & -tt).bit_length() - 1
                        if ds == deg[t]:
                            cnt += 1
                        tt &= tt - 1
                    s_bits &= s_bits - 1
    return cnt

def _edge_list(adj):
    out = []
    for u in range(_N):
        for v in range(u+1, _N):
            if (adj[u] >> v) & 1:
                out.append([u, v])
    return out

def _greedy_build(rng):
    # Start empty, add edges in random order, keep only if valid
    adj = [0]*_N
    edges = [(u, v) for u in range(_N) for v in range(u+1, _N)]
    rng.shuffle(edges)
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        if not _is_valid(adj):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
    return adj

def _local_search(adj, rng, deadline):
    cur = adj
    cur_deg = _degrees(cur)
    cur_v = _violation_count(cur)
    best = cur
    best_v = cur_v
    start = time.time()
    # Try adding edges that don't create violations, then swaps
    while time.time() < deadline:
        improved = False
        # 1) Try adding each missing edge
        for u in range(_N):
            for v in range(u+1, _N):
                if not (cur[u] >> v) & 1:
                    cur[u] |= 1 << v
                    cur[v] |= 1 << u
                    if _is_valid(cur):
                        improved = True
                        if _violation_count(cur) < best_v:
                            best = cur[:]
                            best_v = 0
                        break
                    else:
                        cur[u] &= ~(1 << v)
                        cur[v] &= ~(1 << u)
            if improved:
                break
        if improved:
            cur_deg = _degrees(cur)
            continue
        # 2) Try swaps: remove one edge, add another
        done = False
        for _ in range(200):
            if time.time() >= deadline:
                break
            e1 = None
            e2 = None
            # pick random existing edge to remove
            existing = [(u,v) for u in range(_N) for v in range(u+1,_N) if (cur[u]>>v)&1]
            if not existing:
                break
            e1 = rng.choice(existing)
            # pick random missing edge to add
            missing = [(u,v) for u in range(_N) for v in range(u+1,_N) if not (cur[u]>>v)&1]
            if not missing:
                break
            e2 = rng.choice(missing)
            u1,v1 = e1
            u2,v2 = e2
            cur[u1] &= ~(1<<v1); cur[v1] &= ~(1<<u1)
            cur[u2] |= (1<<v2); cur[v2] |= (1<<u2)
            if _is_valid(cur):
                if _violation_count(cur) < best_v:
                    best = cur[:]
                    best_v = 0
            else:
                cur[u1] |= (1<<v1); cur[v1] |= (1<<u1)
                cur[u2] &= ~(1<<v2); cur[v2] &= ~(1<<u2)
            done = True
        if not done:
            break
    return best

def construct():
    start = time.time()
    deadline = start + 80  # safety margin
    best_adj = None
    best_edges = 0

    rng = random.Random(42)
    # Try multiple greedy starts
    for seed in range(200):
        if time.time() >= deadline:
            break
        rng = random.Random(seed)
        adj = _greedy_build(rng)
        e = sum(bin(x).count('1') for x in adj)//2
        if e > best_edges:
            best_edges = e
            best_adj = adj
    # Local search from best
    if best_adj is not None:
        if time.time() < deadline:
            adj = _local_search(best_adj, random.Random(7), deadline)
            e = sum(bin(x).count('1') for x in adj)//2
            if e > best_edges:
                best_edges = e
                best_adj = adj
    # Return edges
    if best_adj is None:
        return []
    return _edge_list(best_adj)
# EVOLVE-BLOCK-END

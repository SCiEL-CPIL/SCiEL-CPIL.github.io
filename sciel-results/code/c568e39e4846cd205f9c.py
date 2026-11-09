import numpy as np
import time
import random

def construct():
    n = 20
    start = time.time()
    deadline = start + 135.0  # safety margin under 150s limit

    def verify(adj):
        adj = np.asarray(adj, dtype=np.int8)
        if adj.shape != (n, n):
            return None
        if np.any(np.diag(adj) != 0):
            return None
        if np.any(adj != (adj.T)):
            return None
        if np.any((adj != 0) & (adj != 1)):
            return None
        c = (adj @ adj)
        # c[u,v] = number of common neighbors of u and v
        if np.any(c > 2):
            return None
        return adj

    # Fallback: 5 disjoint K4s (10 edges)
    def fallback():
        adj = np.zeros((n, n), dtype=np.int8)
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    adj[base + a, base + b] = 1
                    adj[base + b, base + a] = 1
        return [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]

    def edges_from_adj(adj):
        return [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]

    def count_edges(adj):
        return int(adj.sum() // 2)

    def can_add_mask(adj, c, u, v):
        """Vectorized check: can we add edge (u,v)?"""
        if adj[u, v]:
            return False
        # For each w in N(u): c[w,v] + 1 <= 2  -> c[w,v] <= 1
        # For each w in N(v): c[u,w] + 1 <= 2  -> c[u,w] <= 1
        if np.any(c[adj[u].astype(bool), v] > 1):
            return False
        if np.any(c[u, adj[v].astype(bool)] > 1):
            return False
        return True

    def do_add(adj, c, u, v):
        adj[u, v] = 1
        adj[v, u] = 1
        nu = adj[u].astype(bool)
        nv = adj[v].astype(bool)
        # common[w][v] increments for w in N(u), w != v
        idx = np.where(nu)[0]
        idx = idx[idx != v]
        c[idx, v] += 1
        c[v, idx] += 1
        idx2 = np.where(nv)[0]
        idx2 = idx2[idx2 != u]
        c[u, idx2] += 1
        c[idx2, u] += 1

    def do_remove(adj, c, u, v):
        adj[u, v] = 0
        adj[v, u] = 0
        nu = adj[u].astype(bool)
        nv = adj[v].astype(bool)
        idx = np.where(nu)[0]
        idx = idx[idx != v]
        c[idx, v] -= 1
        c[v, idx] -= 1
        idx2 = np.where(nv)[0]
        idx2 = idx2[idx2 != u]
        c[u, idx2] -= 1
        c[idx2, u] -= 1

    def greedy_build(rng):
        adj = np.zeros((n, n), dtype=np.int8)
        c = np.zeros((n, n), dtype=np.int16)
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(pairs)
        for u, v in pairs:
            if can_add_mask(adj, c, u, v):
                do_add(adj, c, u, v)
        return adj

    def local_search(adj, rng, max_iters=3000):
        c = (adj @ adj).astype(np.int16)
        cur = count_edges(adj)
        best = cur
        best_adj = adj.copy()
        iters = 0
        while iters < max_iters and time.time() < deadline - 15.0:
            iters += 1
            # 1) Try to add a random missing edge
            missing = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u, v]]
            if missing:
                u, v = rng.choice(missing)
                if can_add_mask(adj, c, u, v):
                    do_add(adj, c, u, v)
                    if count_edges(adj) > best:
                        best = count_edges(adj)
                        best_adj = adj.copy()
                    continue
            # 2) Try a 1-for-1 edge swap: remove (a,b), add (u,v)
            present = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v]]
            if present and missing:
                a, b = rng.choice(present)
                u, v = rng.choice(missing)
                do_remove(adj, c, a, b)
                if can_add_mask(adj, c, u, v):
                    do_add(adj, c, u, v)
                    if count_edges(adj) > best:
                        best = count_edges(adj)
                        best_adj = adj.copy()
                else:
                    # revert
                    do_add(adj, c, a, b)
        return best_adj

    best = None
    best_e = 0

    # Multiple greedy builds with different seeds
    for trial in range(60):
        if time.time() > deadline - 20.0:
            break
        trial_rng = random.Random(trial * 9973 + 17)
        adj = greedy_build(trial_rng)
        v = verify(adj)
        if v is not None:
            e = count_edges(adj)
            if e > best_e:
                best_e = e
                best = adj
            # local search to improve
            ls = local_search(adj.copy(), random.Random(trial * 131 + 5), max_iters=2000)
            v2 = verify(ls)
            if v2 is not None:
                e2 = count_edges(ls)
                if e2 > best_e:
                    best_e = e2
                    best = ls

    # Final: run local search on the best found for a bit more
    if best is not None and time.time() < deadline - 15.0:
        ls = local_search(best.copy(), random.Random(12345), max_iters=3000)
        v = verify(ls)
        if v is not None and count_edges(ls) > best_e:
            best = ls
            best_e = count_edges(ls)

    if best is None:
        return fallback()
    return edges_from_adj(best)

if __name__ == "__main__":
    result = construct()
    print(f"Edges: {len(result)}")

import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 138.0
    n = 16
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    all_8subsets = [list(S) for S in combinations(range(n), 8)]

    def find_q3(edge_set, adj):
        for S in all_8subsets:
            ok = True
            for v in S:
                d = 0
                for u in S:
                    if u != v and u in adj[v]:
                        d += 1
                        if d > 3:
                            ok = False
                            break
                if not ok:
                    break
                if d != 3:
                    ok = False
                    break
            if not ok:
                continue
            color = {}
            is_bip = True
            for v in S:
                if v not in color:
                    color[v] = 0
                    stack = [v]
                    while stack and is_bip:
                        u = stack.pop()
                        for w in S:
                            if w in adj[u]:
                                if w not in color:
                                    color[w] = 1 - color[u]
                                    stack.append(w)
                                elif color[w] == color[u]:
                                    is_bip = False
                                    break
            if is_bip:
                return tuple(S)
        return None

    def build_adj(edge_set):
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    random.seed(42)
    best_edge_set = set()
    best_count = 0

    for trial in range(3):
        if time.time() - start_time > DEADLINE - 50:
            break
        p = 0.4
        edge_set = set()
        for e in all_edges:
            if random.random() < p:
                edge_set.add(e)
        adj = build_adj(edge_set)
        for _ in range(50):
            if time.time() - start_time > DEADLINE - 40:
                break
            q3 = find_q3(edge_set, adj)
            if q3 is None:
                break
            q3_edges = [(u, v) for u in q3 for v in q3 if u < v and (u, v) in edge_set]
            if q3_edges:
                e = random.choice(q3_edges)
                edge_set.discard(e)
                adj[e[0]].discard(e[1])
                adj[e[1]].discard(e[0])
        for e in all_edges:
            if time.time() - start_time > DEADLINE - 15:
                break
            if e in edge_set:
                continue
            edge_set.add(e)
            adj[e[0]].add(e[1])
            adj[e[1]].add(e[0])
            if find_q3(edge_set, adj) is not None:
                edge_set.discard(e)
                adj[e[0]].discard(e[1])
                adj[e[1]].discard(e[0])
        if len(edge_set) > best_count:
            best_count = len(edge_set)
            best_edge_set = set(edge_set)

    if best_edge_set:
        edge_set = set(best_edge_set)
        adj = build_adj(edge_set)
        for _ in range(500):
            if time.time() - start_time > DEADLINE - 5:
                break
            missing = [e for e in all_edges if e not in edge_set]
            if not missing:
                break
            e = random.choice(missing)
            edge_set.add(e)
            adj[e[0]].add(e[1])
            adj[e[1]].add(e[0])
            if find_q3(edge_set, adj) is not None:
                edge_set.discard(e)
                adj[e[0]].discard(e[1])
                adj[e[1]].discard(e[0])
        if len(edge_set) > best_count:
            best_count = len(edge_set)
            best_edge_set = set(edge_set)

    if not best_edge_set:
        # Fallback: K3,13
        best_edge_set = set((a, b) for a in range(3) for b in range(3, 16))

    return [[u, v] for (u, v) in best_edge_set]

"""Construct a K2,3-free graph on 20 vertices maximizing edges."""
import time
import random
from itertools import combinations

N = 20
DEADLINE = 115.0


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    all_edges = list(combinations(range(N), 2))
    m_total = len(all_edges)  # 190

    def popcount(x):
        return bin(x).count('1')

    def can_add(adj, u, v):
        """Check if edge (u,v) can be added."""
        mu = adj[u]
        mv = adj[v]
        # For each neighbor w of v (w != u): common(u,w) must be <= 1
        t = mv
        w = 0
        while t:
            if t & 1 and w != u:
                if popcount(mu & adj[w]) >= 2:
                    return False
            t >>= 1
            w += 1
        # For each neighbor w of u (w != v): common(v,w) must be <= 1
        t = mu
        w = 0
        while t:
            if t & 1 and w != v:
                if popcount(mv & adj[w]) >= 2:
                    return False
            t >>= 1
            w += 1
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def edge_count(adj):
        s = 0
        for i in range(N):
            s += popcount(adj[i])
        return s // 2

    def greedy_build(rng):
        adj = [0] * N
        order = list(range(m_total))
        rng.shuffle(order)
        for idx in order:
            u, v = all_edges[idx]
            if can_add(adj, u, v):
                add_edge(adj, u, v)
        return adj

    # Multiple greedy starts with different seeds
    best_adj = None
    best_cnt = 0
    rng = random.Random(42)
    for i in range(80):
        if time.time() > deadline - 40:
            break
        adj = greedy_build(rng)
        cnt = edge_count(adj)
        if cnt > best_cnt:
            best_cnt = cnt
            best_adj = adj[:]

    if best_adj is None:
        best_adj = greedy_build(rng)
        best_cnt = edge_count(best_adj)

    # Local search with 1-swap and 2-swap moves
    def local_search(adj, cnt, time_limit=5.0):
        start = time.time()
        while time.time() - start < time_limit and time.time() < deadline - 5:
            improved = False
            # Build list of present edges
            present = []
            for u in range(N):
                t = adj[u]
                w = u + 1
                while t:
                    if t & 1:
                        present.append((u, w))
                    t >>= 1
                    w += 1
            rng.shuffle(present)

            for (ru, rv) in present:
                if time.time() > deadline - 5:
                    break
                saved = adj[:]
                remove_edge(saved, ru, rv)
                new_cnt = cnt - 1

                # Try to add back up to 2 edges
                missing = []
                for u in range(N):
                    t = ~saved[u] & ((1 << N) - 1)
                    w = u + 1
                    while t:
                        if t & 1:
                            missing.append((u, w))
                        t >>= 1
                        w += 1
                rng.shuffle(missing)

                added = 0
                for (au, av) in missing:
                    if added >= 2:
                        break
                    if can_add(saved, au, av):
                        add_edge(saved, au, av)
                        new_cnt += 1
                        added += 1

                if new_cnt > cnt:
                    adj = saved
                    cnt = new_cnt
                    improved = True
                    break
            if not improved:
                # Try 2-edge removal + 3-edge addition
                if len(present) >= 2:
                    found = False
                    for i in range(min(len(present), 20)):
                        if time.time() > deadline - 5:
                            break
                        for j in range(i + 1, min(len(present), 20)):
                            if time.time() > deadline - 5:
                                break
                            (ru1, rv1) = present[i]
                            (ru2, rv2) = present[j]
                            saved = adj[:]
                            remove_edge(saved, ru1, rv1)
                            remove_edge(saved, ru2, rv2)
                            new_cnt = cnt - 2

                            missing = []
                            for u in range(N):
                                t = ~saved[u] & ((1 << N) - 1)
                                w = u + 1
                                while t:
                                    if t & 1:
                                        missing.append((u, w))
                                    t >>= 1
                                    w += 1
                            rng.shuffle(missing)

                            added = 0
                            for (au, av) in missing:
                                if added >= 3:
                                    break
                                if can_add(saved, au, av):
                                    add_edge(saved, au, av)
                                    new_cnt += 1
                                    added += 1

                            if new_cnt > cnt:
                                adj = saved
                                cnt = new_cnt
                                found = True
                                break
                        if found:
                            break
                    if found:
                        improved = True
            if not improved:
                break
        return adj, cnt

    # Run local search
    for i in range(15):
        if time.time() > deadline - 10:
            break
        best_adj, best_cnt = local_search(best_adj, best_cnt, time_limit=4.0)

    # Final greedy polish
    for _ in range(3):
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for idx in range(m_total):
                u, v = all_edges[idx]
                if not ((best_adj[u] >> v) & 1):
                    if can_add(best_adj, u, v):
                        add_edge(best_adj, u, v)
                        improved = True
                        best_cnt = edge_count(best_adj)

    # Build edge list
    edges = []
    for u in range(N):
        t = best_adj[u]
        w = u + 1
        while t:
            if t & 1:
                edges.append([u, w])
            t >>= 1
            w += 1

    return edges

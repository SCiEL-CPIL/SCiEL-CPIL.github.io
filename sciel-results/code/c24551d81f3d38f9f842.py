import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80

    n = 15
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def is_valid(adj):
        degs = [len(adj[i]) for i in range(n)]
        groups = {}
        for i in range(n):
            d = degs[i]
            if d not in groups:
                groups[d] = []
            groups[d].append(i)

        for d, group in groups.items():
            if len(group) < 2:
                continue
            for i in range(len(group)):
                s = group[i]
                for j in range(i + 1, len(group)):
                    t = group[j]
                    for u in adj[s]:
                        if u == t:
                            continue
                        for v in adj[t]:
                            if v == s or v == u:
                                continue
                            if u in adj[v]:
                                return False
        return True

    def edge_count(adj):
        return sum(len(adj[i]) for i in range(n)) // 2

    def make_bipartite_base():
        adj = [set() for _ in range(n)]
        for i in range(7):
            for j in range(7, 15):
                adj[i].add(j)
                adj[j].add(i)
        return adj

    # Start from K_{7,8} bipartite (56 edges, provably valid)
    best_adj = make_bipartite_base()
    best_count = edge_count(best_adj)

    def local_search_add(adj, max_time):
        adj = [set(x) for x in adj]
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for u in range(n):
                for v in range(u + 1, n):
                    if time.time() > max_time:
                        return adj
                    if v not in adj[u]:
                        adj[u].add(v)
                        adj[v].add(u)
                        if is_valid(adj):
                            improved = True
                        else:
                            adj[u].discard(v)
                            adj[v].discard(u)
        return adj

    def local_search_swap(adj, max_time):
        adj = [set(x) for x in adj]
        edges_in = [(u, v) for u in range(n) for v in adj[u] if u < v]
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for e1 in edges_in:
                if time.time() > max_time:
                    return adj
                a, b = e1
                # Try removing e1 and adding any missing edge
                adj[a].discard(b)
                adj[b].discard(a)
                for u in range(n):
                    for v in range(u + 1, n):
                        if time.time() > max_time:
                            adj[a].add(b)
                            adj[b].add(a)
                            return adj
                        if (u, v) == (a, b) or (u, v) == (b, a):
                            continue
                        if v not in adj[u]:
                            adj[u].add(v)
                            adj[v].add(u)
                            if is_valid(adj):
                                # Keep the swap
                                edges_in = [(x, y) for x in range(n) for y in adj[x] if x < y]
                                improved = True
                                break
                            else:
                                adj[u].discard(v)
                                adj[v].discard(u)
                    if improved:
                        break
                if not improved:
                    # Restore e1
                    adj[a].add(b)
                    adj[b].add(a)
        return adj

    # Multiple restarts with different strategies
    for trial in range(60):
        if time.time() > deadline - 10:
            break

        random.seed(trial)

        # Strategy 1: Start from bipartite, greedily add
        if trial % 3 == 0:
            cur_adj = make_bipartite_base()
            cur_adj = local_search_add(cur_adj, deadline - 5)
        # Strategy 2: Start from best, perturb and re-add
        elif trial % 3 == 1:
            cur_adj = [set(best_adj[i]) for i in range(n)]
            # Remove some random edges
            edges_in = [(u, v) for u in range(n) for v in cur_adj[u] if u < v]
            random.shuffle(edges_in)
            for e in edges_in[:random.randint(2, 5)]:
                cur_adj[e[0]].discard(e[1])
                cur_adj[e[1]].discard(e[0])
            cur_adj = local_search_add(cur_adj, deadline - 5)
        # Strategy 3: Start from best, try swaps
        else:
            cur_adj = [set(best_adj[i]) for i in range(n)]
            cur_adj = local_search_swap(cur_adj, deadline - 5)
            # Then try adding
            cur_adj = local_search_add(cur_adj, deadline - 5)

        cnt = edge_count(cur_adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = [set(cur_adj[i]) for i in range(n)]

    edges = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                edges.append([int(u), int(v)])

    return edges

import time
import random

def construct():
    n = 20
    start = time.time()
    deadline = start + 140.0

    def bit_count(m):
        return m.bit_count()

    def is_valid_add(adj, u, v):
        # 1. u and v must not share > 2 neighbours
        if bit_count(adj[u] & adj[v]) > 2:
            return False
        # 2. for all w in N(u)\{v}: |N(v) & N(w)| <= 1
        m = adj[u] & ~(1 << v)
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            if bit_count(adj[v] & adj[w]) > 1:
                return False
            m ^= lsb
        # 3. for all x in N(v)\{u}: |N(u) & N(x)| <= 1
        m = adj[v] & ~(1 << u)
        while m:
            lsb = m & -m
            x = lsb.bit_length() - 1
            if bit_count(adj[u] & adj[x]) > 1:
                return False
            m ^= lsb
        return True

    def edge_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            mask = adj[u]
            for v in range(u + 1, n):
                if (mask >> v) & 1:
                    edges.append([u, v])
        return edges

    def add_only(adj):
        adj = list(adj)
        for u in range(n):
            for v in range(u + 1, n):
                if not ((adj[u] >> v) & 1):
                    if is_valid_add(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        return adj

    def random_valid_graph(rng, target_deg=3):
        adj = [0] * n
        order = list(range(n))
        rng.shuffle(order)
        for i in order:
            cands = [j for j in range(n) if j != i and not ((adj[i] >> j) & 1)]
            rng.shuffle(cands)
            added = 0
            for j in cands:
                if added >= target_deg:
                    break
                if is_valid_add(adj, i, j):
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    added += 1
        return add_only(adj)

    def local_search(adj, rng, budget):
        adj = list(adj)
        t0 = time.time()
        improved = True
        while improved and (time.time() - t0) < budget:
            improved = False
            # add-only
            for u in range(n):
                for v in range(u + 1, n):
                    if not ((adj[u] >> v) & 1):
                        if is_valid_add(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            improved = True
            if improved:
                continue
            # rebuild lists
            present, missing = [], []
            for u in range(n):
                for v in range(u + 1, n):
                    (present if (adj[u] >> v) & 1 else missing).append((u, v))
            if not present or not missing:
                break
            # 1-swap
            found = False
            for _ in range(250):
                if time.time() - t0 > budget:
                    break
                u, v = rng.choice(present)
                x, y = rng.choice(missing)
                adj[u] &= ~(1 << v); adj[v] &= ~(1 << u)
                if is_valid_add(adj, x, y):
                    adj[x] |= (1 << y); adj[y] |= (1 << x)
                    improved = True; found = True
                    break
                else:
                    adj[u] |= (1 << v); adj[v] |= (1 << u)
            if found:
                continue
            break
        return adj

    # ---- Genetic algorithm ----
    rng = random.Random(20240517)
    pop_size = 20
    pop = [random_valid_graph(rng, target_deg=rng.choice([2, 3, 4])) for _ in range(pop_size)]
    pop = [local_search(g, rng, 0.3) for g in pop]

    best = max(pop, key=edge_count)
    best_cnt = edge_count(best)

    def tournament(k=3):
        idxs = rng.sample(range(len(pop)), k)
        return max(idxs, key=lambda i: edge_count(pop[i]))

    gen = 0
    while time.time() < deadline - 4.0:
        gen += 1
        newpop = []
        # elitism
        pop_sorted = sorted(pop, key=edge_count, reverse=True)
        newpop.append(list(pop_sorted[0]))
        # fill rest via GA
        while len(newpop) < pop_size and time.time() < deadline - 4.0:
            ia = tournament(); ib = tournament()
            A, B = pop[ia], pop[ib]
            # validity-preserving crossover: symmetric difference of edge sets
            child = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    in_a = (A[u] >> v) & 1
                    in_b = (B[u] >> v) & 1
                    if in_a != in_b:
                        child[u] |= (1 << v)
                        child[v] |= (1 << u)
            # mutation: add-only completion (deterministically valid)
            child = add_only(child)
            # occasional short local search
            if rng.random() < 0.5:
                child = local_search(child, rng, 0.15)
            newpop.append(child)
        pop = newpop
        g_best = max(pop, key=edge_count)
        gc = edge_count(g_best)
        if gc > best_cnt:
            best_cnt = gc
            best = list(g_best)
        # inject a fresh random individual occasionally to maintain diversity
        if gen % 5 == 0 and time.time() < deadline - 4.0:
            pop[-1] = random_valid_graph(rng, target_deg=rng.choice([3, 4, 5]))
            pop[-1] = local_search(pop[-1], rng, 0.2)

    # Final polish on the best
    if time.time() < deadline - 2.0:
        best = local_search(best, random.Random(999), 1.5)
        best = add_only(best)

    edges = adj_to_edges(best)

    # Independent verification
    adj = [0] * n
    ok = True
    for u, v in edges:
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            ok = False; break
        adj[u] |= (1 << v); adj[v] |= (1 << u)
    if ok:
        for u in range(n):
            if adj[u] & (1 << u):
                ok = False; break
    if ok:
        for u in range(n):
            for v in range(u + 1, n):
                if bit_count(adj[u] & adj[v]) > 2:
                    ok = False; break
            if not ok:
                break
    if not ok:
        # fallback: simple greedy
        fa = [0] * n
        rr = random.Random(0)
        all_e = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rr.shuffle(all_e)
        for u, v in all_e:
            if is_valid_add(fa, u, v):
                fa[u] |= (1 << v); fa[v] |= (1 << u)
        edges = adj_to_edges(fa)
    return edges

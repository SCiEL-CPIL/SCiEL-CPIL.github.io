import time
import random
import math
import numpy as np

def construct():
    n = 20
    start_time = time.time()
    time_limit = 145.0

    def popcount(x):
        return bin(x).count('1')

    def count_edges(adj):
        return sum(popcount(x) for x in adj) // 2

    def is_k23_free(adj):
        for u in range(n):
            for v in range(u + 1, n):
                if popcount(adj[u] & adj[v]) >= 3:
                    return False
        return True

    def valid_add(adj, u, v, common, nbrs):
        if adj[u] & (1 << v):
            return False
        if common[u][v] > 2:
            return False
        for w in nbrs[u]:
            if common[w][v] > 1:
                return False
        for w in nbrs[v]:
            if common[w][u] > 1:
                return False
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def build_common(adj):
        common = [[0] * n for _ in range(n)]
        for a in range(n):
            for b in range(a + 1, n):
                c = popcount(adj[a] & adj[b])
                common[a][b] = c
                common[b][a] = c
        return common

    def build_nbrs(adj):
        nbrs = [[] for _ in range(n)]
        for u in range(n):
            temp = adj[u]
            lst = []
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                lst.append(v)
                temp -= lsb
            nbrs[u] = lst
        return nbrs

    def get_edges(adj):
        edges = []
        for u in range(n):
            for v in nbrs_get(adj, u):
                if u < v:
                    edges.append((u, v))
        return edges

    def nbrs_get(adj, u):
        temp = adj[u]
        lst = []
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            lst.append(v)
            temp -= lsb
        return lst

    def get_non_edges(adj):
        ne = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    ne.append((u, v))
        return ne

    def greedy_fill(adj, common, nbrs):
        changed = True
        while changed and (time.time() - start_time < time_limit):
            changed = False
            ne = get_non_edges(adj)
            random.shuffle(ne)
            for u, v in ne:
                if valid_add(adj, u, v, common, nbrs):
                    add_edge(adj, u, v)
                    # update common
                    for w in range(n):
                        if w != u and w != v:
                            common[u][w] = popcount(adj[u] & adj[w])
                            common[w][u] = common[u][w]
                            common[v][w] = popcount(adj[v] & adj[w])
                            common[w][v] = common[v][w]
                    common[u][v] = popcount(adj[u] & adj[v])
                    common[v][u] = common[u][v]
                    nbrs[u] = nbrs_get(adj, u)
                    nbrs[v] = nbrs_get(adj, v)
                    changed = True

    def make_k4_union():
        adj = [0] * n
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    add_edge(adj, base + a, base + b)
        return adj

    def make_random_start():
        adj = [0] * n
        # start with a few random K4s and some random edges
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    if random.random() < 0.9:
                        add_edge(adj, base + a, base + b)
        # add some random edges
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)) and random.random() < 0.15:
                    add_edge(adj, u, v)
        return adj

    best_edges = 0
    best_adj = None
    best_common = None
    best_nbrs = None

    # Baseline
    adj0 = make_k4_union()
    c0 = build_common(adj0)
    nb0 = build_nbrs(adj0)
    greedy_fill(adj0, c0, nb0)
    best_edges = count_edges(adj0)
    best_adj = adj0[:]
    best_common = [row[:] for row in c0]
    best_nbrs = [row[:] for row in nb0]

    # Multi-restart simulated annealing
    num_restarts = 100
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break

        if r % 10 == 0:
            adj = make_k4_union()
        elif r % 10 == 5:
            adj = make_random_start()
        else:
            adj = best_adj[:]
            edges = get_edges(adj)
            random.shuffle(edges)
            for u, v in edges[:2]:
                remove_edge(adj, u, v)

        common = build_common(adj)
        nbrs = build_nbrs(adj)
        greedy_fill(adj, common, nbrs)
        current_edges = count_edges(adj)

        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]
            best_common = [row[:] for row in common]
            best_nbrs = [row[:] for row in nbrs]

        # Simulated annealing
        T = 0.8
        T_min = 0.05
        cooling = 0.997

        while T > T_min and (time.time() - start_time < time_limit):
            move_type = random.random()

            if move_type < 0.40:
                ne = get_non_edges(adj)
                if not ne:
                    T *= cooling
                    continue
                u, v = random.choice(ne)
                if valid_add(adj, u, v, common, nbrs):
                    add_edge(adj, u, v)
                    for w in range(n):
                        if w != u and w != v:
                            common[u][w] = popcount(adj[u] & adj[w])
                            common[w][u] = common[u][w]
                            common[v][w] = popcount(adj[v] & adj[w])
                            common[w][v] = common[v][w]
                    common[u][v] = popcount(adj[u] & adj[v])
                    common[v][u] = common[u][v]
                    nbrs[u] = nbrs_get(adj, u)
                    nbrs[v] = nbrs_get(adj, v)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                        best_common = [row[:] for row in common]
                        best_nbrs = [row[:] for row in nbrs]
            elif move_type < 0.60:
                edges = get_edges(adj)
                if not edges:
                    T *= cooling
                    continue
                u, v = random.choice(edges)
                if random.random() < math.exp(-1.0 / T):
                    remove_edge(adj, u, v)
                    for w in range(n):
                        if w != u and w != v:
                            common[u][w] = popcount(adj[u] & adj[w])
                            common[w][u] = common[u][w]
                            common[v][w] = popcount(adj[v] & adj[w])
                            common[w][v] = common[v][w]
                    common[u][v] = popcount(adj[u] & adj[v])
                    common[v][u] = common[u][v]
                    nbrs[u] = nbrs_get(adj, u)
                    nbrs[v] = nbrs_get(adj, v)
                    current_edges -= 1
            elif move_type < 0.85:
                edges = get_edges(adj)
                ne = get_non_edges(adj)
                if not edges or not ne:
                    T *= cooling
                    continue
                u1, v1 = random.choice(edges)
                u2, v2 = random.choice(ne)
                remove_edge(adj, u1, v1)
                for w in range(n):
                    if w != u1 and w != v1:
                        common[u1][w] = popcount(adj[u1] & adj[w])
                        common[w][u1] = common[u1][w]
                        common[v1][w] = popcount(adj[v1] & adj[w])
                        common[w][v1] = common[v1][w]
                common[u1][v1] = popcount(adj[u1] & adj[v1])
                common[v1][u1] = common[u1][v1]
                nbrs[u1] = nbrs_get(adj, u1)
                nbrs[v1] = nbrs_get(adj, v1)

                if valid_add(adj, u2, v2, common, nbrs):
                    add_edge(adj, u2, v2)
                    for w in range(n):
                        if w != u2 and w != v2:
                            common[u2][w] = popcount(adj[u2] & adj[w])
                            common[w][u2] = common[u2][w]
                            common[v2][w] = popcount(adj[v2] & adj[w])
                            common[w][v2] = common[v2][w]
                    common[u2][v2] = popcount(adj[u2] & adj[v2])
                    common[v2][u2] = common[u2][v2]
                    nbrs[u2] = nbrs_get(adj, u2)
                    nbrs[v2] = nbrs_get(adj, v2)
                else:
                    add_edge(adj, u1, v1)
                    for w in range(n):
                        if w != u1 and w != v1:
                            common[u1][w] = popcount(adj[u1] & adj[w])
                            common[w][u1] = common[u1][w]
                            common[v1][w] = popcount(adj[v1] & adj[w])
                            common[w][v1] = common[v1][w]
                    common[u1][v1] = popcount(adj[u1] & adj[v1])
                    common[v1][u1] = common[u1][v1]
                    nbrs[u1] = nbrs_get(adj, u1)
                    nbrs[v1] = nbrs_get(adj, v1)
            else:
                edges = get_edges(adj)
                ne = get_non_edges(adj)
                if len(edges) < 2 or len(ne) < 2:
                    T *= cooling
                    continue
                idx1, idx2 = random.sample(range(len(edges)), 2)
                u1, v1 = edges[idx1]
                u2, v2 = edges[idx2]
                idx3, idx4 = random.sample(range(len(ne)), 2)
                u3, v3 = ne[idx3]
                u4, v4 = ne[idx4]

                remove_edge(adj, u1, v1)
                remove_edge(adj, u2, v2)
                # rebuild common and nbrs for the 4 affected vertices and their interactions
                affected = set([u1, v1, u2, v2])
                for a in affected:
                    for b in range(n):
                        if a != b:
                            common[a][b] = popcount(adj[a] & adj[b])
                    nbrs[a] = nbrs_get(adj, a)

                ok3 = valid_add(adj, u3, v3, common, nbrs)
                if ok3:
                    add_edge(adj, u3, v3)
                    affected2 = set([u3, v3])
                    for a in affected2:
                        for b in range(n):
                            if a != b:
                                common[a][b] = popcount(adj[a] & adj[b])
                        nbrs[a] = nbrs_get(adj, a)
                    ok4 = valid_add(adj, u4, v4, common, nbrs)
                    if ok4:
                        add_edge(adj, u4, v4)
                        affected3 = set([u4, v4])
                        for a in affected3:
                            for b in range(n):
                                if a != b:
                                    common[a][b] = popcount(adj[a] & adj[b])
                            nbrs[a] = nbrs_get(adj, a)
                        if random.random() < 0.7:
                            pass
                        else:
                            remove_edge(adj, u4, v4)
                            remove_edge(adj, u3, v3)
                            add_edge(adj, u1, v1)
                            add_edge(adj, u2, v2)
                            common = build_common(adj)
                            nbrs = build_nbrs(adj)
                    else:
                        remove_edge(adj, u3, v3)
                        add_edge(adj, u1, v1)
                        add_edge(adj, u2, v2)
                        common = build_common(adj)
                        nbrs = build_nbrs(adj)
                else:
                    add_edge(adj, u1, v1)
                    add_edge(adj, u2, v2)
                    common = build_common(adj)
                    nbrs = build_nbrs(adj)

            T *= cooling

    # Final greedy fill on best
    if best_adj:
        c = build_common(best_adj)
        nb = build_nbrs(best_adj)
        greedy_fill(best_adj, c, nb)
        final_edges = count_edges(best_adj)
        if final_edges > best_edges:
            best_edges = final_edges

    # Polish pass
    if best_adj and (time.time() - start_time < time_limit - 8):
        polish_adj = best_adj[:]
        polish_common = build_common(polish_adj)
        polish_nbrs = build_nbrs(polish_adj)
        polish_edges = count_edges(polish_adj)
        improved = True
        polish_iter = 0
        while improved and (time.time() - start_time < time_limit - 8) and polish_iter < 15:
            polish_iter += 1
            improved = False
            # Try adding any non-edge
            for u in range(n):
                for v in range(u + 1, n):
                    if not (polish_adj[u] & (1 << v)):
                        if valid_add(polish_adj, u, v, polish_common, polish_nbrs):
                            add_edge(polish_adj, u, v)
                            for w in range(n):
                                if w != u and w != v:
                                    polish_common[u][w] = popcount(polish_adj[u] & polish_adj[w])
                                    polish_common[w][u] = polish_common[u][w]
                                    polish_common[v][w] = popcount(polish_adj[v] & polish_adj[w])
                                    polish_common[w][v] = polish_common[v][w]
                            polish_common[u][v] = popcount(polish_adj[u] & polish_adj[v])
                            polish_common[v][u] = polish_common[u][v]
                            polish_nbrs[u] = nbrs_get(polish_adj, u)
                            polish_nbrs[v] = nbrs_get(polish_adj, v)
                            polish_edges += 1
                            improved = True
            # Try 1-1 swaps
            edges = get_edges(polish_adj)
            ne = get_non_edges(polish_adj)
            for a, b in edges:
                if time.time() - start_time > time_limit - 8:
                    break
                for c, d in ne:
                    if (a, b) == (c, d) or (a, b) == (d, c):
                        continue
                    remove_edge(polish_adj, a, b)
                    for w in range(n):
                        if w != a and w != b:
                            polish_common[a][w] = popcount(polish_adj[a] & polish_adj[w])
                            polish_common[w][a] = polish_common[a][w]
                            polish_common[b][w] = popcount(polish_adj[b] & polish_adj[w])
                            polish_common[w][b] = polish_common[b][w]
                    polish_common[a][b] = popcount(polish_adj[a] & polish_adj[b])
                    polish_common[b][a] = polish_common[a][b]
                    polish_nbrs[a] = nbrs_get(polish_adj, a)
                    polish_nbrs[b] = nbrs_get(polish_adj, b)

                    if valid_add(polish_adj, c, d, polish_common, polish_nbrs):
                        add_edge(polish_adj, c, d)
                        for w in range(n):
                            if w != c and w != d:
                                polish_common[c][w] = popcount(polish_adj[c] & polish_adj[w])
                                polish_common[w][c] = polish_common[c][w]
                                polish_common[d][w] = popcount(polish_adj[d] & polish_adj[w])
                                polish_common[w][d] = polish_common[d][w]
                        polish_common[c][d] = popcount(polish_adj[c] & polish_adj[d])
                        polish_common[d][c] = polish_common[c][d]
                        polish_nbrs[c] = nbrs_get(polish_adj, c)
                        polish_nbrs[d] = nbrs_get(polish_adj, d)
                        # try to add more
                        further = False
                        for x in range(n):
                            for y in range(x + 1, n):
                                if not (polish_adj[x] & (1 << y)):
                                    if valid_add(polish_adj, x, y, polish_common, polish_nbrs):
                                        add_edge(polish_adj, x, y)
                                        for w in range(n):
                                            if w != x and w != y:
                                                polish_common[x][w] = popcount(polish_adj[x] & polish_adj[w])
                                                polish_common[w][x] = polish_common[x][w]
                                                polish_common[y][w] = popcount(polish_adj[y] & polish_adj[w])
                                                polish_common[w][y] = polish_common[y][w]
                                        polish_common[x][y] = popcount(polish_adj[x] & polish_adj[y])
                                        polish_common[y][x] = polish_common[x][y]
                                        polish_nbrs[x] = nbrs_get(polish_adj, x)
                                        polish_nbrs[y] = nbrs_get(polish_adj, y)
                                        polish_edges += 1
                                        further = True
                        if further:
                            improved = True
                            break
                        else:
                            remove_edge(polish_adj, c, d)
                            add_edge(polish_adj, a, b)
                            polish_common = build_common(polish_adj)
                            polish_nbrs = build_nbrs(polish_adj)
                    else:
                        add_edge(polish_adj, a, b)
                        polish_common = build_common(polish_adj)
                        polish_nbrs = build_nbrs(polish_adj)
                if improved:
                    break
            if polish_edges > best_edges:
                best_edges = polish_edges
                best_adj = polish_adj[:]

    result = []
    for u in range(n):
        for v in nbrs_get(best_adj, u):
            if u < v:
                result.append([u, v])
    return result

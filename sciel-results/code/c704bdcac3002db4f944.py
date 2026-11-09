import time
import random

def construct():
    deadline = time.time() + 110

    def find_c5(adj, n):
        full_mask = (1 << n) - 1
        for a in range(n):
            for b in range(a + 1, n):
                if not (adj[a] >> b) & 1:
                    continue
                c_mask = adj[b] & ~(1 << a) & full_mask
                while c_mask:
                    c_bit = c_mask & (-c_mask)
                    c = c_bit.bit_length() - 1
                    c_mask ^= c_bit
                    d_mask = adj[c] & ~(1 << a) & ~(1 << b) & ~(1 << c) & full_mask
                    while d_mask:
                        d_bit = d_mask & (-d_mask)
                        d = d_bit.bit_length() - 1
                        d_mask ^= d_bit
                        e_mask = adj[d] & adj[a] & ~(1 << a) & ~(1 << b) & ~(1 << c) & ~(1 << d) & full_mask
                        if e_mask:
                            e = (e_mask & (-e_mask)).bit_length() - 1
                            return (a, b, c, d, e)
        return None

    def approx_alpha(adj, n):
        best = 0
        for _ in range(30):
            order = list(range(n))
            random.shuffle(order)
            sel = []
            for v in order:
                if all(not (adj[v] >> w) & 1 for w in sel):
                    sel.append(v)
            best = max(best, len(sel))
        return best

    # Start with 7 K4 cliques (28 vertices) + 1 K5 clique (5 vertices) = 33 vertices
    best_n = 33
    best_adj = [0] * 33
    for b in range(7):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                best_adj[base + i] |= (1 << (base + j))
                best_adj[base + j] |= (1 << (base + i))
    # K5 on vertices 28-32
    base = 28
    for i in range(5):
        for j in range(i + 1, 5):
            best_adj[base + i] |= (1 << (base + j))
            best_adj[base + j] |= (1 << (base + i))

    for target_n in range(34, 36):
        if time.time() > deadline:
            break
        found = False
        for attempt in range(5000):
            if time.time() > deadline:
                break
            adj = best_adj[:] + [0] * (target_n - best_n)
            for i in range(best_n, target_n):
                for j in range(target_n):
                    if j > i and random.random() < 0.45:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            for _ in range(1200):
                if time.time() > deadline:
                    break
                c5 = find_c5(adj, target_n)
                if c5:
                    a, b, c, d, e = c5
                    u, v = random.choice([(a, b), (b, c), (c, d), (d, e), (e, a)])
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    alpha_val = approx_alpha(adj, target_n)
                    if alpha_val < 9:
                        found = True
                        best_n = target_n
                        best_adj = adj[:]
                        break
                    else:
                        order = list(range(target_n))
                        random.shuffle(order)
                        indep = []
                        for v in order:
                            if all(not (adj[v] >> w) & 1 for w in indep):
                                indep.append(v)
                            if len(indep) >= 9:
                                break
                        if len(indep) >= 2:
                            u, v = random.sample(indep, 2)
                            if not ((adj[u] >> v) & 1):
                                adj[u] |= (1 << v)
                                adj[v] |= (1 << u)
            if found:
                break
        if found:
            break

    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
    return {'n': best_n, 'edges': edges}

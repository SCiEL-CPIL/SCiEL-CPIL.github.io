"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 105
    n = 18

    def edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def has_k4(adj):
        for a in range(n):
            for b in range(a + 1, n):
                if not (adj[a] >> b) & 1:
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if (common >> c) & 1:
                        if (common & adj[c]) != 0:
                            return True
        return False

    def max_triangle_free_size(adj, time_limit=5.0):
        best = [0]
        deg = [bin(adj[v]).count('1') for v in range(n)]
        order = sorted(range(n), key=lambda v: -deg[v])
        t0 = time.time()

        def backtrack(idx, current_mask, current_size):
            if time.time() - t0 > time_limit:
                return
            if idx == n:
                if current_size > best[0]:
                    best[0] = current_size
                return
            if current_size + (n - idx) <= best[0]:
                return
            v = order[idx]
            # Don't include v
            backtrack(idx + 1, current_mask, current_size)
            # Include v
            neighbors_in_S = current_mask & adj[v]
            has_edge = False
            tmp = neighbors_in_S
            while tmp:
                bit = tmp & (-tmp)
                u = bit.bit_length() - 1
                rest = tmp & ~bit
                if (adj[u] & rest) != 0:
                    has_edge = True
                    break
                tmp = rest
            if not has_edge:
                backtrack(idx + 1, current_mask | (1 << v), current_size + 1)

        backtrack(0, 0, 0)
        return best[0]

    random.seed(42)
    candidates = []

    # K_{6,6,6}
    adj1 = [0] * n
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj1[a] |= (1 << b)
                    adj1[b] |= (1 << a)
    candidates.append(adj1)

    # Random dense K4-free graphs
    for trial in range(50):
        if time.time() > deadline - 20:
            break
        adj = [0] * n
        density = 0.5 + random.random() * 0.4
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        for _ in range(200):
            if not has_k4(adj):
                break
            found = False
            for a in range(n):
                for b in range(a + 1, n):
                    if not (adj[a] >> b) & 1:
                        continue
                    common = adj[a] & adj[b]
                    for c in range(b + 1, n):
                        if (common >> c) & 1:
                            common_abc = common & adj[c]
                            if common_abc:
                                d_bit = common_abc & (-common_abc)
                                d = d_bit.bit_length() - 1
                                pair = random.choice([(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)])
                                x, y = pair
                                adj[x] &= ~(1 << y)
                                adj[y] &= ~(1 << x)
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
        if not has_k4(adj):
            candidates.append(adj)

    best_adj = None
    best_score = 18
    for adj in candidates:
        if time.time() > deadline - 15:
            break
        score = max_triangle_free_size(adj, time_limit=3.0)
        if score < best_score:
            best_score = score
            best_adj = adj[:]

    if best_adj is None:
        best_adj = candidates[0]
        best_score = max_triangle_free_size(best_adj, time_limit=3.0)

    adj = best_adj[:]
    for iteration in range(1000):
        if time.time() > deadline - 5:
            break
        if best_score <= 4:
            break
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        old_adj = adj[:]
        if (adj[i] >> j) & 1:
            adj[i] &= ~(1 << j)
            adj[j] &= ~(1 << i)
        else:
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
        if has_k4(adj):
            adj = old_adj
            continue
        score = max_triangle_free_size(adj, time_limit=2.0)
        if score < best_score:
            best_score = score
            best_adj = adj[:]
        else:
            adj = old_adj

    return edges_from_adj(best_adj)
# EVOLVE-BLOCK-END

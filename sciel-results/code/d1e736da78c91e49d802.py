"""Construct a dense C4-rainbow edge-coloured graph on 20 vertices."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 140.0
    n = 20
    num_colours = 20

    def time_ok():
        return time.time() - start < deadline - 5.0

    def build_greedy(seed):
        rng = random.Random(seed)
        edge_color = {}
        adj = {i: {} for i in range(n)}

        def add_edge(u, v, c):
            if u > v:
                u, v = v, u
            edge_color[(u, v)] = c
            adj[u][v] = c
            adj[v][u] = c

        def is_valid_add(u, v, c):
            # 4-cycle u-v-w-x-u: w in adj[v] (w!=u), x in adj[u] (x not in {v,w}), edge (w,x)
            for w, cw in adj[v].items():
                if w == u:
                    continue
                for x, cxu in adj[u].items():
                    if x == v or x == w:
                        continue
                    if x not in adj[w]:
                        continue
                    cwx = adj[w][x]
                    if len({c, cw, cwx, cxu}) != 4:
                        return False
            return True

        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(all_pairs)

        for (u, v) in all_pairs:
            if not time_ok():
                break
            cols = list(range(num_colours))
            rng.shuffle(cols)
            for c in cols:
                if is_valid_add(u, v, c):
                    add_edge(u, v, c)
                    break
        return edge_color

    best = None
    best_count = -1
    for seed in range(300):
        if not time_ok():
            break
        ec = build_greedy(seed)
        cnt = len(ec)
        if cnt > best_count:
            best_count = cnt
            best = ec
        if best_count >= 185:
            break

    if best is None:
        return [[0, v, 0] for v in range(1, n)]

    return [[u, v, c] for (u, v), c in best.items()]
# EVOLVE-BLOCK-END

import sys
import itertools
import time
import random

N = 16
E = N * (N - 1) // 2  # 120

_edges = []
_idx = {}
for i in range(N):
    for j in range(i + 1, N):
        _idx[(i, j)] = len(_edges)
        _edges.append((i, j))

def _cycles_for_quad(quad):
    matchings = [((0,1),(2,3)), ((0,2),(1,3)), ((0,3),(1,2))]
    all6 = set()
    for x in range(4):
        for y in range(x + 1, 4):
            all6.add((x, y))
    cyc_edges = []
    for m in matchings:
        diag = set()
        for (x, y) in m:
            diag.add((x, y))
        ce = all6 - diag
        edge_set = set()
        for (x, y) in ce:
            u = quad[x]; v = quad[y]
            if u > v: u, v = v, u
            edge_set.add((u, v))
        cyc_edges.append(edge_set)
    return cyc_edges

_all_quads = list(itertools.combinations(range(N), 4))
_cycles = []
for q in _all_quads:
    for es in _cycles_for_quad(q):
        _cycles.append(tuple(_idx[e] for e in es))

_edge_to_cycles = [[] for _ in range(E)]
for ci, cyc in enumerate(_cycles):
    for e in cyc:
        _edge_to_cycles[e].append(ci)

def count_distinct(colors):
    return len(set(colors))

def is_valid(colors):
    for cyc in _cycles:
        a = colors[cyc[0]]; b = colors[cyc[1]]
        c = colors[cyc[2]]; d = colors[cyc[3]]
        if a != b and a != c and a != d and b != c and b != d and c != d:
            return False
    return True

def _base_color():
    cols = [0] * E
    for (i, j), e in _idx.items():
        cols[e] = min(i, j)
    return cols

def _try_add_new_color(colors, edge, new_color):
    for ci in _edge_to_cycles[edge]:
        cyc = _cycles[ci]
        cs = [colors[x] for x in cyc]
        pos = cyc.index(edge)
        cs[pos] = new_color
        if len(set(cs)) == 4:
            return None
    new = colors[:]
    new[edge] = new_color
    return new

def construct():
    deadline = time.time() + 82.0

    best = _base_color()
    best_count = count_distinct(best)
    assert is_valid(best)

    edge_order = sorted(range(E), key=lambda e: _edges[e][0], reverse=True)

    colors = best[:]
    used = set(colors)
    for e in edge_order:
        if time.time() > deadline - 6.0:
            break
        for nc in range(14, 120):
            if nc in used:
                continue
            trial = _try_add_new_color(colors, e, nc)
            if trial is not None:
                colors = trial
                used.add(nc)
                break

    if count_distinct(colors) > best_count and is_valid(colors):
        best = colors
        best_count = count_distinct(best)

    rng = random.Random(2024)

    def improve(cols, used_set):
        cnt = len(used_set)
        progress = True
        while progress and time.time() < deadline - 2.0:
            progress = False
            order = list(range(E))
            rng.shuffle(order)
            for e in order:
                for nc in range(14, 120):
                    if nc in used_set:
                        continue
                    trial = _try_add_new_color(cols, e, nc)
                    if trial is not None:
                        cols = trial
                        used_set.add(nc)
                        cnt += 1
                        progress = True
                        break
                if progress:
                    break
        return cols, used_set, cnt

    for restart in range(500):
        if time.time() > deadline - 2.0:
            break
        start = best[:]
        used_set = set(start)
        k = rng.randint(0, 4)
        for _ in range(k):
            e = rng.choice(edge_order[:12])
            start[e] = _edges[e][0]
        cols, used_set, cnt = improve(start, used_set)
        if cnt > best_count and is_valid(cols):
            best = cols
            best_count = cnt

    return list(best)

if __name__ == "__main__":
    c = construct()
    print("distinct:", count_distinct(c), "valid:", is_valid(c), "len:", len(c))

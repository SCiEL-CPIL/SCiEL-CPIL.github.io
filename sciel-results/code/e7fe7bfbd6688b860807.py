import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

_TREES = instance_trees()

def _adjacency(edges, k):
    adj = [[] for _ in range(k)]
    for a, b in edges:
        adj[a].append(b); adj[b].append(a)
    return adj

_ADJS = [_adjacency(e, i + 1) for i, e in enumerate(_TREES)]
N = 20

def _edge_key(a, b):
    return (a, b) if a < b else (b, a)

def _embed_gen(tidx, used, rng, node_budget):
    """Yield (mapping, added_edges) for valid edge-disjoint embeddings of tree tidx."""
    k = tidx + 1
    adj = _ADJS[tidx]
    if k == 1:
        yield [0], set()
        return
    adj_ord = [list(adj[v]) for v in range(k)]
    for v in range(k):
        rng.shuffle(adj_ord[v])
    img_order = list(range(N)); rng.shuffle(img_order)
    mapping = [-1] * k
    used_local = set(used)
    used_img = set()
    counter = [0]

    def dfs(v):
        if counter[0] > node_budget:
            return
        for nb in adj_ord[v]:
            if mapping[nb] != -1:
                continue
            pv = mapping[v]
            for img in img_order:
                if img in used_img:
                    continue
                e = _edge_key(pv, img)
                if e in used_local:
                    continue
                counter[0] += 1
                if counter[0] > node_budget:
                    return
                used_img.add(img); used_local.add(e); mapping[nb] = img
                if all(m != -1 for m in mapping):
                    yield list(mapping), set(used_local - used)
                else:
                    for sol in dfs(nb):
                        yield sol
                mapping[nb] = -1; used_local.discard(e); used_img.discard(img)
                if counter[0] > node_budget:
                    return

    for root_img in img_order:
        mapping[0] = root_img
        used_img.add(root_img)
        for sol in dfs(0):
            yield sol
        mapping[0] = -1
        used_img.discard(root_img)
        if counter[0] > node_budget:
            return

def construct():
    deadline = time.time() + 115  # safety margin under 150s kill
    best = None
    best_val = -1

    def value_of(res):
        return sum(i for i in range(20) if res[i] is not None)

    orders = [list(range(20)), list(range(19, -1, -1))]
    r0 = Random(999)
    for _ in range(80):
        o = list(range(20)); r0.shuffle(o); orders.append(o)

    # Phase 1: greedy embedding (first valid mapping per tree) over many orders
    for oi, order in enumerate(orders):
        if time.time() > deadline:
            break
        rng = Random(1000 + oi)
        used = set()
        res = [None] * 20
        ok = True
        for tidx in order:
            if time.time() > deadline:
                ok = False; break
            it = _embed_gen(tidx, used, rng, 40000)
            m = next(it, None)
            if m is None:
                ok = False; break
            mapping, added = m
            used.update(added)
            res[tidx] = [int(x) for x in mapping]
        val = value_of(res)
        if val > best_val:
            best_val = val
            best = [list(x) if x is not None else None for x in res]
        if ok:
            return best

    # Phase 2: full backtracking fallback if greedy did not reach 190
    if best_val < 190:
        bt_orders = [list(range(20)), list(range(19, -1, -1))]
        rb = Random(777)
        for _ in range(4):
            o = list(range(20)); rb.shuffle(o); bt_orders.append(o)
        for bi, order in enumerate(bt_orders):
            if time.time() > deadline - 2:
                break
            rng = Random(5000 + bi)
            used = set()
            res = [None] * 20

            def rec(pos):
                if time.time() > deadline - 2:
                    return False
                if pos == 20:
                    return True
                tidx = order[pos]
                for mapping, added in _embed_gen(tidx, used, rng, 60000):
                    used.update(added)
                    res[tidx] = mapping
                    if rec(pos + 1):
                        return True
                    used.difference_update(added)
                    res[tidx] = None
                return False

            if rec(0):
                best = [list(int(x) for x in m) if m is not None else None for m in res]
                best_val = 190
                return best

    return best

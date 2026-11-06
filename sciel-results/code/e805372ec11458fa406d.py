"""Combined program: K3,13 structural floor (inspiration) + fast signature-based
Q3 subgraph-isomorphism check + greedy/local/random refinement (current)."""
import time
import random
from itertools import permutations

# --- Static cube data (computed once) ---
_CUBE_ADJ = [[0] * 8 for _ in range(8)]
for a in range(8):
    for b in range(8):
        if a != b and (a ^ b) in (1, 2, 4):
            _CUBE_ADJ[a][b] = 1
_CUBE_DEG = [3] * 8

def _embedding_exists(sig):
    """Return True if the 3-cube embeds into the 8-vertex graph given by
    sig[i][j] (1 iff edge between set-positions i and j)."""
    # degree prefilter: every cube vertex has degree 3, so each set-vertex
    # that hosts it must have degree >= 3.
    eligible = [i for i in range(8)
                if sum(sig[i][j] for j in range(8) if j != i) >= 3]
    if len(eligible) < 8:
        return False
    # try all bijections cube -> set (restricted to eligible) with pruning
    for g in permutations(eligible):
        ok = True
        for a in range(8):
            sa = g[a]
            for b in range(8):
                if _CUBE_ADJ[a][b] and not sig[sa][g[b]]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return True
    return False


def _q3_free(edges):
    """edges: set of (u,v) with u<v. Return True if no Q3 subgraph."""
    present = edges
    # enumerate all 8-subsets lazily via combinations
    for vs in _COMB8:
        # degree prefilter
        deg = [0] * 8
        for i in range(8):
            ui = vs[i]
            for j in range(8):
                if i == j:
                    continue
                vj = vs[j]
                a, b = (ui, vj) if ui < vj else (vj, ui)
                if (a, b) in present:
                    deg[i] += 1
        if any(d < 3 for d in deg):
            continue
        # build signature
        sig = [[0] * 8 for _ in range(8)]
        for i in range(8):
            ui = vs[i]
            for j in range(8):
                if i == j:
                    continue
                vj = vs[j]
                a, b = (ui, vj) if ui < vj else (vj, ui)
                if (a, b) in present:
                    sig[i][j] = 1
        if _embedding_exists(sig):
            return False
    return True


# precompute all 8-subsets of 0..15
_COMB8 = list(__import__('itertools').combinations(range(16), 8))


def construct():
    start_time = time.time()
    deadline = start_time + 112.0   # safety margin under 150s kill

    n = 16

    # ---- Structural floor from inspiration: K3,13 (39 edges), Q3-free ----
    def make_k313():
        return set((a, b) for a in range(3) for b in range(3, 16))

    best_edges = make_k313()
    best_count = len(best_edges)

    # ---- Fast candidate-edge Q3 test: does adding edge create a Q3? ----
    def adding_creates_q3(cur, u, v):
        e = (u, v) if u < v else (v, u)
        if e in cur:
            return False
        test = cur | {e}
        # only 8-sets containing both u and v can be affected
        for vs in _COMB8:
            if u not in vs or v not in vs:
                continue
            deg = [0] * 8
            for i in range(8):
                ui = vs[i]
                for j in range(8):
                    if i == j:
                        continue
                    vj = vs[j]
                    a, b = (ui, vj) if ui < vj else (vj, ui)
                    if (a, b) in test:
                        deg[i] += 1
            if any(d < 3 for d in deg):
                continue
            sig = [[0] * 8 for _ in range(8)]
            for i in range(8):
                ui = vs[i]
                for j in range(8):
                    if i == j:
                        continue
                    vj = vs[j]
                    a, b = (ui, vj) if ui < vj else (vj, ui)
                    if (a, b) in test:
                        sig[i][j] = 1
            if _embedding_exists(sig):
                return True
        return False

    # ---- Greedy edge addition (prioritize within-part then cross) ----
    cur = set(best_edges)
    all_pairs = [(i, j) for i in range(16) for j in range(i + 1, 16)
                 if (i, j) not in cur]

    def prio(e):
        u, v = e
        if u < 3 and v < 3:
            return 0
        if u >= 3 and v >= 3:
            return 1
        return 2

    all_pairs.sort(key=prio)
    for e in all_pairs:
        if time.time() > deadline - 30:
            break
        if not adding_creates_q3(cur, e[0], e[1]):
            cur.add(e)
    if len(cur) > best_count:
        best_count = len(cur)
        best_edges = set(cur)

    # ---- Local improvement: add + remove (simulated-anneal style) ----
    def local_search(start, budget):
        cur = set(start)
        best = (len(cur), set(cur))
        t = time.time()
        improved = True
        while improved and time.time() < budget:
            improved = False
            # try to add every missing edge
            missing = [(i, j) for i in range(16) for j in range(i + 1, 16)
                       if (i, j) not in cur]
            random.shuffle(missing)
            for e in missing:
                if time.time() > budget:
                    break
                if not adding_creates_q3(cur, e[0], e[1]):
                    cur.add(e)
                    improved = True
                    if len(cur) > best[0]:
                        best = (len(cur), set(cur))
            # try to remove an edge then add two (swap to escape local optima)
            if time.time() > budget:
                break
            edges_list = list(cur)
            random.shuffle(edges_list)
            for rem in edges_list[:8]:
                if time.time() > budget:
                    break
                cur.discard(rem)
                before = len(cur)
                # greedily add back as many as possible
                miss2 = [(i, j) for i in range(16) for j in range(i + 1, 16)
                         if (i, j) not in cur]
                random.shuffle(miss2)
                for e in miss2:
                    if time.time() > budget:
                        break
                    if not adding_creates_q3(cur, e[0], e[1]):
                        cur.add(e)
                if len(cur) > best[0]:
                    best = (len(cur), set(cur))
                # keep the better of (cur, cur+rem)
                if len(cur) < before:
                    cur.add(rem)
        return best

    # seed local search from the best structure found
    budget = deadline - 12
    cnt, edg = local_search(best_edges, budget)
    if cnt > best_count:
        best_count = cnt
        best_edges = edg

    # ---- Random dense graphs (extra coverage) ----
    random.seed(123)
    for trial in range(200):
        if time.time() > deadline - 3:
            break
        p = random.uniform(0.55, 0.72)
        edges = set()
        for i in range(16):
            for j in range(i + 1, 16):
                if random.random() < p:
                    edges.add((i, j))
        if len(edges) > best_count and _q3_free(edges):
            best_count = len(edges)
            best_edges = edges

    # ---- Final safety: verify the best is Q3-free before returning ----
    if not _q3_free(best_edges):
        best_edges = make_k313()
        best_count = len(best_edges)

    return [[u, v] for (u, v) in best_edges]

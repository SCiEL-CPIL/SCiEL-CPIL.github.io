import time
import random
import sys
import heapq

def construct():
    start_time = time.time()
    time_limit = 135.0  # safety margin vs 150s kill

    d = 7
    n_vertices = 1 << d
    n_edges = d * (1 << (d - 1))  # 448

    # ---- edges ----
    edges = []
    edge_index = {}
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_index[(u, v)] = idx
                edges.append((u, v))
                idx += 1

    # ---- faces (squares) ----
    square_set = set()
    for u in range(n_vertices):
        for i in range(d):
            for j in range(i + 1, d):
                v1 = u ^ (1 << i)
                v2 = u ^ (1 << j)
                v3 = u ^ (1 << i) ^ (1 << j)
                e1 = (min(u, v1), max(u, v1))
                e2 = (min(v1, v3), max(v1, v3))
                e3 = (min(v3, v2), max(v3, v2))
                e4 = (min(v2, u), max(v2, u))
                idxs = []
                for e in (e1, e2, e3, e4):
                    idxs.append(edge_index[e])
                square_set.add(frozenset(idxs))

    faces = [list(s) for s in square_set]
    n_faces = len(faces)

    # ---- incidence ----
    edge_faces = [[] for _ in range(n_edges)]
    for f in range(n_faces):
        for e in faces[f]:
            edge_faces[e].append(f)

    def time_exceeded():
        return (time.time() - start_time) > time_limit

    # ---------- Greedy completion ----------
    def greedy_complete(in_set):
        """Fill in every edge that keeps the set C4-free. Returns size."""
        # count per face of selected edges
        fcnt = [0] * n_faces
        for e in range(n_edges):
            if in_set[e]:
                for f in edge_faces[e]:
                    fcnt[f] += 1
        size = sum(1 for e in range(n_edges) if in_set[e])
        order = list(range(n_edges))
        random.shuffle(order)
        for e in order:
            if in_set[e]:
                continue
            ok = True
            for f in edge_faces[e]:
                if fcnt[f] == 3:
                    ok = False
                    break
            if ok:
                in_set[e] = True
                size += 1
                for f in edge_faces[e]:
                    fcnt[f] += 1
        return size

    # ---------- Local search ----------
    def local_search(in_set, budget):
        # fcnt: per-face selected count; ffree: per-face free (not selected) count
        fcnt = [0] * n_faces
        ffree = [0] * n_faces
        for e in range(n_edges):
            if in_set[e]:
                for f in edge_faces[e]:
                    fcnt[f] += 1
            else:
                for f in edge_faces[e]:
                    ffree[f] += 1
        size = sum(1 for e in range(n_edges) if in_set[e])
        best_size = size
        best_state = in_set[:]

        def drop(e):
            nonlocal size
            in_set[e] = False
            size -= 1
            for f in edge_faces[e]:
                fcnt[f] -= 1
                ffree[f] += 1

        def add(e):
            nonlocal size
            in_set[e] = True
            size += 1
            for f in edge_faces[e]:
                fcnt[f] += 1
                ffree[f] -= 1

        def try_add(e):
            # returns True if e can be added
            for f in edge_faces[e]:
                if fcnt[f] == 3:
                    return False
            return True

        def fill():
            nonlocal size
            order = list(range(n_edges))
            random.shuffle(order)
            for e in order:
                if not in_set[e] and try_add(e):
                    add(e)

        # phase 1: fill
        fill()
        if size > best_size:
            best_size = size
            best_state = in_set[:]

        # phase 2: drop-one / drop-one-add-one
        improved = True
        while improved and not time_exceeded():
            improved = False
            order = list(range(n_edges))
            random.shuffle(order)
            for e in order:
                if time_exceeded():
                    break
                if not in_set[e]:
                    continue
                # save
                saved_in = in_set[e]
                saved_size = size
                saved_fc = fcnt[:]
                saved_ff = ffree[:]
                drop(e)
                # add all that become possible
                fill()
                if size > best_size:
                    best_size = size
                    best_state = in_set[:]
                    improved = True
                else:
                    # restore
                    in_set[e] = saved_in
                    size = saved_size
                    fcnt[:] = saved_fc
                    ffree[:] = saved_ff
        # final fill
        fill()
        if size > best_size:
            best_size = size
            best_state = in_set[:]
        return best_state

    # ---------- Loopy Belief Propagation ----------
    def run_bp(iterations=220):
        # node marginals p[e] = P(selected); message node->face m_e_f = log p/(1-p)
        # message face->node n_f_e = log P(f not full | e=1)/P(f not full | e=0)
        logit = [0.0] * n_edges
        n_f_e = [0.0] * n_edges
        for _ in range(iterations):
            if time_exceeded():
                break
            # node -> face
            m = [0.0] * n_edges
            for e in range(n_edges):
                m[e] = logit[e] + n_f_e[e]
            # face -> node
            new_n = [0.0] * n_edges
            for f in range(n_faces):
                e0, e1, e2, e3 = faces[f]
                l0, l1, l2, l3 = m[e0], m[e1], m[e2], m[e3]
                # log P(not full | e=1) = log(1 - p1 p2 p3); log P(not full | e=0) = 0
                def log1mprod(a, b, c):
                    # log(1 - prod) using logit sums; stable
                    # prod = sigmoid(a+b+c)
                    s = a + b + c
                    # log(1 - sigmoid(s)) = log(1+e^{-s}) - s ... use stable
                    if s >= 0:
                        return -s + math_log1p(math_exp(-s))
                    else:
                        return math_log1p(math_exp(s))
                lf = log1mprod(l1, l2, l3)
                new_n[e0] = lf
                lf = log1mprod(l0, l2, l3)
                new_n[e1] = lf
                lf = log1mprod(l0, l1, l3)
                new_n[e2] = lf
                lf = log1mprod(l0, l1, l2)
                new_n[e3] = lf
            n_f_e = new_n
            # update node logit
            for e in range(n_edges):
                logit[e] = n_f_e[e]
        # final marginals
        p = []
        for e in range(n_edges):
            x = logit[e]
            if x > 30:
                p.append(1.0)
            elif x < -30:
                p.append(0.0)
            else:
                p.append(1.0 / (1.0 + math_exp(-x)))
        return p

    import math
    math_exp = math.exp
    math_log1p = math.log1p

    # ---------- Symmetry-aware greedy ----------
    def symmetry_greedy(perm):
        # perm: list of d bit positions in processing order
        in_set = [False] * n_edges
        # greedy by face density: process vertices, add edges in perm order
        order = list(range(n_edges))
        # order edges by (number of faces already constrained) is complex; just shuffle with perm bias
        # Build edges grouped by dim
        by_dim = [[] for _ in range(d)]
        for e in range(n_edges):
            u, v = edges[e]
            dim = (u ^ v).bit_length() - 1
            by_dim[dim].append(e)
        for dim in perm:
            lst = by_dim[dim]
            random.shuffle(lst)
            for e in lst:
                ok = True
                for f in edge_faces[e]:
                    cnt = 0
                    for k in faces[f]:
                        if k != e and in_set[k]:
                            cnt += 1
                    if cnt == 3:
                        ok = False
                        break
                if ok:
                    in_set[e] = True
        return in_set

    best_edges_global = []
    best_size_global = 0

    def consider(in_set):
        nonlocal best_edges_global, best_size_global
        greedy_complete(in_set)
        state = local_search(in_set, time_limit)
        sz = sum(1 for e in range(n_edges) if state[e])
        if sz > best_size_global:
            best_size_global = sz
            best_edges_global = state[:]

    # ---- Seed 1: BP ----
    p = run_bp()
    # stochastic rounding
    for trial in range(3):
        if time_exceeded():
            break
        in_set = [False] * n_edges
        for e in range(n_edges):
            if random.random() < p[e]:
                in_set[e] = True
        consider(in_set)

    # ---- Seed 2: BP deterministic (p>0.5) ----
    if not time_exceeded():
        in_set = [False] * n_edges
        for e in range(n_edges):
            if p[e] > 0.5:
                in_set[e] = True
        consider(in_set)

    # ---- Seed 3: symmetry greedy ----
    for _ in range(4):
        if time_exceeded():
            break
        perm = list(range(d))
        random.shuffle(perm)
        in_set = symmetry_greedy(perm)
        consider(in_set)

    # ---- Seed 4: one full dimension ----
    for dim_start in range(d):
        if time_exceeded():
            break
        in_set = [False] * n_edges
        for u in range(n_vertices):
            v = u ^ (1 << dim_start)
            if u < v:
                in_set[edge_index[(u, v)]] = True
        consider(in_set)

    # ---- More BP restarts with temperature (stochastic) ----
    while not time_exceeded():
        if best_size_global >= 308:
            break
        p = run_bp(iterations=120)
        for trial in range(2):
            if time_exceeded():
                break
            in_set = [False] * n_edges
            for e in range(n_edges):
                if random.random() < p[e]:
                    in_set[e] = True
            consider(in_set)

    final_edges = []
    for e_idx in best_edges_global:
        if e_idx < n_edges:
            u, v = edges[e_idx]
            final_edges.append([u, v])
    return final_edges

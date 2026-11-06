import time
import random
import sys
import math

import numpy as np

_START_TIME = time.time()
_DEADLINE = 140.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

# ---------------------------------------------------------------------------
# Seed: Greenwood-Gleason colouring of K32 (valid, 32 vertices)
# ---------------------------------------------------------------------------
def _seed_32():
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)

    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                colors.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)
    return {'n': 32, 'colors': colors}

# ---------------------------------------------------------------------------
# Index / adjacency helpers
# ---------------------------------------------------------------------------
def _compute_index(n, i, j):
    return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

def _build_adj(colors, n):
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    return adj

def _count_mono_triangles(adj, n):
    count = 0
    for c in range(4):
        for i in range(n):
            neigh = adj[c][i] & ~((1 << (i + 1)) - 1)
            j_mask = neigh
            while j_mask:
                j_bit = j_mask & (-j_mask)
                j = j_bit.bit_length() - 1
                j_mask ^= j_bit
                common = adj[c][i] & adj[c][j] & ~((1 << (j + 1)) - 1)
                count += common.bit_count()
    return count

def _find_one_mono_triangle(adj, n):
    for c in range(4):
        for i in range(n):
            neigh = adj[c][i] & ~((1 << (i + 1)) - 1)
            j_mask = neigh
            while j_mask:
                j_bit = j_mask & (-j_mask)
                j = j_bit.bit_length() - 1
                j_mask ^= j_bit
                common = adj[c][i] & adj[c][j] & ~((1 << (j + 1)) - 1)
                if common:
                    k_bit = common & (-common)
                    k = k_bit.bit_length() - 1
                    return (i, j, k, c)
    return None

# ---------------------------------------------------------------------------
# Initializations
# ---------------------------------------------------------------------------
def _greedy_init(n):
    m = n * (n - 1) // 2
    colors = [0] * m
    adj = [[0] * n for _ in range(4)]
    for v in range(1, n):
        for u in range(v):
            best_c = 0
            min_cost = float('inf')
            for c in range(4):
                cost = (adj[c][u] & ((1 << u) - 1)).bit_count()
                if cost < min_cost:
                    min_cost = cost
                    best_c = c
            idx = _compute_index(n, u, v)
            colors[idx] = best_c
            adj[best_c][u] |= (1 << v)
            adj[best_c][v] |= (1 << u)
    return colors

def _random_init(n):
    m = n * (n - 1) // 2
    return [random.randint(0, 3) for _ in range(m)]

# ---------------------------------------------------------------------------
# Spectral embedding initializations (STRUCTURAL CHANGE)
# ---------------------------------------------------------------------------
def _spectral_quadrant(n, rng):
    """Embed vertices via top eigenvectors of a random symmetric matrix,
    then split into 4 quadrants using the two most-variance directions."""
    A = np.zeros((n, n), dtype=float)
    A = rng.standard_normal((n, n))
    A = (A + A.T) / 2.0
    # push eigenvalues apart / decorrelate: A + small identity shift
    A += 0.01 * np.eye(n)
    try:
        w, V = np.linalg.eigh(A)
    except Exception:
        return None
    V = V[:, ::-1]  # descending
    # pick two orthogonal directions with largest spread
    best = None
    cols = list(range(V.shape[1]))
    for a in cols:
        va = V[:, a]
        for b in cols:
            if a == b:
                continue
            vb = V[:, b]
            # orthogonality of eigenvectors already holds; measure spread
            spread = np.var(va) + np.var(vb)
            if best is None or spread > best[0]:
                best = (spread, a, b)
    _, a, b = best
    x = V[:, a]
    y = V[:, b]
    mx = x.mean(); my = y.mean()
    label = np.zeros(n, dtype=int)
    label[(x >= mx) & (y >= my)] = 0
    label[(x < mx) & (y >= my)] = 1
    label[(x >= mx) & (y < my)] = 2
    label[(x < mx) & (y < my)] = 3
    return label

def _spectral_kmeans(n, rng):
    """Embed via top eigenvectors, then k-means into 4 clusters."""
    A = np.zeros((n, n), dtype=float)
    A = rng.standard_normal((n, n))
    A = (A + A.T) / 2.0
    A += 0.01 * np.eye(n)
    try:
        w, V = np.linalg.eigh(A)
    except Exception:
        return None
    V = V[:, ::-1]
    # use top min(8,n) eigenvectors for a richer embedding
    d = min(8, n)
    X = V[:, :d]
    # k-means
    n_iter = 25
    idx = rng.choice(n, size=4, replace=False)
    centers = X[idx].copy()
    label = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        # assign
        dist = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        label = dist.argmin(axis=1)
        # update
        for k in range(4):
            mask = label == k
            if mask.any():
                centers[k] = X[mask].mean(axis=0)
            else:
                centers[k] = X[rng.integers(n)]
    return label

def _label_to_colors(n, label):
    m = n * (n - 1) // 2
    colors = [0] * m
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            colors[idx] = int(label[i])
            idx += 1
    return colors

# ---------------------------------------------------------------------------
# Spectral bisection refinement: move a vertex's region to reduce triangles
# ---------------------------------------------------------------------------
def _spectral_refine(n, label, time_budget):
    """Iteratively reassign a vertex's color to minimize monochromatic
    triangles. label is a numpy int array of length n."""
    start = time.time()
    label = label.copy()
    # build per-color adjacency bitsets from label
    def build(label):
        adj = [[0] * n for _ in range(4)]
        for i in range(n):
            ci = label[i]
            for j in range(i + 1, n):
                cj = label[j]
                if ci == cj:
                    adj[ci][i] |= (1 << j)
                    adj[ci][j] |= (1 << i)
        return adj
    adj = build(label)

    def tri_count(adj):
        c = 0
        for k in range(4):
            for i in range(n):
                neigh = adj[k][i] & ~((1 << (i + 1)) - 1)
                j_mask = neigh
                while j_mask:
                    jb = j_mask & (-j_mask)
                    j = jb.bit_length() - 1
                    j_mask ^= jb
                    common = adj[k][i] & adj[k][j] & ~((1 << (j + 1)) - 1)
                    c += common.bit_count()
        return c

    total = tri_count(adj)
    improved = True
    while improved and (time.time() - start) < time_budget:
        improved = False
        for v in range(n):
            cv = label[v]
            # cost of moving v to color c = number of monochromatic triangles
            # formed with v in color c = (adj[c][v] neighbours among each other)
            best_c = cv
            best_cost = float('inf')
            for c in range(4):
                # triangles through v using color c = edges of adj[c][v]
                neigh = adj[c][v]
                cnt = 0
                mm = neigh
                while mm:
                    jb = mm & (-mm)
                    j = jb.bit_length() - 1
                    mm ^= jb
                    cnt += (adj[c][j] & neigh).bit_count()
                cnt //= 2
                if cnt < best_cost:
                    best_cost = cnt
                    best_c = c
            if best_c != cv:
                # remove v from color cv
                adj[cv][v] &= ~((1 << n) - 1)
                for nb_bit in [b for b in range(n) if (adj[cv][v] >> b) & 1]:
                    adj[cv][nb_bit] &= ~(1 << v)
                adj[cv][v] = 0
                # add v to best_c
                adj[best_c][v] = 0
                for u in range(n):
                    if u != v:
                        adj[best_c][v] |= (1 << u)
                        adj[best_c][u] |= (1 << v)
                label[v] = best_c
                improved = True
    return label

# ---------------------------------------------------------------------------
# Local search (edge recoloring) - refinement machinery
# ---------------------------------------------------------------------------
def _local_search(n, colors, time_budget, idx_to_edge):
    m = n * (n - 1) // 2
    start = time.time()
    adj = _build_adj(colors, n)
    tri_count = _count_mono_triangles(adj, n)
    if tri_count == 0:
        return colors
    best_tri = tri_count
    best = colors[:]
    no_improve = 0
    max_no_improve = 60
    while no_improve < max_no_improve and (time.time() - start) < time_budget:
        tri = _find_one_mono_triangle(adj, n)
        if tri is None:
            return best
        i, j, k, c = tri
        edges = [(i, j), (i, k), (j, k)]
        best_delta = float('inf')
        best_edge = None
        best_new_c = -1
        for u, v in edges:
            for new_c in range(4):
                if new_c == c:
                    continue
                removed = (adj[c][u] & adj[c][v]).bit_count()
                added = (adj[new_c][u] & adj[new_c][v]).bit_count()
                delta = added - removed
                if delta < best_delta:
                    best_delta = delta
                    best_edge = (u, v)
                    best_new_c = new_c
        if best_edge is None or best_delta >= 0:
            no_improve += 1
            if no_improve > 15:
                ri = random.randint(0, m - 1)
                u, v = idx_to_edge[ri]
                old_c = colors[ri]
                new_c = random.randint(0, 3)
                if new_c != old_c:
                    removed = (adj[old_c][u] & adj[old_c][v]).bit_count()
                    added = (adj[new_c][u] & adj[new_c][v]).bit_count()
                    tri_count += added - removed
                    colors[ri] = new_c
                    adj[old_c][u] &= ~(1 << v)
                    adj[old_c][v] &= ~(1 << u)
                    adj[new_c][u] |= (1 << v)
                    adj[new_c][v] |= (1 << u)
                    if tri_count < best_tri:
                        best_tri = tri_count
                        best = colors[:]
            continue
        u, v = best_edge
        idx = _compute_index(n, u, v)
        removed = (adj[c][u] & adj[c][v]).bit_count()
        added = (adj[best_new_c][u] & adj[best_new_c][v]).bit_count()
        tri_count += added - removed
        colors[idx] = best_new_c
        adj[c][u] &= ~(1 << v)
        adj[c][v] &= ~(1 << u)
        adj[best_new_c][u] |= (1 << v)
        adj[best_new_c][v] |= (1 << u)
        if tri_count < best_tri:
            best_tri = tri_count
            best = colors[:]
            no_improve = 0
        else:
            no_improve += 1
    return best if best_tri == 0 else None

# ---------------------------------------------------------------------------
# Simulated annealing (edge recoloring)
# ---------------------------------------------------------------------------
def _sa_search(n, colors, time_budget, idx_to_edge):
    m = n * (n - 1) // 2
    start = time.time()
    adj = _build_adj(colors, n)
    tri_count = _count_mono_triangles(adj, n)
    if tri_count == 0:
        return colors
    best_tri = tri_count
    best = colors[:]
    T = 5.0
    T_min = 0.01
    cooling = 0.9995
    accepted = 0
    total_steps = 0
    while (time.time() - start) < time_budget:
        ri = random.randint(0, m - 1)
        old_c = colors[ri]
        new_c = random.randint(0, 3)
        if new_c == old_c:
            continue
        u, v = idx_to_edge[ri]
        removed = (adj[old_c][u] & adj[old_c][v]).bit_count()
        added = (adj[new_c][u] & adj[new_c][v]).bit_count()
        delta = added - removed
        total_steps += 1
        if delta <= 0:
            colors[ri] = new_c
            adj[old_c][u] &= ~(1 << v)
            adj[old_c][v] &= ~(1 << u)
            adj[new_c][u] |= (1 << v)
            adj[new_c][v] |= (1 << u)
            tri_count += delta
            accepted += 1
            if tri_count < best_tri:
                best_tri = tri_count
                best = colors[:]
                if tri_count == 0:
                    return colors
        else:
            if T > T_min:
                if random.random() < math.exp(-delta / T):
                    colors[ri] = new_c
                    adj[old_c][u] &= ~(1 << v)
                    adj[old_c][v] &= ~(1 << u)
                    adj[new_c][u] |= (1 << v)
                    adj[new_c][v] |= (1 << u)
                    tri_count += delta
                    accepted += 1
        T *= cooling
        if total_steps > 1000 and accepted < total_steps * 0.01:
            break
    return best if best_tri == 0 else None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def construct():
    try:
        best = _seed_32()
        best_n = 32

        rng = np.random.default_rng(12345)
        random.seed(12345)

        targets = [50, 48, 46, 44, 42, 40]

        for target_n in targets:
            if target_n <= best_n:
                continue
            remaining = _DEADLINE - (time.time() - _START_TIME)
            if remaining < 4:
                break

            m = target_n * (target_n - 1) // 2
            idx_to_edge = [None] * m
            idx = 0
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    idx_to_edge[idx] = (i, j)
                    idx += 1

            # budget split
            spec_budget = min(remaining - 3, 25)
            ls_budget = min(remaining - 2, 30)
            sa_budget = min(remaining - 1, 25)

            # ---- Spectral family: generate candidate initializations ----
            candidates = []
            if spec_budget > 1:
                t0 = time.time()
                # several spectral embeddings
                for _ in range(6):
                    if time.time() - t0 > spec_budget * 0.5:
                        break
                    lab = _spectral_quadrant(target_n, rng)
                    if lab is not None:
                        candidates.append(lab)
                for _ in range(4):
                    if time.time() - t0 > spec_budget * 0.8:
                        break
                    lab = _spectral_kmeans(target_n, rng)
                    if lab is not None:
                        candidates.append(lab)
                # also include greedy and random as candidates
                candidates.append(np.array(_greedy_init(target_n)))
                candidates.append(np.array(_random_init(target_n)))

                # refine each candidate via spectral bisection, keep best
                best_cand = None
                best_cand_tri = float('inf')
                per = max(0.5, spec_budget / max(1, len(candidates)))
                for lab in candidates:
                    lab = _spectral_refine(target_n, lab, per)
                    cols = _label_to_colors(target_n, lab)
                    adj = _build_adj(cols, target_n)
                    tc = _count_mono_triangles(adj, target_n)
                    if tc < best_cand_tri:
                        best_cand_tri = tc
                        best_cand = cols
                    if tc == 0:
                        return {'n': target_n, 'colors': cols}

                if best_cand is not None:
                    # local search from the best spectral candidate
                    if ls_budget > 1:
                        res = _local_search(target_n, best_cand[:], min(ls_budget, 15), idx_to_edge)
                        if res is not None:
                            return {'n': target_n, 'colors': res}
                    if sa_budget > 1:
                        res = _sa_search(target_n, best_cand[:], min(sa_budget, 12), idx_to_edge)
                        if res is not None:
                            return {'n': target_n, 'colors': res}

            # ---- Fallback: pure local search / SA from greedy ----
            remaining = _DEADLINE - (time.time() - _START_TIME)
            if remaining > 3:
                gc = _greedy_init(target_n)
                res = _local_search(target_n, gc[:], min(remaining - 2, 20), idx_to_edge)
                if res is not None:
                    return {'n': target_n, 'colors': res}
                remaining = _DEADLINE - (time.time() - _START_TIME)
                if remaining > 3:
                    res = _sa_search(target_n, gc[:], min(remaining - 1, 15), idx_to_edge)
                    if res is not None:
                        return {'n': target_n, 'colors': res}

        return best
    except Exception:
        try:
            return _seed_32()
        except Exception:
            return {'n': 1, 'colors': []}

if __name__ == "__main__":
    res = construct()
    print(res['n'])

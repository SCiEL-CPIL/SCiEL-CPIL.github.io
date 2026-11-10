# EVOLVE-BLOCK-START
import numpy as np
import time

_T0 = time.time()
_DEADLINE = _T0 + 250.0  # safety margin under 300s wall-clock


def _segments_cross(ax, ay, bx, by, cx, cy, dx, dy):
    """Vectorized proper-crossing test for segments (a,b) vs (c,d)."""
    def orient(px, py, qx, qy, rx, ry):
        val = (qx - px) * (ry - py) - (qy - py) * (rx - px)
        return np.sign(val)
    o1 = orient(ax, ay, bx, by, cx, cy)
    o2 = orient(ax, ay, bx, by, dx, dy)
    o3 = orient(cx, cy, dx, dy, ax, ay)
    o4 = orient(cx, cy, dx, dy, bx, by)
    return (o1 != o2) & (o3 != o4)


def _tour_length(tour, D):
    n = len(tour)
    return float(D[tour, np.roll(tour, -1)].sum())


def _two_opt(tour, D, max_passes=80):
    n = len(tour)
    tour = list(tour)
    best = _tour_length(tour, D)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            a, b = tour[i], tour[i + 1]
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:
                    continue
                c, d = tour[j], tour[(j + 1) % n]
                delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                if delta < -1e-12:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    best += delta
                    improved = True
    return tour, best


def _or_opt(d, order, max_passes=20):
    """
    Improve a tour using Or-opt (moving 1 node).
    """
    n = len(order)
    if n < 4:
        return list(order), float(sum(d[order[i], order[(i + 1) % n]] for i in range(n)))
    
    best = list(order)
    best_len = _tour_length(best, d)
    
    for _ in range(max_passes):
        improved = False
        for i in range(n):
            u = best[i]
            prev_node = best[(i - 1) % n]
            next_node = best[(i + 1) % n]
            
            # Cost to remove u from position i
            # Remove edges (prev, u) and (u, next), add (prev, next)
            len_without_u = best_len - d[prev_node, u] - d[u, next_node] + d[prev_node, next_node]
            
            # Create a copy without u
            temp_order = best[:]
            temp_order.pop(i)
            m = len(temp_order)
            
            best_in_local = None
            best_len_local = np.inf
            
            for k in range(m):
                prev_k = temp_order[(k - 1) % m]
                next_k = temp_order[k % m]
                
                delta_insert = d[prev_k, u] + d[u, next_k] - d[prev_k, next_k]
                new_len = len_without_u + delta_insert
                
                if new_len < best_len_local:
                    best_len_local = new_len
                    best_in_local = temp_order[:]
                    best_in_local.insert(k, u)
                    
            if best_in_local is not None and best_len_local < best_len - 1e-10:
                best = best_in_local
                best_len = best_len_local
                improved = True
        if not improved:
            break
            
    return best, best_len


def _nearest_neighbor(D, start):
    n = D.shape[0]
    unvisited = set(range(n))
    unvisited.remove(start)
    path = [start]
    cur = start
    while unvisited:
        # nearest neighbor among unvisited
        cands = list(unvisited)
        dists = D[cur, cands]
        nxt = cands[int(np.argmin(dists))]
        path.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return path


def _compute_reference_tour(D, n):
    """Multi-start nearest-neighbor + 2-opt + Or-opt to get a good reference tour."""
    best_tour = None
    best_len = np.inf
    # multiple starts
    starts = list(range(n))
    for s in starts:
        if time.time() > _DEADLINE:
            break
        path = _nearest_neighbor(D, s)
        tour, ln = _two_opt(path, D)
        tour, ln = _or_opt(D, tour)
        if ln < best_len:
            best_len = ln
            best_tour = tour
    return best_tour, best_len


def heuristics(distance_matrix):
    D = np.asarray(distance_matrix, dtype=np.float64)
    n = D.shape[0]

    # 1) base inverse-distance heuristic
    base = 1.0 / (D + 1e-9)
    base = np.where(np.isfinite(base) & (base > 0), base, 1e-9)

    # 2) build reference tour (NN multi-start + 2-opt + Or-opt) with deadline guard
    try:
        ref_tour, ref_len = _compute_reference_tour(D, n)
    except Exception:
        ref_tour = list(range(n))
        ref_len = np.inf

    # points: reconstruct coordinates from distance matrix via MDS
    coords = None
    try:
        d2 = D ** 2
        r = np.mean(d2, axis=1)
        c = np.mean(d2, axis=0)
        z = np.mean(d2)
        B = -0.5 * (d2 - r[:, None] - c[None, :] + z)
        w, V = np.linalg.eigh(B)
        idx = np.argsort(w)[::-1]
        w = w[idx]
        V = V[:, idx]
        # take top 2 dims
        w2 = np.sqrt(np.clip(w[:2], 0, None))
        coords = V[:, :2] * w2[None, :]
    except Exception:
        coords = None

    H = base.copy()

    if coords is not None and ref_tour is not None and len(ref_tour) == n:
        # topological alignment factor
        # reward edges present in reference tour
        align = np.ones((n, n))
        for k in range(n):
            u = ref_tour[k]
            v = ref_tour[(k + 1) % n]
            align[u, v] *= 3.0
            align[v, u] *= 3.0

        # crossing penalty: penalize edges crossing any reference-tour segment
        # Build reference segment list
        segs = []
        for k in range(n):
            u = ref_tour[k]
            v = ref_tour[(k + 1) % n]
            segs.append((u, v))

        # For each candidate edge (i,j), test crossing against each ref segment
        # Vectorize over edges vs each segment
        cross_count = np.zeros((n, n))
        for (u, v) in segs:
            ax, ay = coords[u]
            bx, by = coords[v]
            # all candidate edges (i,j)
            ii, jj = np.indices((n, n))
            # skip the segment's own endpoints and diagonal
            mask = (ii != u) & (jj != u) & (ii != v) & (jj != v) & (ii != jj)
            if not np.any(mask):
                continue
            cx = coords[ii[mask], 0]
            cy = coords[ii[mask], 1]
            dx = coords[jj[mask], 0]
            dy = coords[jj[mask], 1]
            cross = _segments_cross(ax, ay, bx, by, cx, cy, dx, dy)
            # scatter back
            idx_i = ii[mask][cross]
            idx_j = jj[mask][cross]
            cross_count[idx_i, idx_j] += 1
            cross_count[idx_j, idx_i] += 1

        # heavy penalty for crossing edges (exponential decay)
        cross_penalty = np.exp(-1.0 * cross_count)
        H = H * align * cross_penalty

    # ensure positive & finite
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    return H
# EVOLVE-BLOCK-END

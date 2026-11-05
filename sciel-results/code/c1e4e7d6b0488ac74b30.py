"""Combined: incremental triangle tracking + NumPy scoring + multi-start local search."""
import time
import random
import math
import numpy as np

N = 16
TARGET_EDGES = 65
DEADLINE = 145.0


def _compute_score_np(adj_mat, deg_vec):
    """Compute max triangle degree sum using NumPy vectorization."""
    sums = deg_vec[:, np.newaxis, np.newaxis] + deg_vec[np.newaxis, :, np.newaxis] + deg_vec[np.newaxis, np.newaxis, :]
    tri_mask = adj_mat[:, :, np.newaxis] * adj_mat[np.newaxis, :, :] * adj_mat[:, np.newaxis, :]
    scored_sums = sums * tri_mask
    mx = int(np.max(scored_sums))
    return mx if mx > 0 else 0


def _build_triangles(adj):
    """Build list of triangles as sorted tuples (i,j,k) with i<j<k."""
    tris = []
    for i in range(N):
        for j in range(i + 1, N):
            if not adj[i][j]:
                continue
            for k in range(j + 1, N):
                if adj[i][k] and adj[j][k]:
                    tris.append((i, j, k))
    return tris


def _max_tri_sum(tris, deg):
    """Max degree sum over triangles."""
    if not tris:
        return 0
    best = 0
    for (i, j, k) in tris:
        s = deg[i] + deg[j] + deg[k]
        if s > best:
            best = s
    return best


def _update_triangles(tris, adj, e_rm, e_add):
    """
    Incrementally update triangle list after removing e_rm=(u,v) and adding e_add=(x,y).
    Returns new triangle list.
    """
    u, v = e_rm
    x, y = e_add
    # Ensure consistent ordering
    if u > v:
        u, v = v, u
    if x > y:
        x, y = y, x

    # Triangles to remove: those containing edge (u,v)
    # A triangle contains (u,v) iff it has both u and v as vertices
    new_tris = []
    for t in tris:
        tset = set(t)
        if u in tset and v in tset:
            continue  # destroyed
        new_tris.append(t)

    # Triangles to add: those containing edge (x,y)
    # Third vertex w must be adjacent to both x and y
    for w in range(N):
        if w == x or w == y:
            continue
        if adj[x][w] and adj[y][w]:
            tri = tuple(sorted((x, y, w)))
            # Check it's not already in new_tris (shouldn't be since (x,y) was non-edge)
            new_tris.append(tri)

    return new_tris


def _get_worst_triangle(tris, deg):
    """Find the triangle with maximum degree sum."""
    if not tris:
        return None
    best_tri = None
    best_s = 0
    for (i, j, k) in tris:
        s = deg[i] + deg[j] + deg[k]
        if s > best_s:
            best_s = s
            best_tri = (i, j, k)
    return best_tri


def _local_search(adj, deg, tris, cur_score, t0, rng, iters_budget):
    """Local search with incremental triangle updates."""
    edges = [(i, j) for i in range(N) for j in range(i + 1, N) if adj[i][j]]
    nonedges = [(i, j) for i in range(N) for j in range(i + 1, N) if not adj[i][j]]

    best_score = cur_score
    best_adj = [row[:] for row in adj]
    best_deg = deg[:]

    start_temp = 3.0
    end_temp = 0.05
    n = max(1, iters_budget)
    cool = (end_temp / start_temp) ** (1.0 / n)
    temp = start_temp

    iters = 0
    while iters < iters_budget:
        if (iters % 200 == 0) and (time.time() - t0 > DEADLINE):
            break

        if not edges or not nonedges:
            break

        # Biased move selection: 50% chance to target worst triangle
        worst_tri = _get_worst_triangle(tris, deg)

        if worst_tri is not None and rng.random() < 0.5:
            i, j, k = worst_tri
            tri_edges = [(i, j), (i, k), (j, k)]
            er = tri_edges[rng.randrange(3)]
            if er not in edges:
                er = edges[rng.randrange(len(edges))]
            ea = nonedges[rng.randrange(len(nonedges))]
        else:
            er = edges[rng.randrange(len(edges))]
            ea = nonedges[rng.randrange(len(nonedges))]

        if er == ea:
            iters += 1
            continue

        u, v = er
        x, y = ea

        # Apply swap
        adj[u][v] = adj[v][u] = False
        deg[u] -= 1
        deg[v] -= 1
        adj[x][y] = adj[y][x] = True
        deg[x] += 1
        deg[y] += 1

        # Update triangles incrementally
        new_tris = _update_triangles(tris, adj, er, ea)
        new_score = _max_tri_sum(new_tris, deg)

        # Accept/reject
        accept = False
        if new_score < cur_score:
            accept = True
        elif new_score == cur_score:
            accept = rng.random() < 0.3
        else:
            d = new_score - cur_score
            if temp < 1e-9:
                accept = False
            else:
                accept = math.exp(-d / temp) > rng.random()

        if accept:
            cur_score = new_score
            tris = new_tris
            if new_score < best_score:
                best_score = new_score
                best_adj = [row[:] for row in adj]
                best_deg = deg[:]
            # Update edge lists
            edges.remove(er)
            nonedges.remove(ea)
            nonedges.append(er)
            edges.append(ea)
        else:
            # Undo
            adj[x][y] = adj[y][x] = False
            deg[x] -= 1
            deg[y] -= 1
            adj[u][v] = adj[v][u] = True
            deg[u] += 1
            deg[v] += 1

        iters += 1
        temp *= cool

    return best_adj, best_deg, best_score


def _make_bipartite_start(rng, offset=0):
    """Create a K_{8,8}+1 starting graph with a random partition."""
    adj = [[False] * N for _ in range(N)]
    # Random bipartition
    perm = list(range(N))
    rng.shuffle(perm)
    part_a = perm[:8]
    part_b = perm[8:]
    for i in part_a:
        for j in part_b:
            adj[i][j] = adj[j][i] = True

    # Add one internal edge in part_a
    a = part_a[0]
    b = part_a[1]
    adj[a][b] = adj[b][a] = True

    deg = [0] * N
    for i in range(N):
        for j in range(N):
            if adj[i][j]:
                deg[i] += 1

    tris = _build_triangles(adj)
    score = _max_tri_sum(tris, deg)
    return adj, deg, tris, score


def _make_near_bipartite_start(rng, n_internal=1):
    """Create a near-bipartite graph with a few internal edges."""
    adj = [[False] * N for _ in range(N)]
    perm = list(range(N))
    rng.shuffle(perm)
    part_a = perm[:8]
    part_b = perm[8:]
    for i in part_a:
        for j in part_b:
            adj[i][j] = adj[j][i] = True

    # Add n_internal random internal edges in part_a
    internal_pairs = [(part_a[i], part_a[j]) for i in range(8) for j in range(i + 1, 8)]
    rng.shuffle(internal_pairs)
    for a, b in internal_pairs[:n_internal]:
        adj[a][b] = adj[b][a] = True

    # If we added more than 1 internal edge, we need to remove some bipartite edges to keep 65
    # Actually, let's just ensure exactly 65 edges
    # Current edges: 64 bipartite + n_internal
    # We need exactly 65, so if n_internal > 1, remove (n_internal - 1) bipartite edges
    if n_internal > 1:
        bip_edges = [(i, j) for i in part_a for j in part_b if adj[i][j]]
        rng.shuffle(bip_edges)
        for i, j in bip_edges[:n_internal - 1]:
            adj[i][j] = adj[j][i] = False

    deg = [0] * N
    for i in range(N):
        for j in range(N):
            if adj[i][j]:
                deg[i] += 1

    tris = _build_triangles(adj)
    score = _max_tri_sum(tris, deg)
    return adj, deg, tris, score


def construct():
    t0 = time.time()
    rng = random.Random(12345)

    best_adj = None
    best_deg = None
    best_score = float('inf')

    # Try multiple starting configurations
    starts = []

    # Start 1: standard K_{8,8} + 1 internal edge
    adj1, deg1, tris1, score1 = _make_bipartite_start(rng, offset=0)
    starts.append((adj1, deg1, tris1, score1))

    # Start 2: different random partition
    adj2, deg2, tris2, score2 = _make_bipartite_start(rng, offset=1)
    starts.append((adj2, deg2, tris2, score2))

    # Start 3: near-bipartite with 2 internal edges (remove 1 bipartite)
    adj3, deg3, tris3, score3 = _make_near_bipartite_start(rng, n_internal=2)
    starts.append((adj3, deg3, tris3, score3))

    # Start 4: another random partition
    adj4, deg4, tris4, score4 = _make_bipartite_start(rng, offset=2)
    starts.append((adj4, deg4, tris4, score4))

    # Start 5: yet another
    adj5, deg5, tris5, score5 = _make_bipartite_start(rng, offset=3)
    starts.append((adj5, deg5, tris5, score5))

    # Pick the best starting point and run local search
    # Allocate time proportionally, but prioritize the best start
    starts.sort(key=lambda x: x[3])

    # Run local search on the best 2-3 starts
    elapsed = time.time() - t0
    remaining = DEADLINE - elapsed - 3.0  # safety margin

    if remaining > 5.0:
        # Allocate 60% to best start, 25% to second, 15% to third
        for idx, (adj, deg, tris, score) in enumerate(starts[:3]):
            if time.time() - t0 > DEADLINE - 2.0:
                break
            if idx == 0:
                budget = int(remaining * 0.55)
            elif idx == 1:
                budget = int(remaining * 0.25)
            else:
                budget = int(remaining * 0.15)

            budget = max(100, budget)
            b_adj, b_deg, b_score = _local_search(adj, deg, tris, score, t0, rng, budget)

            if b_score < best_score:
                best_score = b_score
                best_adj = b_adj
                best_deg = b_deg

    # If we didn't find anything (shouldn't happen), use the best start
    if best_adj is None:
        best_adj, best_deg, _, best_score = starts[0]

    # Final verification and edge extraction
    edges = [[i, j] for i in range(N) for j in range(i + 1, N) if best_adj[i][j]]

    # Safety: ensure exactly 65 edges
    if len(edges) != TARGET_EDGES:
        # Fallback: use the first start
        adj_f, deg_f, _, _ = starts[0]
        edges = [[i, j] for i in range(N) for j in range(i + 1, N) if adj_f[i][j]]

    return {'n': N, 'edges': edges}


if __name__ == "__main__":
    res = construct()
    print("edges:", len(res["edges"]))

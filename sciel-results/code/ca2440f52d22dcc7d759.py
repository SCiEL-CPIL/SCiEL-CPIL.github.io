"""
Combined: incremental triangle tracking + multi-start local search +
MILP-style targeted repair for sub-24 triangle degree sums.

Strategy:
  1. Generate diverse near-bipartite starting graphs (K_{8,8}+1 variants).
  2. Run simulated annealing with strict edge-swaps (keeps 65 edges).
  3. If score is still >= 24, attempt a targeted repair:
     - Identify all triangles with degree sum > 23.
     - For each such triangle, try removing one of its edges and adding
       a carefully chosen non-edge that minimizes new high-sum triangles.
     - This is a greedy "conflict elimination" pass inspired by the
       MILP/branch-and-bound idea: directly enforce the score constraint.
  4. Final polish: short SA run with very low temperature.
"""
import time
import random
import math
import numpy as np

N = 16
TARGET_EDGES = 65
DEADLINE = 145.0  # safety margin before 150s kill


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
    if u > v:
        u, v = v, u
    if x > y:
        x, y = y, x

    new_tris = []
    for t in tris:
        tset = set(t)
        if u in tset and v in tset:
            continue
        new_tris.append(t)

    for w in range(N):
        if w == x or w == y:
            continue
        if adj[x][w] and adj[y][w]:
            tri = tuple(sorted((x, y, w)))
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
    """Local search with incremental triangle updates and biased move selection."""
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
            temp *= cool
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
            edges.remove(er)
            nonedges.remove(ea)
            nonedges.append(er)
            edges.append(ea)
        else:
            adj[x][y] = adj[y][x] = False
            deg[x] -= 1
            deg[y] -= 1
            adj[u][v] = adj[v][u] = True
            deg[u] += 1
            deg[v] += 1

        iters += 1
        temp *= cool

    return best_adj, best_deg, best_score


def _targeted_repair(adj, deg, tris, cur_score, t0, rng, max_iters=5000):
    """
    Targeted repair: directly try to eliminate triangles with degree sum > 23.
    Inspired by MILP conflict-elimination: for each "bad" triangle, try removing
    one of its edges and adding a non-edge that minimizes the resulting max sum.
    """
    if cur_score <= 23:
        return adj, deg, tris, cur_score

    best_adj = [row[:] for row in adj]
    best_deg = deg[:]
    best_tris = tris[:]
    best_score = cur_score

    edges = [(i, j) for i in range(N) for j in range(i + 1, N) if adj[i][j]]
    nonedges = [(i, j) for i in range(N) for j in range(i + 1, N) if not adj[i][j]]

    iters = 0
    while iters < max_iters:
        if time.time() - t0 > DEADLINE - 2.0:
            break
        if not edges or not nonedges:
            break

        # Find all triangles with sum > 23
        bad_tris = [(i, j, k) for (i, j, k) in tris
                    if deg[i] + deg[j] + deg[k] > 23]

        if not bad_tris:
            break

        # Pick a random bad triangle
        bt = bad_tris[rng.randrange(len(bad_tris))]
        i, j, k = bt
        tri_edges = [(i, j), (i, k), (j, k)]

        # Try removing each edge of the bad triangle
        best_swap = None
        best_new_score = cur_score

        for er in tri_edges:
            if er not in edges:
                continue
            u, v = er
            # Try a few candidate non-edges
            candidates = nonedges[:]
            rng.shuffle(candidates)
            for ea in candidates[:20]:
                x, y = ea
                # Quick estimate: what would the new degrees be?
                # Remove (u,v): deg[u]-1, deg[v]-1
                # Add (x,y): deg[x]+1, deg[y]+1
                new_deg = deg[:]
                new_deg[u] -= 1
                new_deg[v] -= 1
                new_deg[x] += 1
                new_deg[y] += 1

                # Apply swap to get new triangles
                adj[u][v] = adj[v][u] = False
                adj[x][y] = adj[y][x] = True
                new_tris = _update_triangles(tris, adj, er, ea)
                new_score = _max_tri_sum(new_tris, new_deg)

                # Undo for next candidate
                adj[x][y] = adj[y][x] = False
                adj[u][v] = adj[v][u] = True

                if new_score < best_new_score:
                    best_new_score = new_score
                    best_swap = (er, ea, new_tris, new_score)

        if best_swap is None:
            iters += 1
            continue

        er, ea, new_tris, new_score = best_swap
        u, v = er
        x, y = ea

        # Apply the best swap
        adj[u][v] = adj[v][u] = False
        deg[u] -= 1
        deg[v] -= 1
        adj[x][y] = adj[y][x] = True
        deg[x] += 1
        deg[y] += 1
        tris = new_tris
        cur_score = new_score

        if new_score < best_score:
            best_score = new_score
            best_adj = [row[:] for row in adj]
            best_deg = deg[:]
            best_tris = tris[:]

        # Update edge lists
        edges.remove(er)
        nonedges.remove(ea)
        nonedges.append(er)
        edges.append(ea)

        iters += 1

    return best_adj, best_deg, best_tris, best_score


def _make_bipartite_start(rng, offset=0):
    """Create a K_{8,8}+1 starting graph with a random partition."""
    adj = [[False] * N for _ in range(N)]
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

    internal_pairs = [(part_a[i], part_a[j]) for i in range(8) for j in range(i + 1, 8)]
    rng.shuffle(internal_pairs)
    for a, b in internal_pairs[:n_internal]:
        adj[a][b] = adj[b][a] = True

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


def _make_circulant_start(rng):
    """Create a circulant-like starting graph with 65 edges."""
    adj = [[False] * N for _ in range(N)]
    # Use a set of offsets to create a near-regular graph
    offsets = [1, 2, 3, 4, 5, 6, 7]  # 7 offsets * 16 / 2 = 56 edges
    for i in range(N):
        for d in offsets:
            j = (i + d) % N
            if i < j:
                adj[i][j] = adj[j][i] = True

    # Count edges
    ec = sum(1 for i in range(N) for j in range(i + 1, N) if adj[i][j])
    # Need 65 - ec more edges
    if ec < TARGET_EDGES:
        missing = TARGET_EDGES - ec
        nonedges = [(i, j) for i in range(N) for j in range(i + 1, N) if not adj[i][j]]
        rng.shuffle(nonedges)
        for i, j in nonedges[:missing]:
            adj[i][j] = adj[j][i] = True
    elif ec > TARGET_EDGES:
        extra = ec - TARGET_EDGES
        edges = [(i, j) for i in range(N) for j in range(i + 1, N) if adj[i][j]]
        rng.shuffle(edges)
        for i, j in edges[:extra]:
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

    # Generate diverse starting configurations
    starts = []

    # Start 1: standard K_{8,8} + 1 internal edge
    adj1, deg1, tris1, score1 = _make_bipartite_start(rng, offset=0)
    starts.append((adj1, deg1, tris1, score1))

    # Start 2: different random partition
    adj2, deg2, tris2, score2 = _make_bipartite_start(rng, offset=1)
    starts.append((adj2, deg2, tris2, score2))

    # Start 3: near-bipartite with 2 internal edges
    adj3, deg3, tris3, score3 = _make_near_bipartite_start(rng, n_internal=2)
    starts.append((adj3, deg3, tris3, score3))

    # Start 4: another random partition
    adj4, deg4, tris4, score4 = _make_bipartite_start(rng, offset=2)
    starts.append((adj4, deg4, tris4, score4))

    # Start 5: circulant-like
    adj5, deg5, tris5, score5 = _make_circulant_start(rng)
    starts.append((adj5, deg5, tris5, score5))

    # Start 6: near-bipartite with 3 internal edges
    adj6, deg6, tris6, score6 = _make_near_bipartite_start(rng, n_internal=3)
    starts.append((adj6, deg6, tris6, score6))

    # Sort by initial score
    starts.sort(key=lambda x: x[3])

    # Run local search on top starts
    elapsed = time.time() - t0
    remaining = DEADLINE - elapsed - 5.0

    if remaining > 5.0:
        # Allocate time to top 3-4 starts
        top_k = min(4, len(starts))
        # Weighted allocation: best gets more time
        weights = [0.4, 0.25, 0.2, 0.15][:top_k]
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        for idx, (adj, deg, tris, score) in enumerate(starts[:top_k]):
            if time.time() - t0 > DEADLINE - 3.0:
                break
            budget = int(remaining * weights[idx])
            budget = max(200, budget)

            b_adj, b_deg, b_score = _local_search(adj, deg, tris, score, t0, rng, budget)

            # Try targeted repair if score is still high
            if b_score > 23 and time.time() - t0 < DEADLINE - 5.0:
                r_adj, r_deg, r_tris, r_score = _targeted_repair(
                    b_adj, b_deg, _build_triangles(b_adj), b_score, t0, rng, max_iters=3000
                )
                if r_score < b_score:
                    b_adj, b_deg, b_score = r_adj, r_deg, r_score

            if b_score < best_score:
                best_score = b_score
                best_adj = b_adj
                best_deg = b_deg

    # If we didn't find anything, use the best start
    if best_adj is None:
        best_adj, best_deg, _, best_score = starts[0]

    # Final polish: short SA with very low temperature if time remains
    if time.time() - t0 < DEADLINE - 3.0 and best_score > 23:
        polish_budget = int((DEADLINE - 3.0 - (time.time() - t0)) * 50)
        polish_budget = max(100, min(polish_budget, 5000))
        p_adj, p_deg, p_score = _local_search(
            best_adj, best_deg, _build_triangles(best_adj), best_score, t0, rng, polish_budget
        )
        if p_score < best_score:
            best_score = p_score
            best_adj = p_adj
            best_deg = p_deg

    # Final verification and edge extraction
    edges = [[i, j] for i in range(N) for j in range(i + 1, N) if best_adj[i][j]]

    # Safety: ensure exactly 65 edges
    if len(edges) != TARGET_EDGES:
        adj_f, deg_f, _, _ = starts[0]
        edges = [[i, j] for i in range(N) for j in range(i + 1, N) if adj_f[i][j]]

    return {'n': N, 'edges': edges}


if __name__ == "__main__":
    res = construct()
    print("edges:", len(res["edges"]))

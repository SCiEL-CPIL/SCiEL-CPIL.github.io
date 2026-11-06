"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    _t0 = time.time()
    _DEADLINE = 138.0  # safety margin before 150s kill

    n = 20
    C = 20  # number of colors

    # Build complete graph K20
    edges = []
    edge_idx = {}
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            edge_idx[(i, j)] = len(edges) - 1
            edge_idx[(j, i)] = len(edges) - 1
    m = len(edges)  # 190

    # Precompute all 4-cycles
    cycles = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    cycles.append((edge_idx[(a,b)], edge_idx[(b,c)], edge_idx[(c,d)], edge_idx[(d,a)]))
                    cycles.append((edge_idx[(a,b)], edge_idx[(b,d)], edge_idx[(d,c)], edge_idx[(c,a)]))
                    cycles.append((edge_idx[(a,c)], edge_idx[(c,b)], edge_idx[(b,d)], edge_idx[(d,a)]))

    ncyc = len(cycles)
    cycles_np = np.array(cycles, dtype=np.int32)

    # For each edge, list of (cycle_idx, position)
    edge_to_cycles = [[] for _ in range(m)]
    for ci in range(ncyc):
        for pos in range(4):
            e = cycles_np[ci, pos]
            edge_to_cycles[e].append((ci, pos))

    # Precompute for each edge and each cycle it's in, the other 3 edge indices
    # to speed up delta computation
    # For edge e at position pos in cycle ci, the other edges are at positions (0,1,2,3)\{pos}
    # We'll precompute: for each (e, ci, pos), store the 3 other edge indices
    edge_cycle_other = {}
    for e in range(m):
        edge_cycle_other[e] = []
        for (ci, pos) in edge_to_cycles[e]:
            others = [cycles_np[ci, p] for p in range(4) if p != pos]
            edge_cycle_other[e].append((ci, pos, tuple(others)))

    # Initial coloring
    colors = np.zeros(m, dtype=np.int32)
    for idx, (i, j) in enumerate(edges):
        colors[idx] = (i * 7 + j * 3) % C

    # Count violations (vectorized)
    def count_violations(col):
        cyc_colors = col[cycles_np]
        v = 0
        for p in range(4):
            for q in range(p + 1, 4):
                v += int(np.sum(cyc_colors[:, p] == cyc_colors[:, q]))
        return v

    cur_violations = count_violations(colors)
    best_violations = cur_violations
    best_colors = colors.copy()

    # Simulated annealing
    T = 3.0
    T_min = 0.005
    cooling = 0.99999
    max_steps = 5000000

    step = 0
    while step < max_steps:
        if time.time() - _t0 > _DEADLINE:
            break
        step += 1

        e = random.randint(0, m - 1)
        c_old = int(colors[e])
        c_new = random.randint(0, C - 1)
        if c_new == c_old:
            c_new = (c_new + 1) % C

        # Delta computation
        dv = 0
        for (ci, pos, others) in edge_cycle_other[e]:
            oc0, oc1, oc2 = others
            # Old conflicts: c_old matches any of the 3 others
            old_conf = (1 if colors[oc0] == c_old else 0) + \
                       (1 if colors[oc1] == c_old else 0) + \
                       (1 if colors[oc2] == c_old else 0)
            # New conflicts: c_new matches any of the 3 others
            new_conf = (1 if colors[oc0] == c_new else 0) + \
                       (1 if colors[oc1] == c_new else 0) + \
                       (1 if colors[oc2] == c_new else 0)
            dv += new_conf - old_conf

        if dv <= 0 or (T > T_min and random.random() < np.exp(-dv / T)):
            colors[e] = c_new
            cur_violations += dv
            if cur_violations < best_violations:
                best_violations = cur_violations
                best_colors = colors.copy()

        T *= cooling
        if T < T_min:
            T = T_min

    # Use best colors
    colors = best_colors

    # Build result
    result = []
    for idx, (i, j) in enumerate(edges):
        result.append([i, j, int(colors[idx])])

    # Quick verification
    adj = {}
    for e in result:
        u, v, c = e
        if u not in adj:
            adj[u] = {}
        if v not in adj:
            adj[v] = {}
        adj[u][v] = c
        adj[v][u] = c

    valid = True
    for a in range(n):
        if time.time() - _t0 > _DEADLINE:
            break
        for b in range(a + 1, n):
            if time.time() - _t0 > _DEADLINE:
                break
            for c in range(b + 1, n):
                if time.time() - _t0 > _DEADLINE:
                    break
                for d in range(c + 1, n):
                    if time.time() - _t0 > _DEADLINE:
                        break
                    for cyc in [
                        [(a,b),(b,c),(c,d),(d,a)],
                        [(a,b),(b,d),(d,c),(c,a)],
                        [(a,c),(c,b),(b,d),(d,a)]
                    ]:
                        cols = []
                        ok = True
                        for (u,v) in cyc:
                            if u in adj and v in adj[u]:
                                cols.append(adj[u][v])
                            else:
                                ok = False
                                break
                        if ok and len(cols) == 4:
                            if len(set(cols)) < 4:
                                valid = False
                                break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        return [[0, v, 0] for v in range(1, 20)]

    return result
# EVOLVE-BLOCK-END

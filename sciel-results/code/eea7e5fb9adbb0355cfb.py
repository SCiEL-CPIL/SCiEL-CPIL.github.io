"""Construct 91 edges on 31 vertices minimizing C4 count.
Uses PG(2,5) polarity graph cores (C4-free, 90 edges) + one edge,
then 2-swap local search to reduce total C4s.
"""
import numpy as np
import time
import random


def construct():
    t0 = time.time()
    deadline = t0 + 115.0

    # Enumerate all 31 points of PG(2,5)
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == y == z == 0:
                    continue
                if x != 0:
                    inv = pow(x, 3, 5)
                    key = (1, (y * inv) % 5, (z * inv) % 5)
                elif y != 0:
                    inv = pow(y, 3, 5)
                    key = (0, 1, (z * inv) % 5)
                else:
                    key = (0, 0, 1)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    assert len(points) == 31

    def count_c4(adj):
        total = 0
        for u in range(31):
            for v in range(u + 1, 31):
                cn = len(adj[u] & adj[v])
                if cn >= 2:
                    total += cn * (cn - 1) // 2
        return total // 4

    def build_adj(edge_list):
        adj = [set() for _ in range(31)]
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    # Try all symmetric 3x3 forms over GF(5)
    # Diagonal: B(p,q) = a*x1*x2 + b*y1*y2 + c*z1*z2
    # Non-diagonal: B(p,q) = a*x1*x2 + d*(x1*y2+y1*x2) + b*y1*y2 + c*z1*z2
    forms = []
    for a in range(1, 5):
        for b in range(1, 5):
            for c in range(1, 5):
                forms.append((a, b, c, 0))
                for d in range(1, 5):
                    forms.append((a, b, c, d))

    random.seed(42)
    random.shuffle(forms)

    best_c4 = float('inf')
    best_edges = None

    for form in forms:
        if time.time() > deadline - 30:
            break
        a, b, c, d = form
        edges_core = []
        for i in range(31):
            for j in range(i + 1, 31):
                x1, y1, z1 = points[i]
                x2, y2, z2 = points[j]
                val = (a * x1 * x2 + d * (x1 * y2 + y1 * x2) + b * y1 * y2 + c * z1 * z2) % 5
                if val == 0:
                    edges_core.append([i, j])

        if len(edges_core) != 90:
            continue

        adj = build_adj(edges_core)

        # Verify C4-free
        if count_c4(adj) != 0:
            continue

        # Find best single edge to add (minimizes new C4s)
        best_add = float('inf')
        best_edge = None
        for u in range(31):
            for v in range(u + 1, 31):
                if v in adj[u]:
                    continue
                add = 0
                for a2 in adj[u]:
                    for b2 in adj[v]:
                        if b2 in adj[a2]:
                            add += 1
                if add < best_add:
                    best_add = add
                    best_edge = [u, v]

        if best_edge is None:
            continue

        candidate = edges_core + [best_edge]
        cand_adj = build_adj(candidate)
        c4_val = count_c4(cand_adj)

        if c4_val < best_c4:
            best_c4 = c4_val
            best_edges = candidate

    if best_edges is None:
        return _greedy_construct()

    # Local search: 2-swap (remove 2 edges, add 2 non-edges)
    edges = [list(e) for e in best_edges]
    adj = build_adj(edges)
    current_c4 = count_c4(adj)
    best_c4_ls = current_c4
    best_edges_ls = [list(e) for e in edges]

    all_pairs = [(i, j) for i in range(31) for j in range(i + 1, 31)]
    T = 5.0
    cooling = 0.998
    max_time = deadline - time.time() - 5.0

    while time.time() < deadline - 5.0 and T > 0.01:
        # Pick 2 edges to remove
        i1 = random.randint(0, 90)
        i2 = random.randint(0, 90)
        if i1 == i2:
            T *= cooling
            continue
        u1, v1 = edges[i1]
        u2, v2 = edges[i2]

        # Pick 2 non-edges to add
        e1 = None
        e2 = None
        for _ in range(30):
            a3, b3 = random.choice(all_pairs)
            if b3 not in adj[a3] and (a3, b3) != (u1, v1) and (a3, b3) != (u2, v2):
                if e1 is None:
                    e1 = (a3, b3)
                else:
                    e2 = (a3, b3)
                    break
        if e1 is None or e2 is None:
            T *= cooling
            continue

        # Compute C4 delta: remove (u1,v1),(u2,v2); add e1,e2
        # C4s destroyed by removing an edge (u,v): paths of length 3 from u to v through other edges
        def c4s_using_edge(adj_local, u, v):
            cnt = 0
            for a2 in adj_local[u]:
                if a2 == v:
                    continue
                for b2 in adj_local[v]:
                    if b2 == u:
                        continue
                    if b2 in adj_local[a2]:
                        cnt += 1
            return cnt

        # C4s created by adding edge (u,v) to adj_local
        def c4s_created(adj_local, u, v):
            cnt = 0
            for a2 in adj_local[u]:
                for b2 in adj_local[v]:
                    if b2 in adj_local[a2]:
                        cnt += 1
            return cnt

        # Approximate: compute change
        c4_rm1 = c4s_using_edge(adj, u1, v1)
        c4_rm2 = c4s_using_edge(adj, u2, v2)
        c4_add1 = c4s_created(adj, e1[0], e1[1])
        c4_add2 = c4s_created(adj, e2[0], e2[1])

        # The interactions between the two removals/additions are complex;
        # use a rough estimate for acceptance, then verify
        delta_est = (c4_add1 + c4_add2) - (c4_rm1 + c4_rm2)

        if delta_est < 0 or random.random() < np.exp(-max(0, delta_est) / T):
            # Apply
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            edges[i1] = [e1[0], e1[1]]
            edges[i2] = [e2[0], e2[1]]
            adj[e1[0]].add(e1[1])
            adj[e1[1]].add(e1[0])
            adj[e2[0]].add(e2[1])
            adj[e2[1]].add(e2[0])

            current_c4 = count_c4(adj)
            if current_c4 < best_c4_ls:
                best_c4_ls = current_c4
                best_edges_ls = [list(e) for e in edges]
        else:
            # Don't apply, just cool
            pass

        T *= cooling

    if best_c4_ls < best_c4:
        best_edges = best_edges_ls

    # Validate
    assert len(best_edges) == 91
    edge_set = set(tuple(sorted(e)) for e in best_edges)
    assert len(edge_set) == 91

    return best_edges


def _greedy_construct():
    random.seed(42)
    edges = []
    adj = [set() for _ in range(31)]
    remaining = []
    for i in range(31):
        for j in range(i + 1, 31):
            remaining.append([i, j])
    random.shuffle(remaining)

    for _ in range(91):
        best_e = None
        best_add = float('inf')
        for e in remaining:
            u, v = e
            if v in adj[u]:
                continue
            add = 0
            for a in adj[u]:
                for b in adj[v]:
                    if b in adj[a]:
                        add += 1
            if add < best_add:
                best_add = add
                best_e = e
            if add == 0:
                break
        if best_e is None:
            break
        u, v = best_e
        edges.append(best_e)
        adj[u].add(v)
        adj[v].add(u)
        remaining.remove(best_e)

    while len(edges) < 91 and remaining:
        for e in remaining:
            u, v = e
            if v not in adj[u]:
                edges.append(e)
                adj[u].add(v)
                adj[v].add(u)
                remaining.remove(e)
                break

    return edges[:91]

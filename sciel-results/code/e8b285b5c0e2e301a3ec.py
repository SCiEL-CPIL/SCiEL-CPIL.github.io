import time
import sys
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # 1. Generate vertices of PG(2, 5)
    pts = []
    for x in range(5):
        for y in range(5):
            pts.append((x, y, 1))
    for y in range(5):
        pts.append((1, y, 0))
    pts.append((0, 1, 0))

    n = len(pts)
    assert n == 31

    # 2. Define Polarity: x1*x2 + y1*y2 + z1*z2 = 0 mod 5
    def is_adjacent(p1, p2):
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        return (x1*x2 + y1*y2 + z1*z2) % 5 == 0

    # Build adjacency for core graph (90 edges)
    adj = [set() for _ in range(31)]
    edges_core = []
    for i in range(31):
        for j in range(i+1, 31):
            if is_adjacent(pts[i], pts[j]):
                adj[i].add(j)
                adj[j].add(i)
                edges_core.append([i, j])

    edge_count = sum(len(s) for s in adj) // 2
    assert edge_count == 90, f"Expected 90 core edges, got {edge_count}"

    # 3. Find best edge to add (minimize new C4s)
    best_u = -1
    best_v = -1
    min_common = float('inf')

    for u in range(31):
        for v in range(u+1, 31):
            if v in adj[u]:
                continue
            common = len(adj[u] & adj[v])
            if common < min_common:
                min_common = common
                best_u = u
                best_v = v
                if min_common == 0:
                    break
        if min_common == 0:
            break

    if best_u == -1:
        for u in range(31):
            for v in range(u+1, 31):
                if v not in adj[u]:
                    best_u = u
                    best_v = v
                    break
            if best_u != -1:
                break

    # Add the edge
    adj[best_u].add(best_v)
    adj[best_v].add(best_u)
    edges_current = edges_core + [[best_u, best_v]]

    # 4. Bitset representation
    bitsets = [0] * 31
    for i in range(31):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        bitsets[i] = mask

    def count_c4_bitset(bs):
        total = 0
        for i in range(31):
            for j in range(i+1, 31):
                common = bs[i] & bs[j]
                c = bin(common).count('1')
                if c >= 2:
                    total += c * (c - 1) // 2
        return total // 2

    current_c4 = count_c4_bitset(bitsets)

    # 5. Local search with O(1) delta computation per swap
    edge_set = set()
    for e in edges_current:
        edge_set.add((min(e[0], e[1]), max(e[0], e[1])))

    non_edges = [(i, j) for i in range(31) for j in range(i+1, 31) if (i, j) not in edge_set]

    best_c4 = current_c4
    best_bitsets = bitsets[:]

    def swap_delta(bs, u, v, a, b):
        """Delta in C4 count when removing (u,v) and adding (a,b)"""
        # C4s destroyed: those using edge (u,v) = C(c(u,v), 2)
        c_uv = bin(bs[u] & bs[v]).count('1')
        destroyed = c_uv * (c_uv - 1) // 2 if c_uv >= 2 else 0

        # Temporarily remove (u,v) to compute c(a,b) after removal
        bu = bs[u]
        bv = bs[v]
        bs[u] = bu & ~(1 << v)
        bs[v] = bv & ~(1 << u)

        c_ab = bin(bs[a] & bs[b]).count('1')
        created = c_ab * (c_ab - 1) // 2 if c_ab >= 2 else 0

        # Restore
        bs[u] = bu
        bs[v] = bv

        return created - destroyed

    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        edge_order = list(range(len(edges_current)))
        random.shuffle(edge_order)

        for rem_idx in edge_order:
            if time.time() > deadline - 10:
                break

            ru, rv = edges_current[rem_idx]
            ru, rv = min(ru, rv), max(ru, rv)

            best_swap = None
            best_delta = 0

            for add_edge in non_edges:
                au, av = add_edge
                au, av = min(au, av), max(au, av)

                if (au, av) == (ru, rv):
                    continue

                d = swap_delta(bitsets, ru, rv, au, av)

                if d < best_delta:
                    best_delta = d
                    best_swap = (au, av)

            if best_swap is not None and best_delta < 0:
                au, av = best_swap
                # Apply the swap
                bitsets[ru] &= ~(1 << rv)
                bitsets[rv] &= ~(1 << ru)
                bitsets[au] |= (1 << av)
                bitsets[av] |= (1 << au)

                edges_current[rem_idx] = [au, av]
                edge_set.discard((ru, rv))
                edge_set.add((au, av))
                non_edges = [(i, j) for i in range(31) for j in range(i+1, 31) if (i, j) not in edge_set]

                best_c4 += best_delta
                best_bitsets = bitsets[:]
                improved = True
                break

    # Reconstruct edges from best_bitsets
    final_edges = []
    for i in range(31):
        for j in range(i+1, 31):
            if best_bitsets[i] & (1 << j):
                final_edges.append([i, j])

    assert len(final_edges) == 91, f"Expected 91 edges, got {len(final_edges)}"

    return final_edges

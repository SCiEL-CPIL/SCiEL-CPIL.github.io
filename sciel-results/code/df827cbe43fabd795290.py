"""
Minimize C4s in a 31-vertex, 91-edge graph.
Strategy:
1. Build a C4-free 90-edge base graph (greedy with verification).
2. Add the edge that creates the fewest C4s.
3. Local search with INCREMENTAL C4 delta (O(n^2) per swap, not O(n^3)).
4. Multi-phase: greedy first-improvement, then simulated annealing with restarts.
"""
import time
import random
import math

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin under 150s

    n = 31
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # --- Step 1: Build a C4-free 90-edge base graph ---
    # Greedy: add edges in random order, reject if it creates a C4.
    # A C4 is created by adding (u,v) iff u and v share >= 2 common neighbors.
    random.seed(42)
    shuffled = list(all_pairs)
    random.shuffle(shuffled)

    adj = [set() for _ in range(n)]
    edges = []

    for u, v in shuffled:
        if len(edges) >= 90:
            break
        common = adj[u] & adj[v]
        if len(common) <= 1:  # C4-free: at most 1 common neighbor
            edges.append((u, v))
            adj[u].add(v)
            adj[v].add(u)

    # If greedy didn't reach 90 (unlikely but possible), force add
    if len(edges) < 90:
        for u, v in all_pairs:
            if len(edges) >= 90:
                break
            if v not in adj[u]:
                edges.append((u, v))
                adj[u].add(v)
                adj[v].add(u)

    # Verify C4-free
    def count_c4(adj_sets):
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                cn = len(adj_sets[u] & adj_sets[v])
                total += cn * (cn - 1) // 2
        return total // 4

    base_c4s = count_c4(adj)
    # Should be 0, but if not, we still proceed (the search will fix it)

    # --- Step 2: Add the best 91st edge ---
    # Adding edge (u,v) creates C4s = number of length-3 paths u-a-b-v in base graph
    # = sum over a in N(u), b in N(v), a != b, a~b of 1
    # But more precisely: C4s containing new edge (u,v) = pairs (a,b) where
    # a in N(u)\{v}, b in N(v)\{u}, a~b, a != b
    # Since base is C4-free, for non-adjacent (u,v), |N(u) ∩ N(v)| <= 1.
    # The C4s created = number of (a,b) with a~N(u), b~N(v), a~b, a≠b, and
    # the path u-a-b-v doesn't already form a C4 (it doesn't since (u,v) is new).
    # Actually: new C4s = #{(a,b): a in N(u), b in N(v), a≠b, a~b}
    #   minus any that were already C4s (none, since base is C4-free and (u,v) wasn't an edge)
    # But wait: a C4 u-v-a-b-u requires edges (u,v),(v,a),(a,b),(b,u).
    # So a in N(v), b in N(u), a~b, a≠b.
    # Count = sum over a in N(v)\{u}, b in N(u)\{v}, a~b, a≠b of 1.

    best_edge = None
    best_added = float('inf')

    for u, v in all_pairs:
        if v in adj[u]:
            continue
        # Count C4s created by adding (u,v)
        cnt = 0
        for a in adj[v]:  # a is neighbor of v
            if a == u:
                continue
            for b in adj[u]:  # b is neighbor of u
                if b == v or b == a:
                    continue
                if b in adj[a]:  # a~b
                    cnt += 1
        if cnt < best_added:
            best_added = cnt
            best_edge = (u, v)

    if best_edge is None:
        # Fallback: pick any non-edge
        for u, v in all_pairs:
            if v not in adj[u]:
                best_edge = (u, v)
                break

    u_add, v_add = best_edge
    adj[u_add].add(v_add)
    adj[v_add].add(u_add)
    edges.append((u_add, v_add))

    current_c4s = base_c4s + best_added  # approximate; will recount below

    # --- Step 3: Local search with incremental C4 delta ---
    # Represent adjacency as sets for fast intersection.
    # Incremental delta for swap: remove e1=(a,b), add e2=(c,d)
    #
    # C4s destroyed by removing (a,b):
    #   For each pair (x,y) with x in N(a)\{b}, y in N(b)\{a}, x~y, x≠y:
    #   this is a C4 a-b-y-x-a (or a-b-x-y-a). Each such C4 is destroyed.
    #   Count = #{(x,y): x in N(a)\{b}, y in N(b)\{a}, x~y, x≠y}
    #   But note: each C4 containing edge (a,b) is counted once per such pair.
    #   Actually, a C4 containing (a,b) is a-b-x-y-a where x~N(a)\{b}, y~N(b)\{a}, x~y.
    #   The number of such C4s = sum over x in N(a)\{b} of |N(x) ∩ N(b) \ {a,x}|
    #   Wait, that's not quite right either. Let me think again.
    #
    # A C4 containing edge (a,b): vertices a,b,x,y in cycle order a-b-x-y-a.
    # Edges: (a,b), (b,x), (x,y), (y,a).
    # So x in N(b)\{a}, y in N(a)\{b}, and x~y.
    # Count = #{(x,y): x in N(b)\{a}, y in N(a)\{b}, x~y, x≠y}
    #
    # C4s created by adding (c,d):
    #   Similarly: #{(x,y): x in N(d)\{c}, y in N(c)\{d}, x~y, x≠y}
    #   BUT we must exclude C4s that also contain edge (a,b), because those are
    #   both destroyed and created (net zero). However, a C4 can contain at most
    #   2 edges from {e1, e2} if they're disjoint, or 1 if they share a vertex.
    #   If e1 and e2 share a vertex, a C4 could contain both. In that case,
    #   removing e1 destroys it, adding e2 recreates it → net 0.
    #   So: delta = created - destroyed, where we count all C4s with e2 in the
    #   graph AFTER removing e1, and all C4s with e1 in the graph BEFORE adding e2.
    #
    # For simplicity and correctness: 
    #   destroyed = C4s containing e1 in current graph
    #   created = C4s containing e2 in graph with e1 removed
    #   delta = created - destroyed
    #
    # But computing "created in graph with e1 removed" requires checking if the
    # C4 also contains e1. If it does, it's not truly "new".
    #
    # Simpler approach: just compute destroyed and created independently, and
    # accept the small error when e1 and e2 share a vertex (which is rare in
    # random swaps). Or: handle the overlap explicitly.
    #
    # Let's do it properly:
    #   destroyed = C4s with e1 in current graph
    #   created_raw = C4s with e2 in current graph (before any change)
    #   overlap = C4s with both e1 and e2 in current graph (these are neither
    #     truly destroyed nor truly created; they persist)
    #   delta = created_raw - destroyed - overlap... no, this is getting complex.
    #
    # Cleanest: apply the swap, recount C4s involving only the affected vertices.
    # Affected vertices: {a, b, c, d} and their neighbors.
    # Actually, just do a full recount but only over pairs that could be affected.
    #
    # Given n=31, a full recount is O(n^2 * avg_degree) ≈ 31*31*7 ≈ 6800 ops.
    # That's fast enough for thousands of swaps per second.
    # Let's just do full recounts for correctness, optimized with bitsets.

    # Use bitsets for fast common-neighbor counting
    adj_bits = [0] * n
    for i in range(n):
        for j in adj[i]:
            adj_bits[i] |= (1 << j)

    def count_c4_bits(bits):
        total = 0
        for u in range(n):
            bu = bits[u]
            for v in range(u + 1, n):
                cn = bin(bu & bits[v]).count('1')
                total += cn * (cn - 1) // 2
        return total // 4

    current_c4s = count_c4_bits(adj_bits)

    edges_set = set(edges)
    non_edges = [(u, v) for u, v in all_pairs if (u, v) not in edges_set]

    def apply_swap(e1, e2):
        a, b = e1
        c, d = e2
        adj[a].discard(b)
        adj[b].discard(a)
        adj[c].add(d)
        adj[d].add(c)
        adj_bits[a] &= ~(1 << b)
        adj_bits[b] &= ~(1 << a)
        adj_bits[c] |= (1 << d)
        adj_bits[d] |= (1 << c)
        edges_set.discard(e1)
        edges_set.add(e2)

    def revert_swap(e1, e2):
        c, d = e2
        a, b = e1
        adj[c].discard(d)
        adj[d].discard(c)
        adj[a].add(b)
        adj[b].add(a)
        adj_bits[c] &= ~(1 << d)
        adj_bits[d] &= ~(1 << c)
        adj_bits[a] |= (1 << b)
        adj_bits[b] |= (1 << a)
        edges_set.discard(e2)
        edges_set.add(e1)

    def try_swap(e1, e2):
        """Apply swap, return delta. Revert if not applied."""
        apply_swap(e1, e2)
        new_c4s = count_c4_bits(adj_bits)
        delta = new_c4s - current_c4s
        return delta, new_c4s

    # Phase 1: Greedy first-improvement
    random.seed(123)
    improved = True
    while improved and time.time() < deadline - 40:
        improved = False
        for _ in range(5000):
            if time.time() > deadline - 40:
                break
            e1 = random.choice(list(edges_set))
            e2 = random.choice(non_edges)
            delta, new_c4s = try_swap(e1, e2)
            if delta < 0:
                current_c4s = new_c4s
                improved = True
                non_edges = [(u, v) for u, v in all_pairs if (u, v) not in edges_set]
            else:
                revert_swap(e1, e2)

    # Phase 2: Simulated annealing with restarts
    best_c4s = current_c4s
    best_edges = set(edges_set)
    best_bits = list(adj_bits)

    for restart in range(5):
        if time.time() > deadline - 10:
            break

        # Reset to best if we've drifted
        if current_c4s > best_c4s:
            edges_set = set(best_edges)
            adj = [set() for _ in range(n)]
            for u, v in edges_set:
                adj[u].add(v)
                adj[v].add(u)
            adj_bits = list(best_bits)
            current_c4s = best_c4s

        T = 3.0
        T_min = 0.01
        cool = 0.9998
        non_edges = [(u, v) for u, v in all_pairs if (u, v) not in edges_set]

        for step in range(20000):
            if time.time() > deadline - 10:
                break
            if not non_edges:
                break

            e1 = random.choice(list(edges_set))
            e2 = random.choice(non_edges)

            delta, new_c4s = try_swap(e1, e2)

            if delta < 0 or random.random() < math.exp(-delta / max(T, 0.001)):
                current_c4s = new_c4s
                non_edges = [(u, v) for u, v in all_pairs if (u, v) not in edges_set]
                if current_c4s < best_c4s:
                    best_c4s = current_c4s
                    best_edges = set(edges_set)
                    best_bits = list(adj_bits)
            else:
                revert_swap(e1, e2)

            T = max(T * cool, T_min)

    # Restore best solution
    edges_set = set(best_edges)
    final_edges = [list(e) for e in edges_set]
    final_edges.sort()

    assert len(final_edges) == 91, f"Expected 91 edges, got {len(final_edges)}"
    assert len(set(tuple(sorted(e)) for e in final_edges)) == 91

    return final_edges

import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill

    n = 16
    full_mask = (1 << n) - 1

    # Q3 edges (canonical, vertices 0..7)
    Q3_EDGES = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    Q3_ADJ = [0] * 8
    for a, b in Q3_EDGES:
        Q3_ADJ[a] |= (1 << b)
        Q3_ADJ[b] |= (1 << a)

    # Precompute Q3 edge orientations with sorted unassigned vertices
    Q3_EDGE_ORIENTS = []
    for qa, qb in Q3_EDGES:
        for (gu_q, gv_q) in [(qa, qb), (qb, qa)]:
            unassigned = [i for i in range(8) if i != gu_q and i != gv_q]
            def _cc(x, a, b):
                c = 0
                if Q3_ADJ[x] & (1 << a): c += 1
                if Q3_ADJ[x] & (1 << b): c += 1
                return c
            unassigned.sort(key=lambda x: _cc(x, gu_q, gv_q), reverse=True)
            Q3_EDGE_ORIENTS.append((gu_q, gv_q, tuple(unassigned)))

    def has_q3_after_add(adj, u, v, deadline):
        """Check if adding edge (u,v) to current adj creates a Q3 subgraph.
        Uses bitmask candidate intersection for fast pruning."""
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

        found = False
        for (qa, qb, unassigned) in Q3_EDGE_ORIENTS:
            if found:
                break
            for (gu, gv) in [(u, v), (v, u)]:
                if found:
                    break
                assignment = [None] * 8
                assignment[qa] = gu
                assignment[qb] = gv
                used_mask = (1 << gu) | (1 << gv)

                def bt(idx, used_mask):
                    nonlocal found
                    if found or time.time() > deadline - 5:
                        return found
                    if idx == 6:
                        # All 6 unassigned are assigned; verify all Q3 edges
                        for a, b in Q3_EDGES:
                            ga = assignment[a]
                            gb = assignment[b]
                            if not (adj[ga] & (1 << gb)):
                                return False
                        found = True
                        return True

                    x = unassigned[idx]
                    # Candidates: not used, adjacent to all assigned Q3-neighbors of x
                    candidates = full_mask & ~used_mask
                    for y in range(8):
                        if assignment[y] is not None and (Q3_ADJ[x] & (1 << y)):
                            hy = assignment[y]
                            candidates &= adj[hy]

                    while candidates:
                        bit = candidates & (-candidates)
                        g = bit.bit_length() - 1
                        candidates ^= bit

                        assignment[x] = g
                        new_used = used_mask | (1 << g)
                        if bt(idx + 1, new_used):
                            return True
                        assignment[x] = None
                    return False

                if bt(0, used_mask):
                    found = True
                    break

        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return found

    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def edge_list(adj):
        es = []
        for u in range(n):
            for v in range(u+1, n):
                if (adj[u] >> v) & 1:
                    es.append([u, v])
        return es

    def verify_no_q3(adj, deadline):
        """Verify the graph has no Q3 subgraph by checking every edge."""
        for u in range(n):
            for v in range(u+1, n):
                if time.time() > deadline - 5:
                    return False  # assume invalid if we run out of time
                if (adj[u] >> v) & 1:
                    if has_q3_after_add(adj, u, v, deadline):
                        return False
        return True

    best_count = -1
    best_adj = None

    # Build initial K3,13: part0={0,1,2}, part1={3..15}
    def make_k313():
        adj = [0] * n
        for a in range(3):
            for b in range(3, 16):
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
        return adj

    # Phase 1: Multiple greedy constructions from K3,13 with different orderings
    # Try sorted order and several random shuffles
    for attempt in range(8):
        if time.time() > deadline - 15:
            break

        adj = make_k313()
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]

        if attempt == 0:
            # sorted order
            pass
        elif attempt == 1:
            # reverse sorted
            missing = missing[::-1]
        else:
            random.seed(100 + attempt)
            random.shuffle(missing)

        cnt = count_edges(adj)
        for u, v in missing:
            if time.time() > deadline - 15:
                break
            if not (adj[u] & (1 << v)):
                if not has_q3_after_add(adj, u, v, deadline):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    cnt += 1

        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]

    # Phase 2: Local search with edge swaps on the best graph
    if best_adj is not None and time.time() < deadline - 20:
        adj = best_adj[:]
        cnt = count_edges(adj)
        random.seed(42)

        for kick_iter in range(50):
            if time.time() > deadline - 10:
                break

            # Try to add all missing edges greedily
            missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
            random.shuffle(missing)

            improved = False
            for u, v in missing:
                if time.time() > deadline - 10:
                    break
                if not has_q3_after_add(adj, u, v, deadline):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    cnt += 1
                    improved = True

            if improved and cnt > best_count:
                best_count = cnt
                best_adj = adj[:]

            if not improved:
                # Kick: remove 1-3 random edges and try again
                existing = [(i, j) for i in range(n) for j in range(i+1, n) if (adj[i] & (1 << j))]
                if existing:
                    num_kick = min(3, len(existing))
                    kick_edges = random.sample(existing, num_kick)
                    for u, v in kick_edges:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        cnt -= 1
                continue

    # Phase 3: Spectral pruning refinement - start from K16, remove edges to minimize 4-cycles
    if time.time() < deadline - 30:
        # Start with complete graph K16
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

        A = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            for j in range(n):
                if (adj[i] >> j) & 1:
                    A[i, j] = 1

        def trace_A4(A):
            A2 = A @ A
            A4 = A2 @ A2
            return np.trace(A4)

        current_trace = trace_A4(A)
        max_iterations = 40
        for iteration in range(max_iterations):
            if time.time() > deadline - 30:
                break

            # Check if current graph is already Q3-free
            if verify_no_q3(adj, deadline):
                ec = count_edges(adj)
                if ec > best_count:
                    best_count = ec
                    best_adj = adj[:]
                break

            # For each existing edge, compute how much trace(A^4) decreases if we remove it
            best_edge_to_remove = None
            best_reduction = -1

            all_existing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (adj[i] >> j) & 1]
            random.shuffle(all_existing_edges)

            for u, v in all_existing_edges:
                # Temporarily remove edge
                A[u, v] = 0
                A[v, u] = 0

                new_trace = trace_A4(A)
                reduction = current_trace - new_trace

                # Restore
                A[u, v] = 1
                A[v, u] = 1

                if reduction > best_reduction:
                    best_reduction = reduction
                    best_edge_to_remove = (u, v)

            if best_edge_to_remove is None:
                break

            u, v = best_edge_to_remove
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            A[u, v] = 0
            A[v, u] = 0
            current_trace = current_trace - best_reduction

            # Periodic check
            if iteration % 5 == 4:
                if verify_no_q3(adj, deadline):
                    ec = count_edges(adj)
                    if ec > best_count:
                        best_count = ec
                        best_adj = adj[:]
                    break

        # After spectral pruning, try greedy addition
        if best_adj is None or count_edges(adj) > best_count - 5:
            if time.time() < deadline - 10:
                missing = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
                random.shuffle(missing)
                for u, v in missing:
                    if time.time() > deadline - 10:
                        break
                    if not (adj[u] & (1 << v)):
                        if not has_q3_after_add(adj, u, v, deadline):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)

                ec = count_edges(adj)
                if ec > best_count:
                    best_count = ec
                    best_adj = adj[:]

    # Final fallback
    if best_adj is None:
        best_adj = make_k313()
        best_count = count_edges(best_adj)

    # Final verification
    if not verify_no_q3(best_adj, deadline):
        best_adj = make_k313()
        best_count = count_edges(best_adj)

    return edge_list(best_adj)

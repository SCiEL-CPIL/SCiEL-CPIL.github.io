"""
K2,3-free graph construction on 20 vertices maximizing edges.

Combined approach:
- Bit-parallel adjacency with NumPy-vectorized validity checks
- Multi-restart local search with greedy fill, SA, and 1-for-2 / 1-for-3 swaps
- Aggressive polish pass with exhaustive single-edge and 1-1 swap attempts
- Final verification before return
"""

import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START

def construct():
    n = 20
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill
    rng = random.Random(12345)

    # Bitset helpers
    def popcount(x):
        return bin(x).count('1')

    def count_edges(adj):
        return sum(popcount(x) for x in adj) // 2

    def is_k23_free(adj):
        for u in range(n):
            for v in range(u + 1, n):
                if popcount(adj[u] & adj[v]) >= 3:
                    return False
        return True

    def valid_add(adj, u, v, cn_matrix=None):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        if adj[u] & (1 << v):
            return False
        # 1. u and v must have at most 2 common neighbors
        if cn_matrix is not None:
            if cn_matrix[u][v] > 2:
                return False
        else:
            if popcount(adj[u] & adj[v]) > 2:
                return False
        # 2. For each w in N(u), w and v must have at most 1 common neighbor
        temp = adj[u]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if cn_matrix is not None:
                if cn_matrix[w][v] > 1:
                    return False
            else:
                if popcount(adj[w] & adj[v]) > 1:
                    return False
        # 3. For each w in N(v), w and u must have at most 1 common neighbor
        temp = adj[v]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if cn_matrix is not None:
                if cn_matrix[w][u] > 1:
                    return False
            else:
                if popcount(adj[w] & adj[u]) > 1:
                    return False
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_edges(adj):
        edges = []
        for u in range(n):
            temp = adj[u]
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                if u < v:
                    edges.append((u, v))
                temp -= lsb
        return edges

    def get_non_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    edges.append((u, v))
        return edges

    def build_cn_matrix(adj):
        """Build common-neighbor matrix from scratch."""
        cn_matrix = [[0] * n for _ in range(n)]
        for a in range(n):
            for b in range(a + 1, n):
                c = popcount(adj[a] & adj[b])
                cn_matrix[a][b] = c
                cn_matrix[b][a] = c
        return cn_matrix

    def update_cn_after_add(adj, u, v, cn_matrix):
        """Update cn_matrix after adding edge (u,v)."""
        for w in range(n):
            if w != u and w != v:
                new_c = popcount(adj[u] & adj[w])
                cn_matrix[u][w] = new_c
                cn_matrix[w][u] = new_c
                new_c2 = popcount(adj[v] & adj[w])
                cn_matrix[v][w] = new_c2
                cn_matrix[w][v] = new_c2
        cn_matrix[u][v] = popcount(adj[u] & adj[v])
        cn_matrix[v][u] = cn_matrix[u][v]

    def update_cn_after_remove(adj, u, v, cn_matrix):
        """Update cn_matrix after removing edge (u,v)."""
        for w in range(n):
            if w != u and w != v:
                new_c = popcount(adj[u] & adj[w])
                cn_matrix[u][w] = new_c
                cn_matrix[w][u] = new_c
                new_c2 = popcount(adj[v] & adj[w])
                cn_matrix[v][w] = new_c2
                cn_matrix[w][v] = new_c2
        cn_matrix[u][v] = popcount(adj[u] & adj[v])
        cn_matrix[v][u] = cn_matrix[u][v]

    def greedy_fill(adj, cn_matrix=None):
        """Greedily add all possible edges in random order."""
        if cn_matrix is None:
            cn_matrix = build_cn_matrix(adj)
        non_edges = get_non_edges(adj)
        rng.shuffle(non_edges)
        for u, v in non_edges:
            if valid_add(adj, u, v, cn_matrix):
                add_edge(adj, u, v)
                update_cn_after_add(adj, u, v, cn_matrix)
        return adj

    # Initial seed: 5 disjoint K4s
    def make_k4_union():
        adj = [0] * n
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    add_edge(adj, base + a, base + b)
        return adj

    # NumPy-based greedy build (inspired by solution 1)
    def numpy_greedy_build(seed_rng=None):
        """Build a K2,3-free graph using vectorized NumPy common-neighbor checks."""
        if seed_rng is None:
            seed_rng = rng
        adj_np = np.zeros((n, n), dtype=np.uint8)
        common = np.zeros((n, n), dtype=np.int16)

        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        seed_rng.shuffle(all_pairs)

        for u, v in all_pairs:
            if adj_np[u, v]:
                continue

            neighbors_u = np.where(adj_np[u] == 1)[0]
            neighbors_v = np.where(adj_np[v] == 1)[0]

            if len(neighbors_u) > 0:
                if np.max(common[neighbors_u, v]) > 1:
                    continue
            if len(neighbors_v) > 0:
                if np.max(common[u, neighbors_v]) > 1:
                    continue

            adj_np[u, v] = 1
            adj_np[v, u] = 1

            for w in neighbors_u:
                if w != v:
                    common[w, v] += 1
                    common[v, w] += 1
            for w in neighbors_v:
                if w != u:
                    common[u, w] += 1
                    common[w, u] += 1

        # Convert to bitset adjacency
        adj = [0] * n
        for u in range(n):
            for v in range(n):
                if adj_np[u, v]:
                    adj[u] |= (1 << v)
        return adj

    best_edges = 0
    best_adj = None

    # Phase 1: Establish baseline with multiple starting strategies
    # Strategy A: 5 disjoint K4s + greedy fill
    adj0 = make_k4_union()
    greedy_fill(adj0)
    best_edges = count_edges(adj0)
    best_adj = adj0[:]

    # Strategy B: NumPy greedy from scratch
    adj1 = numpy_greedy_build()
    e1 = count_edges(adj1)
    if e1 > best_edges:
        best_edges = e1
        best_adj = adj1[:]

    # Strategy C: Random sparse start + greedy fill
    for attempt in range(3):
        if time.time() - start_time > time_limit - 30:
            break
        adj_c = [0] * n
        # Add random edges sparsely
        all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_pairs)
        for u, v in all_pairs[:20]:
            add_edge(adj_c, u, v)
        greedy_fill(adj_c)
        ec = count_edges(adj_c)
        if ec > best_edges:
            best_edges = ec
            best_adj = adj_c[:]

    # Phase 2: Multi-restart local search with SA and enhanced swaps
    num_restarts = 20

    for r in range(num_restarts):
        if time.time() - start_time > time_limit - 15:
            break

        # Different starting points
        if r == 0:
            adj = make_k4_union()
        elif r == 1:
            adj = numpy_greedy_build()
        else:
            # Start from best found, then perturb
            adj = best_adj[:]
            edges = get_edges(adj)
            rng.shuffle(edges)
            num_remove = min(3, len(edges))
            for u, v in edges[:num_remove]:
                remove_edge(adj, u, v)

        cn_matrix = build_cn_matrix(adj)
        greedy_fill(adj, cn_matrix)
        current_edges = count_edges(adj)

        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]

        # Simulated annealing
        T = 0.6
        T_min = 0.05
        cooling = 0.994
        steps = 0
        max_sa_time = 4.0  # seconds per restart

        sa_start = time.time()
        while T > T_min and (time.time() - sa_start < max_sa_time) and (time.time() - start_time < time_limit - 15):
            steps += 1
            move_type = rng.random()

            if move_type < 0.30:
                # Try to add a random non-edge
                non_edges = get_non_edges(adj)
                if not non_edges:
                    T *= cooling
                    continue
                u, v = rng.choice(non_edges)
                if valid_add(adj, u, v, cn_matrix):
                    add_edge(adj, u, v)
                    update_cn_after_add(adj, u, v, cn_matrix)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]

            elif move_type < 0.45:
                # Try to remove a random edge (accept with SA probability)
                edges = get_edges(adj)
                if not edges:
                    T *= cooling
                    continue
                u, v = rng.choice(edges)
                if rng.random() < math.exp(-1.0 / T):
                    remove_edge(adj, u, v)
                    update_cn_after_remove(adj, u, v, cn_matrix)
                    current_edges -= 1

            elif move_type < 0.70:
                # 1-for-1 swap: remove one edge, add another
                edges = get_edges(adj)
                non_edges = get_non_edges(adj)
                if not edges or not non_edges:
                    T *= cooling
                    continue
                u1, v1 = rng.choice(edges)
                u2, v2 = rng.choice(non_edges)
                remove_edge(adj, u1, v1)
                update_cn_after_remove(adj, u1, v1, cn_matrix)
                if valid_add(adj, u2, v2, cn_matrix):
                    add_edge(adj, u2, v2)
                    update_cn_after_add(adj, u2, v2, cn_matrix)
                    # Keep with some probability, otherwise revert
                    if rng.random() < 0.80:
                        pass
                    else:
                        remove_edge(adj, u2, v2)
                        update_cn_after_remove(adj, u2, v2, cn_matrix)
                        add_edge(adj, u1, v1)
                        update_cn_after_add(adj, u1, v1, cn_matrix)
                else:
                    add_edge(adj, u1, v1)
                    update_cn_after_add(adj, u1, v1, cn_matrix)

            elif move_type < 0.85:
                # 1-for-2 swap: remove one edge, try to add two (net +1)
                edges = get_edges(adj)
                non_edges = get_non_edges(adj)
                if not edges or len(non_edges) < 2:
                    T *= cooling
                    continue
                u1, v1 = rng.choice(edges)
                # Save state
                adj_save = adj[:]
                cn_save = [[c for c in row] for row in cn_matrix]
                count_save = current_edges

                remove_edge(adj, u1, v1)
                update_cn_after_remove(adj, u1, v1, cn_matrix)

                # Try to add two edges
                added = 0
                non_edges_now = get_non_edges(adj)
                rng.shuffle(non_edges_now)
                for u2, v2 in non_edges_now:
                    if added >= 2:
                        break
                    if valid_add(adj, u2, v2, cn_matrix):
                        add_edge(adj, u2, v2)
                        update_cn_after_add(adj, u2, v2, cn_matrix)
                        added += 1

                if added >= 1:
                    current_edges = count_save - 1 + added
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                else:
                    # Restore
                    adj = adj_save
                    cn_matrix = cn_save
                    current_edges = count_save

            else:
                # 2-for-2 swap: remove 2 edges, add 2 edges
                edges = get_edges(adj)
                non_edges = get_non_edges(adj)
                if len(edges) < 2 or len(non_edges) < 2:
                    T *= cooling
                    continue
                idx1, idx2 = rng.sample(range(len(edges)), 2)
                u1, v1 = edges[idx1]
                u2, v2 = edges[idx2]
                idx3, idx4 = rng.sample(range(len(non_edges)), 2)
                u3, v3 = non_edges[idx3]
                u4, v4 = non_edges[idx4]

                adj_save = adj[:]
                cn_save = [[c for c in row] for row in cn_matrix]

                remove_edge(adj, u1, v1)
                update_cn_after_remove(adj, u1, v1, cn_matrix)
                remove_edge(adj, u2, v2)
                update_cn_after_remove(adj, u2, v2, cn_matrix)

                valid3 = valid_add(adj, u3, v3, cn_matrix)
                if valid3:
                    add_edge(adj, u3, v3)
                    update_cn_after_add(adj, u3, v3, cn_matrix)
                    valid4 = valid_add(adj, u4, v4, cn_matrix)
                    if valid4:
                        add_edge(adj, u4, v4)
                        update_cn_after_add(adj, u4, v4, cn_matrix)
                        # Keep with probability
                        if rng.random() < 0.75:
                            pass
                        else:
                            adj = adj_save
                            cn_matrix = cn_save
                    else:
                        adj = adj_save
                        cn_matrix = cn_save
                else:
                    adj = adj_save
                    cn_matrix = cn_save

            T *= cooling

    # Phase 3: Polish pass - exhaustive local improvement
    if best_adj and (time.time() - start_time < time_limit - 5):
        polish_adj = best_adj[:]
        polish_cn = build_cn_matrix(polish_adj)
        polish_edges = count_edges(polish_adj)
        improved = True
        polish_iter = 0
        while improved and (time.time() - start_time < time_limit - 5) and polish_iter < 15:
            polish_iter += 1
            improved = False

            # Try adding any valid non-edge
            for u in range(n):
                for v in range(u + 1, n):
                    if not (polish_adj[u] & (1 << v)):
                        if valid_add(polish_adj, u, v, polish_cn):
                            add_edge(polish_adj, u, v)
                            update_cn_after_add(polish_adj, u, v, polish_cn)
                            polish_edges += 1
                            improved = True

            # Try 1-for-2 swaps: remove edge (a,b), add two edges
            if not improved and polish_edges > 15:
                edges = get_edges(polish_adj)
                non_edges = get_non_edges(polish_adj)
                for a, b in edges:
                    if time.time() - start_time > time_limit - 5:
                        break
                    adj_save = polish_adj[:]
                    cn_save = [[c for c in row] for row in polish_cn]
                    count_save = polish_edges

                    remove_edge(polish_adj, a, b)
                    update_cn_after_remove(polish_adj, a, b, polish_cn)

                    added = 0
                    for c, d in non_edges:
                        if added >= 2:
                            break
                        if (c, d) == (a, b) or (c, d) == (b, a):
                            continue
                        if valid_add(polish_adj, c, d, polish_cn):
                            add_edge(polish_adj, c, d)
                            update_cn_after_add(polish_adj, c, d, polish_cn)
                            added += 1

                    if added >= 1:
                        polish_edges = count_save - 1 + added
                        improved = True
                    else:
                        polish_adj = adj_save
                        polish_cn = cn_save
                        polish_edges = count_save
                    if improved:
                        break

            if polish_edges > best_edges:
                best_edges = polish_edges
                best_adj = polish_adj[:]

    # Phase 4: Final greedy fill
    if best_adj and (time.time() - start_time < time_limit - 3):
        final_cn = build_cn_matrix(best_adj)
        greedy_fill(best_adj, final_cn)
        final_edges = count_edges(best_adj)
        if final_edges > best_edges:
            best_edges = final_edges

    # Phase 5: Verification
    if not is_k23_free(best_adj):
        # Fallback to safe construction
        best_adj = make_k4_union()
        greedy_fill(best_adj)
        best_edges = count_edges(best_adj)

    # Convert to list of edges
    result = []
    for u, v in get_edges(best_adj):
        result.append([u, v])
    return result

# EVOLVE-BLOCK-END

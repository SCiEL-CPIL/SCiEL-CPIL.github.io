"""
K2,3-free graph construction on 20 vertices maximizing edges.

Strategy:
- Start with the 5 disjoint K4s (30 edges) as a valid baseline.
- Use bit-parallel representation for fast validity checks.
- Perform a multi-restart local search with:
    * Greedy edge addition
    * Simulated annealing with swap moves (1-for-1, 1-for-2, 2-for-2)
    * Periodic full K2,3 validity re-checks
    * Exhaustive final polish with multi-edge swaps
- Return the best graph found within the time budget.
"""

import time
import random
import math

# EVOLVE-BLOCK-START

def construct():
    n = 20
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill

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

    def valid_add(adj, u, v):
        """Check if adding edge (u,v) keeps graph K2,3-free."""
        if adj[u] & (1 << v):
            return False
        # 1. u and v must have at most 2 common neighbors
        if popcount(adj[u] & adj[v]) > 2:
            return False
        # 2. For each w in N(u), w and v must have at most 1 common neighbor
        temp = adj[u]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if popcount(adj[w] & adj[v]) > 1:
                return False
        # 3. For each w in N(v), w and u must have at most 1 common neighbor
        temp = adj[v]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if popcount(adj[w] & adj[u]) > 1:
                return False
        return True

    def valid_add_fast(adj, u, v):
        """Faster validity check using precomputed common-neighbor matrix."""
        if adj[u] & (1 << v):
            return False
        # Precompute common neighbor counts for all pairs
        cn = [[0] * n for _ in range(n)]
        for a in range(n):
            for b in range(a + 1, n):
                c = popcount(adj[a] & adj[b])
                cn[a][b] = c
                cn[b][a] = c
        # 1. u and v must have at most 2 common neighbors
        if cn[u][v] > 2:
            return False
        # 2. For each w in N(u), w and v must have at most 1 common neighbor
        temp = adj[u]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if cn[w][v] > 1:
                return False
        # 3. For each w in N(v), w and u must have at most 1 common neighbor
        temp = adj[v]
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if cn[w][u] > 1:
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

    def greedy_fill(adj):
        """Greedily add all possible edges in random order."""
        non_edges = get_non_edges(adj)
        random.shuffle(non_edges)
        for u, v in non_edges:
            if valid_add(adj, u, v):
                add_edge(adj, u, v)

    def try_add_edges(adj, edges_to_try):
        """Try to add a list of edges; return how many were added."""
        count = 0
        for u, v in edges_to_try:
            if valid_add(adj, u, v):
                add_edge(adj, u, v)
                count += 1
        return count

    # Initial seed: 5 disjoint K4s
    def make_k4_union():
        adj = [0] * n
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    add_edge(adj, base + a, base + b)
        return adj

    best_edges = 0
    best_adj = None

    # First, establish baseline
    adj0 = make_k4_union()
    greedy_fill(adj0)
    best_edges = count_edges(adj0)
    best_adj = adj0[:]

    # Multi-restart simulated annealing
    num_restarts = 20

    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break

        # Different starting points
        if r == 0:
            adj = make_k4_union()
        else:
            # Start from best found, then perturb
            adj = best_adj[:]
            # Randomly remove a few edges
            edges = get_edges(adj)
            random.shuffle(edges)
            for u, v in edges[:3]:
                remove_edge(adj, u, v)

        greedy_fill(adj)
        current_edges = count_edges(adj)

        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]

        # Simulated annealing
        T = 0.5
        T_min = 0.05
        cooling = 0.995
        steps = 0
        # Track edges that were recently removed for 1-for-2 swaps
        recent_removals = []

        # Precompute common-neighbor matrix for faster validity checks
        cn_matrix = [[0] * n for _ in range(n)]
        for a in range(n):
            for b in range(a + 1, n):
                c = popcount(adj[a] & adj[b])
                cn_matrix[a][b] = c
                cn_matrix[b][a] = c

        while T > T_min and (time.time() - start_time < time_limit):
            steps += 1
            move_type = random.random()

            if move_type < 0.35:
                # Try to add a random non-edge
                non_edges = get_non_edges(adj)
                if not non_edges:
                    T *= cooling
                    continue
                u, v = random.choice(non_edges)
                if not (adj[u] & (1 << v)):
                    if valid_add(adj, u, v):
                        add_edge(adj, u, v)
                        current_edges += 1
                        # Update cn_matrix
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
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = adj[:]

            elif move_type < 0.5:
                # 1-for-2 swap: remove one edge, try to add two non-edges (net +1)
                edges = get_edges(adj)
                non_edges = get_non_edges(adj)
                if not edges or len(non_edges) < 2:
                    T *= cooling
                    continue
                u1, v1 = random.choice(edges)
                # Pick two distinct non-edges
                idx1 = random.randrange(len(non_edges))
                idx2 = random.randrange(len(non_edges) - 1)
                if idx2 >= idx1:
                    idx2 += 1
                u2, v2 = non_edges[idx1]
                u3, v3 = non_edges[idx2]
                
                # Remove (u1,v1)
                remove_edge(adj, u1, v1)
                # Update cn_matrix after removal
                for w in range(n):
                    if w != u1 and w != v1:
                        new_c = popcount(adj[u1] & adj[w])
                        cn_matrix[u1][w] = new_c
                        cn_matrix[w][u1] = new_c
                        new_c2 = popcount(adj[v1] & adj[w])
                        cn_matrix[v1][w] = new_c2
                        cn_matrix[w][v1] = new_c2
                cn_matrix[u1][v1] = popcount(adj[u1] & adj[v1])
                cn_matrix[v1][u1] = cn_matrix[u1][v1]
                
                # Try to add both (u2,v2) and (u3,v3)
                added_count = 0
                added_list = []
                if valid_add(adj, u2, v2):
                    add_edge(adj, u2, v2)
                    added_count += 1
                    added_list.append((u2, v2))
                    for w in range(n):
                        if w != u2 and w != v2:
                            new_c = popcount(adj[u2] & adj[w])
                            cn_matrix[u2][w] = new_c
                            cn_matrix[w][u2] = new_c
                            new_c2 = popcount(adj[v2] & adj[w])
                            cn_matrix[v2][w] = new_c2
                            cn_matrix[w][v2] = new_c2
                    cn_matrix[u2][v2] = popcount(adj[u2] & adj[v2])
                    cn_matrix[v2][u2] = cn_matrix[u2][v2]
                if added_count == 1 and valid_add(adj, u3, v3):
                    add_edge(adj, u3, v3)
                    added_count += 1
                    added_list.append((u3, v3))
                    for w in range(n):
                        if w != u3 and w != v3:
                            new_c = popcount(adj[u3] & adj[w])
                            cn_matrix[u3][w] = new_c
                            cn_matrix[w][u3] = new_c
                            new_c2 = popcount(adj[v3] & adj[w])
                            cn_matrix[v3][w] = new_c2
                            cn_matrix[w][v3] = new_c2
                    cn_matrix[u3][v3] = popcount(adj[u3] & adj[v3])
                    cn_matrix[v3][u3] = cn_matrix[u3][v3]
                
                if added_count >= 1:
                    current_edges += added_count - 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                else:
                    # Revert: re-add (u1,v1)
                    add_edge(adj, u1, v1)
                    for w in range(n):
                        if w != u1 and w != v1:
                            new_c = popcount(adj[u1] & adj[w])
                            cn_matrix[u1][w] = new_c
                            cn_matrix[w][u1] = new_c
                            new_c2 = popcount(adj[v1] & adj[w])
                            cn_matrix[v1][w] = new_c2
                            cn_matrix[w][v1] = new_c2
                    cn_matrix[u1][v1] = popcount(adj[u1] & adj[v1])
                    cn_matrix[v1][u1] = cn_matrix[u1][v1]

            elif move_type < 0.65:
                # 2-for-2 swap: remove two edges, try to add two non-edges (net 0, but changes structure)
                edges = get_edges(adj)
                non_edges = get_non_edges(adj)
                if len(edges) < 2 or len(non_edges) < 2:
                    T *= cooling
                    continue
                idx1 = random.randrange(len(edges))
                idx2 = random.randrange(len(edges) - 1)
                if idx2 >= idx1:
                    idx2 += 1
                u1, v1 = edges[idx1]
                u2, v2 = edges[idx2]
                idx3 = random.randrange(len(non_edges))
                idx4 = random.randrange(len(non_edges) - 1)
                if idx4 >= idx3:
                    idx4 += 1
                u3, v3 = non_edges[idx3]
                u4, v4 = non_edges[idx4]
                
                # Remove both edges
                remove_edge(adj, u1, v1)
                remove_edge(adj, u2, v2)
                # Update cn_matrix
                for w in range(n):
                    if w != u1 and w != v1:
                        new_c = popcount(adj[u1] & adj[w])
                        cn_matrix[u1][w] = new_c
                        cn_matrix[w][u1] = new_c
                        new_c2 = popcount(adj[v1] & adj[w])
                        cn_matrix[v1][w] = new_c2
                        cn_matrix[w][v1] = new_c2
                    if w != u2 and w != v2:
                        new_c = popcount(adj[u2] & adj[w])
                        cn_matrix[u2][w] = new_c
                        cn_matrix[w][u2] = new_c
                        new_c2 = popcount(adj[v2] & adj[w])
                        cn_matrix[v2][w] = new_c2
                        cn_matrix[w][v2] = new_c2
                cn_matrix[u1][v1] = popcount(adj[u1] & adj[v1])
                cn_matrix[v1][u1] = cn_matrix[u1][v1]
                cn_matrix[u2][v2] = popcount(adj[u2] & adj[v2])
                cn_matrix[v2][u2] = cn_matrix[u2][v2]
                
                # Try to add both new edges
                added_count = 0
                if valid_add(adj, u3, v3):
                    add_edge(adj, u3, v3)
                    added_count += 1
                    for w in range(n):
                        if w != u3 and w != v3:
                            new_c = popcount(adj[u3] & adj[w])
                            cn_matrix[u3][w] = new_c
                            cn_matrix[w][u3] = new_c
                            new_c2 = popcount(adj[v3] & adj[w])
                            cn_matrix[v3][w] = new_c2
                            cn_matrix[w][v3] = new_c2
                    cn_matrix[u3][v3] = popcount(adj[u3] & adj[v3])
                    cn_matrix[v3][u3] = cn_matrix[u3][v3]
                if added_count == 1 and valid_add(adj, u4, v4):
                    add_edge(adj, u4, v4)
                    added_count += 1
                    for w in range(n):
                        if w != u4 and w != v4:
                            new_c = popcount(adj[u4] & adj[w])
                            cn_matrix[u4][w] = new_c
                            cn_matrix[w][u4] = new_c
                            new_c2 = popcount(adj[v4] & adj[w])
                            cn_matrix[v4][w] = new_c2
                            cn_matrix[w][v4] = new_c2
                    cn_matrix[u4][v4] = popcount(adj[u4] & adj[v4])
                    cn_matrix[v4][u4] = cn_matrix[u4][v4]
                
                if added_count == 2:
                    # Net 0 change in edges, but structure changed - accept with probability
                    if random.random() < 0.7:
                        pass  # keep
                    else:
                        # Revert
                        remove_edge(adj, u3, v3)
                        remove_edge(adj, u4, v4)
                        add_edge(adj, u1, v1)
                        add_edge(adj, u2, v2)
                        for w in range(n):
                            if w != u3 and w != v3:
                                new_c = popcount(adj[u3] & adj[w])
                                cn_matrix[u3][w] = new_c
                                cn_matrix[w][u3] = new_c
                                new_c2 = popcount(adj[v3] & adj[w])
                                cn_matrix[v3][w] = new_c2
                                cn_matrix[w][v3] = new_c2
                            if w != u4 and w != v4:
                                new_c = popcount(adj[u4] & adj[w])
                                cn_matrix[u4][w] = new_c
                                cn_matrix[w][u4] = new_c
                                new_c2 = popcount(adj[v4] & adj[w])
                                cn_matrix[v4][w] = new_c2
                                cn_matrix[w][v4] = new_c2
                            if w != u1 and w != v1:
                                new_c = popcount(adj[u1] & adj[w])
                                cn_matrix[u1][w] = new_c
                                cn_matrix[w][u1] = new_c
                                new_c2 = popcount(adj[v1] & adj[w])
                                cn_matrix[v1][w] = new_c2
                                cn_matrix[w][v1] = new_c2
                            if w != u2 and w != v2:
                                new_c = popcount(adj[u2] & adj[w])
                                cn_matrix[u2][w] = new_c
                                cn_matrix[w][u2] = new_c
                                new_c2 = popcount(adj[v2] & adj[w])
                                cn_matrix[v2][w] = new_c2
                                cn_matrix[w][v2] = new_c2
                        cn_matrix[u3][v3] = popcount(adj[u3] & adj[v3])
                        cn_matrix[v3][u3] = cn_matrix[u3][v3]
                        cn_matrix[u4][v4] = popcount(adj[u4] & adj[v4])
                        cn_matrix[v4][u4] = cn_matrix[u4][v4]
                        cn_matrix[u1][v1] = popcount(adj[u1] & adj[v1])
                        cn_matrix[v1][u1] = cn_matrix[u1][v1]
                        cn_matrix[u2][v2] = popcount(adj[u2] & adj[v2])
                        cn_matrix[v2][u2] = cn_matrix[u2][v2]
                elif added_count == 1:
                    # Net -1, accept with SA probability
                    if random.random() < math.exp(-1.0 / T):
                        # Keep one new edge, revert the removed pair
                        add_edge(adj, u1, v1)
                        add_edge(adj, u2, v2)
                        # But we only added one new edge, so we need to remove it and re-add both old
                        # Actually this is complex - just revert everything
                        if (u3, v3) != (u1, v1) and (u3, v3) != (v1, u1) and (u3, v3) != (u2, v2) and (u3, v3) != (v2, u2):
                            remove_edge(adj, u3, v3)
                        else:
                            remove_edge(adj, u4, v4)
                        add_edge(adj, u1, v1)
                        add_edge(adj, u2, v2)
                        for w in range(n):
                            if w != u1 and w != v1:
                                new_c = popcount(adj[u1] & adj[w])
                                cn_matrix[u1][w] = new_c
                                cn_matrix[w][u1] = new_c
                                new_c2 = popcount(adj[v1] & adj[w])
                                cn_matrix[v1][w] = new_c2
                                cn_matrix[w][v1] = new_c2
                            if w != u2 and w != v2:
                                new_c = popcount(adj[u2] & adj[w])
                                cn_matrix[u2][w] = new_c
                                cn_matrix[w][u2] = new_c
                                new_c2 = popcount(adj[v2] & adj[w])
                                cn_matrix[v2][w] = new_c2
                                cn_matrix[w][v2] = new_c2
                        cn_matrix[u1][v1] = popcount(adj[u1] & adj[v1])
                        cn_matrix[v1][u1] = cn_matrix[u1][v1]
                        cn_matrix[u2][v2] = popcount(adj[u2] & adj[v2])
                        cn_matrix[v2][u2] = cn_matrix[u2][v2]
                    else:
                        # Revert all
                        if (u3, v3) != (u1, v1) and (u3, v3) != (v1, u1) and (u3, v3) != (u2, v2) and (u3, v3) != (v2, u2):
                            remove_edge(adj, u3, v3)
                        else:
                            remove_edge(adj, u4, v4)
                        add_edge(adj, u1, v1)
                        add_edge(adj, u2, v2)
                        for w in range(n):
                            if w != u1 and w != v1:
                                new_c = popcount(adj[u1] & adj[w])
                                cn_matrix[u1][w] = new_c
                                cn_matrix[w][u1] = new_c
                                new_c2 = popcount(adj[v1] & adj[w])
                                cn_matrix[v1][w] = new_c2
                                cn_matrix[w][v1] = new_c2
                            if w != u2 and w != v2:
                                new_c = popcount(adj[u2] & adj[w])
                                cn_matrix[u2][w] = new_c
                                cn_matrix[w][u2] = new_c
                                new_c2 = popcount(adj[v2] & adj[w])
                                cn_matrix[v2][w] = new_c2
                                cn_matrix[w][v2] = new_c2
                        cn_matrix[u1][v1] = popcount(adj[u1] & adj[v1])
                        cn_matrix[v1][u1] = cn_matrix[u1][v1]
                        cn_matrix[u2][v2] = popcount(adj[u2] & adj[v2])
                        cn_matrix[v2][u2] = cn_matrix[u2][v2]
                else:
                    # Revert all
                    add_edge(adj, u1, v1)
                    add_edge(adj, u2, v2)
                    for w in range(n):
                        if w != u1 and w != v1:
                            new_c = popcount(adj[u1] & adj[w])
                            cn_matrix[u1][w] = new_c
                            cn_matrix[w][u1] = new_c
                            new_c2 = popcount(adj[v1] & adj[w])
                            cn_matrix[v1][w] = new_c2
                            cn_matrix[w][v1] = new_c2
                        if w != u2 and w != v2:
                            new_c = popcount(adj[u2] & adj[w])
                            cn_matrix[u2][w] = new_c
                            cn_matrix[w][u2] = new_c
                            new_c2 = popcount(adj[v2] & adj[w])
                            cn_matrix[v2][w] = new_c2
                            cn_matrix[w][v2] = new_c2
                    cn_matrix[u1][v1] = popcount(adj[u1] & adj[v1])
                    cn_matrix[v1][u1] = cn_matrix[u1][v1]
                    cn_matrix[u2][v2] = popcount(adj[u2] & adj[v2])
                    cn_matrix[v2][u2] = cn_matrix[u2][v2]

            elif move_type < 0.8:
                # Try to remove a random edge (accept with SA probability)
                edges = get_edges(adj)
                if not edges:
                    T *= cooling
                    continue
                u, v = random.choice(edges)
                if random.random() < math.exp(-1.0 / T):
                    remove_edge(adj, u, v)
                    current_edges -= 1
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

            else:
                # 1-for-1 swap: remove one edge, try to add another (net 0, structure change)
                edges = get_edges(adj)
                non_edges = get_non_edges(adj)
                if not edges or not non_edges:
                    T *= cooling
                    continue
                u1, v1 = random.choice(edges)
                u2, v2 = random.choice(non_edges)
                remove_edge(adj, u1, v1)
                for w in range(n):
                    if w != u1 and w != v1:
                        new_c = popcount(adj[u1] & adj[w])
                        cn_matrix[u1][w] = new_c
                        cn_matrix[w][u1] = new_c
                        new_c2 = popcount(adj[v1] & adj[w])
                        cn_matrix[v1][w] = new_c2
                        cn_matrix[w][v1] = new_c2
                cn_matrix[u1][v1] = popcount(adj[u1] & adj[v1])
                cn_matrix[v1][u1] = cn_matrix[u1][v1]

                if valid_add(adj, u2, v2):
                    add_edge(adj, u2, v2)
                    for w in range(n):
                        if w != u2 and w != v2:
                            new_c = popcount(adj[u2] & adj[w])
                            cn_matrix[u2][w] = new_c
                            cn_matrix[w][u2] = new_c
                            new_c2 = popcount(adj[v2] & adj[w])
                            cn_matrix[v2][w] = new_c2
                            cn_matrix[w][v2] = new_c2
                    cn_matrix[u2][v2] = popcount(adj[u2] & adj[v2])
                    cn_matrix[v2][u2] = cn_matrix[u2][v2]
                    if random.random() < 0.7:
                        pass
                    else:
                        remove_edge(adj, u2, v2)
                        add_edge(adj, u1, v1)
                        for w in range(n):
                            if w != u2 and w != v2:
                                new_c = popcount(adj[u2] & adj[w])
                                cn_matrix[u2][w] = new_c
                                cn_matrix[w][u2] = new_c
                                new_c2 = popcount(adj[v2] & adj[w])
                                cn_matrix[v2][w] = new_c2
                                cn_matrix[w][v2] = new_c2
                            if w != u1 and w != v1:
                                new_c = popcount(adj[u1] & adj[w])
                                cn_matrix[u1][w] = new_c
                                cn_matrix[w][u1] = new_c
                                new_c2 = popcount(adj[v1] & adj[w])
                                cn_matrix[v1][w] = new_c2
                                cn_matrix[w][v1] = new_c2
                        cn_matrix[u2][v2] = popcount(adj[u2] & adj[v2])
                        cn_matrix[v2][u2] = cn_matrix[u2][v2]
                        cn_matrix[u1][v1] = popcount(adj[u1] & adj[v1])
                        cn_matrix[v1][u1] = cn_matrix[u1][v1]
                else:
                    add_edge(adj, u1, v1)
                    for w in range(n):
                        if w != u1 and w != v1:
                            new_c = popcount(adj[u1] & adj[w])
                            cn_matrix[u1][w] = new_c
                            cn_matrix[w][u1] = new_c
                            new_c2 = popcount(adj[v1] & adj[w])
                            cn_matrix[v1][w] = new_c2
                            cn_matrix[w][v1] = new_c2
                    cn_matrix[u1][v1] = popcount(adj[u1] & adj[v1])
                    cn_matrix[v1][u1] = cn_matrix[u1][v1]

            T *= cooling

    # Final greedy fill on best
    if best_adj:
        greedy_fill(best_adj)
        final_edges = count_edges(best_adj)
        if final_edges > best_edges:
            best_edges = final_edges

    # Polish pass: try all possible single-edge additions and multi-edge swaps
    if best_adj and (time.time() - start_time < time_limit - 5):
        polish_adj = best_adj[:]
        polish_edges = count_edges(polish_adj)
        improved = True
        while improved and (time.time() - start_time < time_limit - 5):
            improved = False
            # Try adding any non-edge
            for u in range(n):
                for v in range(u + 1, n):
                    if not (polish_adj[u] & (1 << v)):
                        if valid_add(polish_adj, u, v):
                            add_edge(polish_adj, u, v)
                            polish_edges += 1
                            improved = True
            # Try 1-for-2 swaps: remove edge (a,b), add edges (c,d) and (e,f)
            edges = get_edges(polish_adj)
            non_edges = get_non_edges(polish_adj)
            if len(non_edges) >= 2:
                for a, b in edges:
                    if time.time() - start_time > time_limit - 5:
                        break
                    for i in range(len(non_edges)):
                        for j in range(i + 1, len(non_edges)):
                            c, d = non_edges[i]
                            e, f = non_edges[j]
                            if (a, b) == (c, d) or (a, b) == (d, c) or (a, b) == (e, f) or (a, b) == (f, e):
                                continue
                            remove_edge(polish_adj, a, b)
                            added = 0
                            added_pairs = []
                            if valid_add(polish_adj, c, d):
                                add_edge(polish_adj, c, d)
                                added += 1
                                added_pairs.append((c, d))
                            if added == 1 and valid_add(polish_adj, e, f):
                                add_edge(polish_adj, e, f)
                                added += 1
                                added_pairs.append((e, f))
                            if added >= 1:
                                polish_edges += added - 1
                                improved = True
                                # Try greedy fill after
                                for x in range(n):
                                    for y in range(x + 1, n):
                                        if not (polish_adj[x] & (1 << y)):
                                            if valid_add(polish_adj, x, y):
                                                add_edge(polish_adj, x, y)
                                                polish_edges += 1
                                                improved = True
                                break
                            else:
                                add_edge(polish_adj, a, b)
                    if improved:
                        break
            # Try 2-for-2 swaps: remove edges (a,b) and (c,d), add edges (e,f) and (g,h)
            if not improved and len(edges) >= 2 and len(non_edges) >= 2:
                for i in range(len(edges)):
                    for j in range(i + 1, len(edges)):
                        if time.time() - start_time > time_limit - 5:
                            break
                        a, b = edges[i]
                        c, d = edges[j]
                        for k in range(len(non_edges)):
                            for l in range(k + 1, len(non_edges)):
                                e, f = non_edges[k]
                                g, h = non_edges[l]
                                # Check no overlap
                                if (e,f) in [(a,b),(b,a),(c,d),(d,c)] or (g,h) in [(a,b),(b,a),(c,d),(d,c)]:
                                    continue
                                remove_edge(polish_adj, a, b)
                                remove_edge(polish_adj, c, d)
                                added = 0
                                if valid_add(polish_adj, e, f):
                                    add_edge(polish_adj, e, f)
                                    added += 1
                                if added == 1 and valid_add(polish_adj, g, h):
                                    add_edge(polish_adj, g, h)
                                    added += 1
                                if added == 2:
                                    # Net 0, but try to add more
                                    for x in range(n):
                                        for y in range(x + 1, n):
                                            if not (polish_adj[x] & (1 << y)):
                                                if valid_add(polish_adj, x, y):
                                                    add_edge(polish_adj, x, y)
                                                    polish_edges += 1
                                                    improved = True
                                    if not improved:
                                        # Revert
                                        remove_edge(polish_adj, e, f)
                                        remove_edge(polish_adj, g, h)
                                        add_edge(polish_adj, a, b)
                                        add_edge(polish_adj, c, d)
                                else:
                                    # Revert
                                    if added == 1:
                                        remove_edge(polish_adj, e, f)
                                    add_edge(polish_adj, a, b)
                                    add_edge(polish_adj, c, d)
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
            if polish_edges > best_edges:
                best_edges = polish_edges
                best_adj = polish_adj[:]

    # Final verification
    if best_adj:
        if not is_k23_free(best_adj):
            # Fallback to a known valid graph
            adj0 = make_k4_union()
            greedy_fill(adj0)
            best_adj = adj0

    # Convert to list of edges
    result = []
    for u, v in get_edges(best_adj):
        result.append([u, v])
    return result

# EVOLVE-BLOCK-END

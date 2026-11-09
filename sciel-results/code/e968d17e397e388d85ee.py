"""
K2,3-free graph construction on 20 vertices maximizing edges.

Strategy:
- Start with the 5 disjoint K4s (30 edges) as a valid baseline.
- Use bit-parallel representation for fast validity checks.
- Perform a multi-restart local search with:
    * Greedy edge addition
    * Simulated annealing with swap moves (remove one edge, add another)
    * Periodic full K2,3 validity re-checks
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

        while T > T_min and (time.time() - start_time < time_limit):
            steps += 1
            move_type = random.random()

            if move_type < 0.4:
                # Try to add a random non-edge
                non_edges = get_non_edges(adj)
                if not non_edges:
                    T *= cooling
                    continue
                u, v = random.choice(non_edges)
                if valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    current_edges += 1
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                # else: reject (no change)

            elif move_type < 0.6:
                # Try to remove a random edge (accept with SA probability)
                edges = get_edges(adj)
                if not edges:
                    T *= cooling
                    continue
                u, v = random.choice(edges)
                # Delta E = +1 (worse), accept with prob exp(-1/T)
                if random.random() < math.exp(-1.0 / T):
                    remove_edge(adj, u, v)
                    current_edges -= 1

            else:
                # Swap: remove one edge, try to add another (multiple attempts)
                edges = get_edges(adj)
                non_edges = get_non_edges(adj)
                if not edges or not non_edges:
                    T *= cooling
                    continue
                # Try up to 5 swap candidates per iteration
                for _ in range(5):
                    u1, v1 = random.choice(edges)
                    u2, v2 = random.choice(non_edges)
                    # Remove (u1,v1), check if (u2,v2) can be added
                    remove_edge(adj, u1, v1)
                    if valid_add(adj, u2, v2):
                        add_edge(adj, u2, v2)
                        # Edge count unchanged, but structure changed
                        # Accept with higher probability to explore
                        if random.random() < 0.8:
                            pass  # keep the swap
                        else:
                            # Revert
                            remove_edge(adj, u2, v2)
                            add_edge(adj, u1, v1)
                        break
                    else:
                        # Revert
                        add_edge(adj, u1, v1)

            T *= cooling

    # Final polish: exhaustive greedy + systematic swap search
    if best_adj:
        greedy_fill(best_adj)
        final_edges = count_edges(best_adj)
        if final_edges > best_edges:
            best_edges = final_edges
            best_adj = adj[:]

        # Systematic polish: try all possible single edge additions
        changed = True
        while changed:
            changed = False
            non_edges = get_non_edges(best_adj)
            for u, v in non_edges:
                if time.time() - start_time > time_limit:
                    break
                if valid_add(best_adj, u, v):
                    add_edge(best_adj, u, v)
                    final_edges += 1
                    changed = True
            if time.time() - start_time > time_limit:
                break

        # Systematic polish: try 2-edge swaps (remove one, add one) to escape local optima
        for _ in range(5):
            if time.time() - start_time > time_limit:
                break
            improved = False
            edges = get_edges(best_adj)
            non_edges = get_non_edges(best_adj)
            for u1, v1 in edges:
                if time.time() - start_time > time_limit:
                    break
                for u2, v2 in non_edges:
                    if time.time() - start_time > time_limit:
                        break
                    # Try removing (u1,v1) and adding (u2,v2)
                    remove_edge(best_adj, u1, v1)
                    if valid_add(best_adj, u2, v2):
                        add_edge(best_adj, u2, v2)
                        # Check if this leads to further improvements via greedy
                        save_adj = best_adj[:]
                        greedy_fill(save_adj)
                        save_edges = count_edges(save_adj)
                        if save_edges > final_edges:
                            best_adj = save_adj
                            final_edges = save_edges
                            improved = True
                            break
                        else:
                            # Revert
                            remove_edge(best_adj, u2, v2)
                            add_edge(best_adj, u1, v1)
                    else:
                        # Revert
                        add_edge(best_adj, u1, v1)
                if improved:
                    break
            if not improved:
                break

        if final_edges > best_edges:
            best_edges = final_edges

    # Convert to list of edges
    result = []
    for u, v in get_edges(best_adj):
        result.append([u, v])
    return result

# EVOLVE-BLOCK-END

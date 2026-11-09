"""K3,3-free dense graph on 20 vertices via pair-exchange simulated annealing."""
import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 140
    n = 20

    def count_edges(adj_bits):
        return sum(b.bit_count() for b in adj_bits) // 2

    def add_edge(adj_bits, u, v):
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)

    def remove_edge(adj_bits, u, v):
        adj_bits[u] &= ~(1 << v)
        adj_bits[v] &= ~(1 << u)

    def is_k33_free(adj_bits):
        for i in range(n):
            for j in range(i + 1, n):
                ij = adj_bits[i] & adj_bits[j]
                if ij.bit_count() < 3:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if (ij & adj_bits[k]).bit_count() >= 3:
                        return False
        return True

    def violation_count(adj_bits):
        """Count number of K3,3 subgraphs (as a proxy for energy)."""
        violations = 0
        for i in range(n):
            for j in range(i + 1, n):
                ij = adj_bits[i] & adj_bits[j]
                if ij.bit_count() < 3:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if (ij & adj_bits[k]).bit_count() >= 3:
                        violations += 1
        return violations

    def get_missing(adj_bits):
        return [(i, j) for i in range(n) for j in range(i + 1, n)
                if not (adj_bits[i] >> j) & 1]

    def get_existing(adj_bits):
        return [(i, j) for i in range(n) for j in range(i + 1, n)
                if (adj_bits[i] >> j) & 1]

    def init_dense_random():
        """Start with a dense random graph and prune to K3,3-free."""
        adj = [0] * n
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            add_edge(adj, u, v)
            if not is_k33_free(adj):
                remove_edge(adj, u, v)
        return adj

    def init_greedy_random():
        """Greedy random construction."""
        adj = [0] * n
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        for u, v in all_edges:
            add_edge(adj, u, v)
            if not is_k33_free(adj):
                remove_edge(adj, u, v)
        return adj

    def simulated_annealing(adj_bits, deadline):
        """SA with pair-exchange moves: swap existing edge with missing edge, or add/remove."""
        current = adj_bits[:]
        current_edges = count_edges(current)
        current_violations = violation_count(current)

        T_start = 10.0
        T_end = 0.01
        total_steps = 50000
        best_adj = current[:]
        best_edges = current_edges
        best_violations = current_violations

        existing = get_existing(current)
        missing = get_missing(current)

        for step in range(total_steps):
            if time.time() > deadline - 5:
                break

            # Temperature schedule
            progress = step / total_steps
            T = T_start * (T_end / T_start) ** progress

            # Choose move type
            move_type = random.random()
            if move_type < 0.6:
                # Pair exchange: remove one existing, add one missing
                if not existing or not missing:
                    continue
                idx1 = random.randint(0, len(existing) - 1)
                idx2 = random.randint(0, len(missing) - 1)
                e_remove = existing[idx1]
                e_add = missing[idx2]

                u1, v1 = e_remove
                u2, v2 = e_add

                remove_edge(current, u1, v1)
                add_edge(current, u2, v2)

                new_violations = violation_count(current)
                delta_v = new_violations - current_violations
                # Edge count stays same in pair exchange

                if new_violations < current_violations or (T > 0 and random.random() < np.exp(-delta_v / T)):
                    # Accept
                    current_violations = new_violations
                    existing[idx1] = e_add
                    missing[idx2] = e_remove
                else:
                    remove_edge(current, u2, v2)
                    add_edge(current, u1, v1)

            elif move_type < 0.8:
                # Add an edge
                if not missing:
                    continue
                idx = random.randint(0, len(missing) - 1)
                u, v = missing[idx]
                add_edge(current, u, v)
                new_violations = violation_count(current)
                delta_v = new_violations - current_violations
                delta_e = 1  # edge count increases by 1

                if new_violations < current_violations or (T > 0 and random.random() < np.exp(-(delta_v - delta_e * 2) / T)):
                    current_violations = new_violations
                    current_edges += 1
                    missing.pop(idx)
                    existing.append((u, v))
                    if current_edges > best_edges and current_violations == 0:
                        best_edges = current_edges
                        best_adj = current[:]
                        best_violations = 0
                else:
                    remove_edge(current, u, v)

            else:
                # Remove an edge (to escape local minima)
                if not existing:
                    continue
                idx = random.randint(0, len(existing) - 1)
                u, v = existing[idx]
                remove_edge(current, u, v)
                new_violations = violation_count(current)
                delta_v = new_violations - current_violations
                delta_e = -1

                if new_violations < current_violations or (T > 0 and random.random() < np.exp(-(delta_v + delta_e * 2) / T)):
                    current_violations = new_violations
                    current_edges -= 1
                    existing.pop(idx)
                    missing.append((u, v))
                else:
                    add_edge(current, u, v)

            # Track best K3,3-free solution
            if current_violations == 0 and current_edges > best_edges:
                best_edges = current_edges
                best_adj = current[:]
                best_violations = 0

        return best_adj, best_edges

    def local_polish(adj_bits, deadline):
        """Final greedy polish: try to add any missing edge that keeps K3,3-free."""
        current = adj_bits[:]
        current_edges = count_edges(current)
        missing = get_missing(current)
        random.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline - 3:
                break
            add_edge(current, u, v)
            if is_k33_free(current):
                current_edges += 1
            else:
                remove_edge(current, u, v)
        return current, current_edges

    best_adj = None
    best_edges = -1

    # Multiple restarts
    for restart in range(4):
        if time.time() > deadline - 15:
            break

        if restart % 2 == 0:
            adj = init_dense_random()
        else:
            adj = init_greedy_random()

        adj, edges = simulated_annealing(adj, deadline - 10)
        adj, edges = local_polish(adj, deadline - 5)

        if edges > best_edges:
            best_edges = edges
            best_adj = adj[:]

    # Final polish
    if best_adj is not None and time.time() < deadline - 2:
        best_adj, best_edges = local_polish(best_adj, deadline - 2)

    # Convert to list of edges
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                result.append([i, j])

    return result

# EVOLVE-BLOCK-END

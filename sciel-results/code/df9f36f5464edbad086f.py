import time
import random
import math
import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore")

# EVOLVE-BLOCK-START

def construct():
    """
    Maximize edges in a K2,3-free graph on 20 vertices.
    Uses vectorized common-neighbor counting and multi-phase search.
    """
    n = 20
    start_time = time.time()
    time_limit = 135.0  # Safety margin before 150s kill
    deadline = start_time + time_limit

    # Precompute all unordered pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    all_pairs_set = set(all_pairs)

    def count_edges(adj):
        return sum(adj[u] >> v & 1 for u, v in all_pairs)

    def is_edge(adj, u, v):
        return adj[u] >> v & 1

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_edges_from_adj(adj):
        edges = []
        for u, v in all_pairs:
            if adj[u] >> v & 1:
                edges.append([u, v])
        return edges

    def common_neighbors_count(adj, i, j):
        """Count common neighbors of i and j using bit operations."""
        common = adj[i] & adj[j]
        # Use a fast popcount
        return _popcount(common)

    # Fast popcount for 20-bit integers
    def _popcount(x):
        # Python 3.10+ has int.bit_count()
        if hasattr(int, 'bit_count'):
            return x.bit_count()
        # Fallback
        count = 0
        while x:
            count += 1
            x &= x - 1
        return count

    def can_add_edge_fast(adj, u, v):
        """
        Check if adding edge (u,v) keeps the graph K2,3-free.
        Conditions:
        1. common_neighbors(u,v) <= 2
        2. For each w in N(u) \\ {v}: common_neighbors(w,v) <= 1
        3. For each w in N(v) \\ {u}: common_neighbors(w,u) <= 1
        """
        if is_edge(adj, u, v):
            return True

        # Condition 1
        if _popcount(adj[u] & adj[v]) > 2:
            return False

        # Condition 2: for w in N(u), w != v
        temp = adj[u] & ~(1 << v)
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp ^= lsb
            if _popcount(adj[w] & adj[v]) > 1:
                return False

        # Condition 3: for w in N(v), w != u
        temp = adj[v] & ~(1 << u)
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp ^= lsb
            if _popcount(adj[w] & adj[u]) > 1:
                return False

        return True

    def greedy_fill(adj, order=None, dl=None):
        if order is None:
            order = all_pairs
        for u, v in order:
            if dl is not None and time.time() > dl:
                break
            if not is_edge(adj, u, v):
                if can_add_edge_fast(adj, u, v):
                    add_edge(adj, u, v)

    def local_search_swap(adj, max_iters, dl):
        """
        Local search: try to add a non-edge; if blocked, try removing
        one incident edge and re-adding. Also try 1-for-1 edge swaps
        (remove edge a, add edge b) where a is random and b is a non-edge.
        """
        current_edges = count_edges(adj)
        best_edges = current_edges
        best_adj = adj[:]

        for it in range(max_iters):
            if time.time() > dl:
                break

            # Pick a random non-edge to try to add
            for _ in range(20):  # retry a few times to find a non-edge
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v and not is_edge(adj, u, v):
                    break
            else:
                continue

            if can_add_edge_fast(adj, u, v):
                add_edge(adj, u, v)
                current_edges += 1
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
            else:
                # Try removing a random edge to unblock
                # Collect all edges
                edge_list = []
                for eu, ev in all_pairs:
                    if is_edge(adj, eu, ev):
                        edge_list.append((eu, ev))

                if edge_list:
                    rem_u, rem_v = random.choice(edge_list)
                    remove_edge(adj, rem_u, rem_v)
                    current_edges -= 1
                    if can_add_edge_fast(adj, u, v):
                        add_edge(adj, u, v)
                        current_edges += 1
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = adj[:]
                    else:
                        # Revert
                        add_edge(adj, rem_u, rem_v)
                        current_edges += 1

        return best_adj, best_edges

    def simulated_annealing(adj, max_iters, dl):
        """
        SA-style search: accept worsening moves with probability exp(-delta/T).
        """
        current_edges = count_edges(adj)
        best_edges = current_edges
        best_adj = adj[:]
        T_start = 2.0
        T_end = 0.01
        cooling = (T_end / T_start) ** (1.0 / max(1, max_iters))

        T = T_start
        for it in range(max_iters):
            if time.time() > dl:
                break

            # Pick a random non-edge
            for _ in range(20):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v and not is_edge(adj, u, v):
                    break
            else:
                continue

            if can_add_edge_fast(adj, u, v):
                add_edge(adj, u, v)
                current_edges += 1
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
            else:
                # Try a swap: remove random edge, add (u,v)
                edge_list = []
                for eu, ev in all_pairs:
                    if is_edge(adj, eu, ev):
                        edge_list.append((eu, ev))
                if not edge_list:
                    continue
                rem_u, rem_v = random.choice(edge_list)
                remove_edge(adj, rem_u, rem_v)
                current_edges -= 1
                if can_add_edge_fast(adj, u, v):
                    add_edge(adj, u, v)
                    current_edges += 1
                    delta = 0  # neutral swap
                else:
                    # Can't add, revert
                    add_edge(adj, rem_u, rem_v)
                    current_edges += 1
                    continue

                # Accept if better or with probability
                if current_edges >= best_edges or random.random() < math.exp(-1.0 / max(T, 1e-9)):
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_adj = adj[:]
                else:
                    # Revert
                    remove_edge(adj, u, v)
                    add_edge(adj, rem_u, rem_v)
                    current_edges -= 1

            T *= cooling

        return best_adj, best_edges

    # --- Main search loop ---
    best_adj = None
    best_edges = 0

    # Phase 1: Many greedy restarts with different random orderings
    num_greedy = 500
    for r in range(num_greedy):
        if time.time() > deadline - 30:
            break
        adj = [0] * n
        order = all_pairs[:]
        random.shuffle(order)
        greedy_fill(adj, order=order, dl=deadline - 30)
        ce = count_edges(adj)
        if ce > best_edges:
            best_edges = ce
            best_adj = adj[:]

    # Phase 2: Local search on best
    if best_adj is not None and time.time() < deadline - 15:
        improved_adj, improved_edges = local_search_swap(best_adj[:], 30000, deadline - 15)
        if improved_edges > best_edges:
            best_edges = improved_edges
            best_adj = improved_adj

    # Phase 3: Simulated annealing on best
    if best_adj is not None and time.time() < deadline - 5:
        sa_adj, sa_edges = simulated_annealing(best_adj[:], 50000, deadline - 5)
        if sa_edges > best_edges:
            best_edges = sa_edges
            best_adj = sa_adj

    # Phase 4: Final greedy pass
    if best_adj is not None and time.time() < deadline:
        greedy_fill(best_adj, dl=deadline)
        fe = count_edges(best_adj)
        if fe > best_edges:
            best_edges = fe

    # Phase 5: More greedy restarts if time remains
    if time.time() < deadline - 10:
        for r in range(200):
            if time.time() > deadline - 10:
                break
            adj = [0] * n
            order = all_pairs[:]
            random.shuffle(order)
            greedy_fill(adj, order=order, dl=deadline - 10)
            ce = count_edges(adj)
            if ce > best_edges:
                best_edges = ce
                best_adj = adj[:]
        # Quick local search on new best
        if best_adj is not None and time.time() < deadline - 3:
            improved_adj, improved_edges = local_search_swap(best_adj[:], 10000, deadline - 3)
            if improved_edges > best_edges:
                best_edges = improved_edges
                best_adj = improved_adj

    # Final greedy
    if best_adj is not None and time.time() < deadline:
        greedy_fill(best_adj, dl=deadline)

    if best_adj is None:
        # Fallback: 5 disjoint K4s
        best_adj = [0] * n
        for k in range(5):
            base = 4 * k
            for a in range(4):
                for b in range(a + 1, 4):
                    add_edge(best_adj, base + a, base + b)

    return get_edges_from_adj(best_adj)

# EVOLVE-BLOCK-END

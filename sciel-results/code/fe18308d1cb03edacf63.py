import time
import random

def construct():
    deadline = time.time() + 115
    n = 40
    masks = [1 << i for i in range(n)]
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_to_pair = {(u, v): (u, v) for u, v in all_edges}

    def has_c6_with_edge(u, v, adj):
        """Check if adding (u,v) creates a C6 in graph with adjacency list adj."""
        # Path: u - a - b - c - d - v
        # a in N(u), b in N(a)\{u,v}, c in N(b)\{u,v,a}, d in N(c)∩N(v) \ {u,a,b,c}
        a_mask = adj[u] & ~masks[v]
        while a_mask:
            a_bit = a_mask & -a_mask
            a = a_bit.bit_length() - 1
            a_mask ^= a_bit

            b_mask = adj[a] & ~masks[u] & ~masks[v]
            while b_mask:
                b_bit = b_mask & -b_mask
                b = b_bit.bit_length() - 1
                b_mask ^= b_bit

                c_mask = adj[b] & ~masks[u] & ~masks[a] & ~masks[v]
                while c_mask:
                    c_bit = c_mask & -c_mask
                    c = c_bit.bit_length() - 1
                    c_mask ^= c_bit

                    # d must be in N(c) ∩ N(v), and not in {u, a, b, c}
                    d_mask = adj[c] & adj[v]
                    forbidden = masks[u] | masks[a] | masks[b] | masks[c]
                    if d_mask & ~forbidden:
                        return True
        return False

    def add_edge(u, v, adj):
        adj[u] |= masks[v]
        adj[v] |= masks[u]

    def remove_edge(u, v, adj):
        adj[u] &= ~masks[v]
        adj[v] &= ~masks[u]

    def count_edges(adj):
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    def greedy_fill(adj, edge_order=None):
        """Try to add all possible edges in the given order."""
        if edge_order is None:
            edge_order = all_edges[:]
            random.shuffle(edge_order)
        for u, v in edge_order:
            if (adj[u] & masks[v]) == 0:
                if not has_c6_with_edge(u, v, adj):
                    add_edge(u, v, adj)

    def exhaustive_fill(adj):
        """Repeatedly try to add any addable edge until none remain."""
        changed = True
        while changed:
            changed = False
            for u, v in all_edges:
                if (adj[u] & masks[v]) == 0:
                    if not has_c6_with_edge(u, v, adj):
                        add_edge(u, v, adj)
                        changed = True

    # Multi-start: try different initial structures
    best_adj = [0] * n
    best_edges = 0

    start_time = time.time()

    # Strategy 1: Multi-start greedy with random orderings
    while time.time() < deadline - 30:
        adj = [0] * n
        # Random sparse initialization
        for _ in range(random.randint(10, 30)):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and (adj[u] & masks[v]) == 0:
                if not has_c6_with_edge(u, v, adj):
                    add_edge(u, v, adj)
        greedy_fill(adj)
        cnt = count_edges(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = adj[:]

    # Strategy 2: Local search on the best solution found
    adj = best_adj[:]
    current_edges = best_edges

    # Simulated annealing-style local search
    temp = 5.0
    cooling = 0.995
    no_improve_count = 0

    while time.time() < deadline - 10:
        improved = False

        # Try random edge additions
        for _ in range(30):
            if time.time() > deadline - 10:
                break
            x = random.randint(0, n - 1)
            y = random.randint(0, n - 1)
            if x == y:
                continue
            if (adj[x] & masks[y]) != 0:
                continue

            if not has_c6_with_edge(x, y, adj):
                add_edge(x, y, adj)
                current_edges += 1
                improved = True
                greedy_fill(adj)
                new_count = count_edges(adj)
                if new_count > best_edges:
                    best_edges = new_count
                    best_adj = adj[:]
                current_edges = new_count
                no_improve_count = 0
                break

        if not improved:
            # Perturbation: remove 1-3 random edges, then re-fill
            n_remove = random.randint(1, 3)
            removed = []
            for _ in range(n_remove):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v and (adj[u] & masks[v]):
                    remove_edge(u, v, adj)
                    removed.append((u, v))
                    current_edges -= 1

            if removed:
                greedy_fill(adj)
                new_count = count_edges(adj)
                if new_count > best_edges:
                    best_edges = new_count
                    best_adj = adj[:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                current_edges = new_count

            temp *= cooling
            if no_improve_count > 50:
                no_improve_count = 0
                temp = 5.0  # Reset temperature

    # Final exhaustive fill on the best solution
    adj = best_adj[:]
    exhaustive_fill(adj)
    final_count = count_edges(adj)
    if final_count > best_edges:
        best_edges = final_count
        best_adj = adj[:]

    # Build edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] & masks[j]):
                edges.append([i, j])
    return edges

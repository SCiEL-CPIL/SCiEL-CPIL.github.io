import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    n = 16
    target_edges = 65

    # Helper to calculate score (max triangle degree sum)
    def calc_score(adj, degs):
        max_sum = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    while common:
                        lsb = common & -common
                        w = lsb.bit_length() - 1
                        s = degs[u] + degs[v] + degs[w]
                        if s > max_sum:
                            max_sum = s
                        common ^= lsb
        return max_sum

    # QUBO objective: sum of squared triangle degree sums (smoother for SA)
    def calc_qubo(adj, degs):
        total = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    while common:
                        lsb = common & -common
                        w = lsb.bit_length() - 1
                        s = degs[u] + degs[v] + degs[w]
                        total += s * s
                        common ^= lsb
        return total

    def generate_random_graph():
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_edges)
        edges = all_edges[:target_edges]

        adj = [0] * n
        degs = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            degs[u] += 1
            degs[v] += 1
        return adj, degs, edges

    def get_edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i] & (1 << j):
                    edges.append([i, j])
        return edges

    best_score = float('inf')
    best_adj = None
    best_degs = None
    best_edges = None

    # Precompute all possible edges and non-edges for faster sampling
    all_edge_list = [(i, j) for i in range(n) for j in range(i + 1, n)]

    for restart in range(200):
        if time.time() > deadline:
            break

        adj, degs, edges = generate_random_graph()
        current_score = calc_score(adj, degs)
        current_qubo = calc_qubo(adj, degs)

        if current_score < best_score:
            best_score = current_score
            best_adj = adj[:]
            best_degs = degs[:]
            best_edges = get_edges_from_adj(adj)

        # Simulated Annealing phase
        temp = 5.0  # Initial temperature
        cooling = 0.995
        no_improve_count = 0
        max_no_improve = 800

        while time.time() < deadline and no_improve_count < max_no_improve:
            # Pick random edge to remove
            while True:
                u1 = random.randint(0, n - 1)
                v1 = random.randint(0, n - 1)
                if u1 < v1 and (adj[u1] & (1 << v1)):
                    break

            # Pick random non-edge to add
            while True:
                u2 = random.randint(0, n - 1)
                v2 = random.randint(0, n - 1)
                if u2 < v2 and not (adj[u2] & (1 << v2)):
                    break

            # Apply swap
            adj[u1] &= ~(1 << v1)
            adj[v1] &= ~(1 << u1)
            degs[u1] -= 1
            degs[v1] -= 1

            adj[u2] |= (1 << v2)
            adj[v2] |= (1 << u2)
            degs[u2] += 1
            degs[v2] += 1

            new_score = calc_score(adj, degs)
            new_qubo = calc_qubo(adj, degs)

            # Acceptance criterion: use QUBO difference for smoother signal
            delta_qubo = new_qubo - current_qubo
            if delta_qubo <= 0:
                # Always accept improvement
                current_score = new_score
                current_qubo = new_qubo
                no_improve_count = 0
                if current_score < best_score:
                    best_score = current_score
                    best_adj = adj[:]
                    best_degs = degs[:]
                    best_edges = get_edges_from_adj(adj)
            else:
                # Accept worse moves with probability exp(-delta/temp)
                if random.random() < np.exp(-delta_qubo / (temp + 1e-10)):
                    current_score = new_score
                    current_qubo = new_qubo
                    no_improve_count = 0
                    if current_score < best_score:
                        best_score = current_score
                        best_adj = adj[:]
                        best_degs = degs[:]
                        best_edges = get_edges_from_adj(adj)
                else:
                    # Revert
                    adj[u1] |= (1 << v1)
                    adj[v1] |= (1 << u1)
                    degs[u1] += 1
                    degs[v1] += 1

                    adj[u2] &= ~(1 << v2)
                    adj[v2] &= ~(1 << u2)
                    degs[u2] -= 1
                    degs[v2] -= 1

                    no_improve_count += 1

            temp *= cooling

    # Final polish: strict local search on best solution
    if best_adj is not None and time.time() < deadline:
        adj = best_adj[:]
        degs = best_degs[:]
        current_score = calc_score(adj, degs)
        current_qubo = calc_qubo(adj, degs)

        # Strict hill climbing with systematic neighborhood
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for u1 in range(n):
                if time.time() > deadline:
                    break
                for v1 in range(u1 + 1, n):
                    if time.time() > deadline:
                        break
                    if not (adj[u1] & (1 << v1)):
                        continue
                    # Try removing edge (u1, v1)
                    for u2 in range(n):
                        if time.time() > deadline:
                            break
                        for v2 in range(u2 + 1, n):
                            if time.time() > deadline:
                                break
                            if (adj[u2] & (1 << v2)):
                                continue
                            if u2 == u1 and v2 == v1:
                                continue

                            # Apply swap
                            adj[u1] &= ~(1 << v1)
                            adj[v1] &= ~(1 << u1)
                            degs[u1] -= 1
                            degs[v1] -= 1

                            adj[u2] |= (1 << v2)
                            adj[v2] |= (1 << u2)
                            degs[u2] += 1
                            degs[v2] += 1

                            new_score = calc_score(adj, degs)

                            if new_score < current_score:
                                current_score = new_score
                                current_qubo = calc_qubo(adj, degs)
                                improved = True
                                if current_score < best_score:
                                    best_score = current_score
                                    best_adj = adj[:]
                                    best_degs = degs[:]
                                    best_edges = get_edges_from_adj(adj)
                            else:
                                # Revert
                                adj[u1] |= (1 << v1)
                                adj[v1] |= (1 << u1)
                                degs[u1] += 1
                                degs[v1] += 1

                                adj[u2] &= ~(1 << v2)
                                adj[v2] &= ~(1 << u2)
                                degs[u2] -= 1
                                degs[v2] -= 1

    if best_edges is None:
        edges = [[u, v] for u in range(8) for v in range(8, 16)] + [[0, 1]]
        return {'n': 16, 'edges': edges}

    return {'n': 16, 'edges': best_edges}

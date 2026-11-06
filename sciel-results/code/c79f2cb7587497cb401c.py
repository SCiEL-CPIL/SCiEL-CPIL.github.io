import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n = 31
    target_edges = 91
    q = 5

    # 1. Construct the Polarity Graph of PG(2,5)
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    for x in range(q):
        points.append((x, 1, 0))
    points.append((1, 0, 0))

    assert len(points) == 31

    # Define adjacency based on the polarity B(P, Q) = x1*x2 + y1*y2 + z1*z2 = 0 mod 5
    adj = np.zeros((n, n), dtype=np.uint8)
    edges = []

    for i in range(n):
        for j in range(i + 1, n):
            p = points[i]
            q_pt = points[j]
            val = (p[0]*q_pt[0] + p[1]*q_pt[1] + p[2]*q_pt[2]) % q
            if val == 0:
                edges.append((i, j))
                adj[i, j] = 1
                adj[j, i] = 1

    # The polarity graph of PG(2,5) is C4-free and has 90 edges.
    if len(edges) != 90:
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(all_pairs)
        return [list(e) for e in all_pairs[:91]]

    # 2. Add one edge to reach 91 edges.
    # Adding edge (u, v) creates C4s = (A^3)[u, v] where A is adjacency matrix
    A = adj.copy().astype(np.int32)
    A2 = np.dot(A, A)
    A3 = np.dot(A2, A)

    best_edge = None
    min_c4s = float('inf')

    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0:
                c4s_created = A3[i, j]
                if c4s_created < min_c4s:
                    min_c4s = c4s_created
                    best_edge = (i, j)

    if best_edge:
        u, v = best_edge
        edges.append(best_edge)
        adj[u, v] = 1
        adj[v, u] = 1

    # 3. Fast C4 count: # C4s = (1/2) * sum_{u<v} C(common_neighbors(u,v), 2)
    # where common_neighbors(u,v) = (A^2)[u,v]
    def count_c4(adj_mat):
        A_int = adj_mat.astype(np.int32)
        A2 = np.dot(A_int, A_int)
        # Upper triangle values
        upper = np.triu(A2, k=1)
        c_vals = upper.astype(np.int64)
        c_choose_2 = c_vals * (c_vals - 1) // 2
        total = np.sum(c_choose_2)
        return total // 2

    current_score = count_c4(adj)
    best_score = current_score
    best_adj = adj.copy()
    best_edges = edges.copy()

    # 4. Local search: greedy + simulated annealing with edge swaps
    # Precompute edge and non-edge lists
    def get_edge_nonedge_lists(adj_mat):
        e_list = []
        ne_list = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj_mat[i, j]:
                    e_list.append((i, j))
                else:
                    ne_list.append((i, j))
        return e_list, ne_list

    e_list, ne_list = get_edge_nonedge_lists(adj)

    # Phase 1: Greedy local search - try to find improving swaps
    greedy_rounds = 0
    max_greedy_rounds = 50
    while greedy_rounds < max_greedy_rounds and time.time() < deadline - 15:
        improved = False
        # Try all pairs of (edge to remove, non-edge to add) - but that's O(E*NE) ~ 91*385 ~ 35k
        # Each evaluation is O(n^2) for C4 count, so ~35k * 1000 = 35M ops, might be slow
        # Instead, sample a subset
        num_trials = min(len(e_list) * len(ne_list), 2000)
        for _ in range(num_trials):
            if time.time() > deadline - 15:
                break
            e_idx = random.randint(0, len(e_list) - 1)
            u, v = e_list[e_idx]
            ne_idx = random.randint(0, len(ne_list) - 1)
            a, b = ne_list[ne_idx]

            if (a == u and b == v) or (a == v and b == u):
                continue

            adj[u, v] = 0
            adj[v, u] = 0
            adj[a, b] = 1
            adj[b, a] = 1

            new_score = count_c4(adj)

            if new_score < current_score:
                current_score = new_score
                improved = True
                e_list[e_idx] = (a, b)
                ne_list[ne_idx] = (u, v)
                if current_score < best_score:
                    best_score = current_score
                    best_adj = adj.copy()
                    best_edges = get_edge_nonedge_lists(adj)[0]
            else:
                adj[u, v] = 1
                adj[v, u] = 1
                adj[a, b] = 0
                adj[b, a] = 0

        greedy_rounds += 1
        if not improved:
            break

    # Phase 2: Simulated annealing to escape local optima
    if time.time() < deadline - 5:
        temp = 2.0
        cooling_rate = 0.999
        min_temp = 0.1
        while time.time() < deadline - 5 and temp > min_temp:
            e_idx = random.randint(0, len(e_list) - 1)
            u, v = e_list[e_idx]
            ne_idx = random.randint(0, len(ne_list) - 1)
            a, b = ne_list[ne_idx]

            if (a == u and b == v) or (a == v and b == u):
                temp *= cooling_rate
                continue

            adj[u, v] = 0
            adj[v, u] = 0
            adj[a, b] = 1
            adj[b, a] = 1

            new_score = count_c4(adj)
            delta = new_score - current_score

            if delta < 0 or random.random() < np.exp(-delta / temp):
                current_score = new_score
                e_list[e_idx] = (a, b)
                ne_list[ne_idx] = (u, v)
                if current_score < best_score:
                    best_score = current_score
                    best_adj = adj.copy()
                    best_edges = get_edge_nonedge_lists(adj)[0]
            else:
                adj[u, v] = 1
                adj[v, u] = 1
                adj[a, b] = 0
                adj[b, a] = 0

            temp *= cooling_rate

    return [list(e) for e in best_edges]

if __name__ == "__main__":
    edges = construct()
    print(f"Generated {len(edges)} edges")
    n = 31
    adj = np.zeros((n, n), dtype=int)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
    A2 = np.dot(adj, adj)
    np.fill_diagonal(A2, 0)
    prod = A2 * (A2 - 1)
    sum_upper = np.sum(np.triu(prod, 1))
    c4s = sum_upper // 4
    print(f"C4 count: {c4s}")

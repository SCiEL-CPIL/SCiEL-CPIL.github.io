"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]
    if n == 0:
        return np.array([], dtype=int)

    start_time = time.time()
    DEADLINE = start_time + 140.0

    def time_left():
        return DEADLINE - time.time()

    E = int(np.sum(adj)) // 2
    degrees = adj.sum(axis=1)

    def cut_size_from_labels(labels):
        x = 2 * labels.astype(np.float64) - 1
        xTadjx = x @ (adj @ x)
        return int(E - 0.5 * xTadjx)

    def local_search(labels, max_rounds=50):
        """Hill-climbing with incremental gain updates."""
        labels = labels.copy()
        best_labels = labels.copy()
        x = 2 * labels.astype(np.float64) - 1
        adj_x = adj @ x  # adj_x[v] = sum_u adj[v,u] * x[u]
        gains = x * adj_x  # gains[v] = x[v] * adj_x[v]
        current_cut = cut_size_from_labels(labels)
        best_cut = current_cut

        for rnd in range(max_rounds):
            if time_left() < 2.0:
                break
            improved = False
            # Find all vertices with positive gain, process in descending order
            pos_mask = gains > 0
            if not np.any(pos_mask):
                break
            pos_indices = np.where(pos_mask)[0]
            order = pos_indices[np.argsort(-gains[pos_indices])]

            for v in order:
                if time_left() < 1.0:
                    break
                if gains[v] <= 0:
                    continue
                # Flip v
                old_xv = x[v]
                x[v] = -x[v]
                labels[v] = 1 - labels[v]
                current_cut += int(gains[v])
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
                improved = True

                # Update adj_x and gains for all neighbors of v
                # adj_x[u] changes by adj[u,v] * (new_x[v] - old_x[v]) = adj[u,v] * (-2*old_xv)
                # Since adj is symmetric, adj[u,v] = adj[v,u]
                neighbors = np.where(adj[v] == 1)[0]
                if len(neighbors) > 0:
                    delta_adj_x = -2.0 * old_xv * adj[v, neighbors]
                    adj_x[neighbors] += delta_adj_x
                    # gains[u] = x[u] * adj_x[u] for u != v
                    gains[neighbors] = x[neighbors] * adj_x[neighbors]
                # Update gain for v itself
                gains[v] = x[v] * adj_x[v]

            if not improved:
                break

        final_cut = cut_size_from_labels(labels)
        if final_cut > best_cut:
            best_labels = labels.copy()
            best_cut = final_cut
        return best_labels, best_cut

    def goe_heuristic(labels):
        """Greedy Ordering Elimination heuristic."""
        result = labels.copy()
        assigned = np.zeros(n, dtype=bool)
        remaining = np.arange(n)

        for _ in range(n):
            if time_left() < 2.0:
                break
            if len(remaining) == 0:
                break
            # Find vertex with highest degree among unassigned
            best_v = remaining[0]
            best_deg = -1
            for v in remaining:
                if not assigned[v] and degrees[v] > best_deg:
                    best_deg = degrees[v]
                    best_v = v

            if best_v == -1:
                break

            # Assign best_v to minimize same-side neighbors
            neighbors = np.where(adj[best_v] == 1)[0]
            unassigned_neighbors = neighbors[~assigned[neighbors]]
            if len(unassigned_neighbors) > 0:
                side0_count = np.sum(result[unassigned_neighbors] == 0)
                side1_count = np.sum(result[unassigned_neighbors] == 1)
                if side1_count >= side0_count:
                    result[best_v] = 0
                else:
                    result[best_v] = 1
            else:
                result[best_v] = 0

            assigned[best_v] = True
            remaining = remaining[remaining != best_v]

        return result

    def simulated_annealing(labels, max_iters=500000, T_start=8.0, T_end=0.01, alpha=0.99997):
        """Fast SA with incremental adj_x updates."""
        labels = labels.copy()
        best_labels = labels.copy()
        x = 2 * labels.astype(np.float64) - 1
        adj_x = adj @ x
        current_cut = cut_size_from_labels(labels)
        best_cut = current_cut
        T = T_start

        for it in range(max_iters):
            if time_left() < 3.0:
                break

            v = np.random.randint(0, n)
            # delta = gain from flipping v = x[v] * adj_x[v]
            delta = int(x[v] * adj_x[v])

            if delta >= 0:
                # Always accept
                old_xv = x[v]
                x[v] = -x[v]
                labels[v] = 1 - labels[v]
                current_cut += delta
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
                # Update adj_x for neighbors
                neighbors = np.where(adj[v] == 1)[0]
                if len(neighbors) > 0:
                    adj_x[neighbors] += -2.0 * old_xv * adj[v, neighbors]
            else:
                # Accept with probability exp(delta / T)
                if T > 0.001 and delta > -50 * T:
                    if np.random.random() < np.exp(delta / T):
                        old_xv = x[v]
                        x[v] = -x[v]
                        labels[v] = 1 - labels[v]
                        current_cut += delta
                        # Update adj_x for neighbors
                        neighbors = np.where(adj[v] == 1)[0]
                        if len(neighbors) > 0:
                            adj_x[neighbors] += -2.0 * old_xv * adj[v, neighbors]

            T = max(T * alpha, T_end)

        return best_labels, best_cut

    def multi_start_local(num_starts):
        """Try multiple random starts with local search."""
        best_labels = None
        best_cut = -1
        for s in range(num_starts):
            if time_left() < 3.0:
                break
            labels = np.random.randint(0, 2, size=n)
            labels, cut = local_search(labels, max_rounds=30)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
        return best_labels, best_cut

    # Strategy:
    # 1. Start with alternating pattern, run local search
    # 2. Try GOE heuristic + local search
    # 3. Multiple random restarts with local search
    # 4. Simulated annealing from best and from random
    # 5. Final local search polish

    # Step 1: Alternating start
    labels_alt = np.arange(n) % 2
    best_labels, best_cut = local_search(labels_alt, max_rounds=50)

    # Step 2: GOE heuristic
    labels_goe = goe_heuristic(np.zeros(n, dtype=int))
    labels_goe_ls, cut_goe = local_search(labels_goe, max_rounds=50)
    if cut_goe > best_cut:
        best_cut = cut_goe
        best_labels = labels_goe_ls

    # Step 3: Random restarts with local search
    if time_left() > 10.0:
        labels_rs, cut_rs = multi_start_local(20)
        if cut_rs > best_cut:
            best_cut = cut_rs
            best_labels = labels_rs

    # Step 4: Simulated annealing
    if time_left() > 15.0:
        labels_sa, cut_sa = simulated_annealing(best_labels, max_iters=300000, T_start=8.0, T_end=0.01)
        if cut_sa > best_cut:
            best_cut = cut_sa
            best_labels = labels_sa

    if time_left() > 10.0:
        labels_rand = np.random.randint(0, 2, size=n)
        labels_sa2, cut_sa2 = simulated_annealing(labels_rand, max_iters=200000, T_start=10.0, T_end=0.01)
        if cut_sa2 > best_cut:
            best_cut = cut_sa2
            best_labels = labels_sa2

    # Step 5: Final local search polish
    if time_left() > 3.0:
        labels_final, cut_final = local_search(best_labels, max_rounds=50)
        if cut_final > best_cut:
            best_cut = cut_final
            best_labels = labels_final

    return best_labels.astype(int)
# EVOLVE-BLOCK-END

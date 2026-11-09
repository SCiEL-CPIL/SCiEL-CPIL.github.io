"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _base_priority(adj, n):
    degrees = adj.sum(axis=1).astype(np.float64)
    neighbor_degree_sums = (adj @ degrees).astype(np.float64)
    inv_deg = 1.0 / (degrees + 1.0)
    neighbor_inv_degree_sums = (adj @ inv_deg).astype(np.float64)

    M = 100000.0
    costs = degrees * M - neighbor_degree_sums + neighbor_inv_degree_sums
    return -costs


def _greedy_decode(adj, n, priority, selected=None):
    order = np.argsort(-priority, kind='stable')
    in_set = np.zeros(n, dtype=bool)
    if selected is not None:
        in_set[selected] = True
    for v in order:
        v = int(v)
        if in_set[v]:
            continue
        if adj[v].any():
            continue
        in_set[v] = True
        adj[v] = True
        adj[np.where(adj[v])[0]] = True
    return in_set


def priority(adjacency):
    start = time.time()
    deadline = start + 130.0

    adj_bool = np.asarray(adjacency, dtype=bool)
    n = adj_bool.shape[0]

    # Base priority (refined multi-term heuristic)
    base_prio = _base_priority(adj_bool, n)

    # Seed with greedy decode using a mutable copy of adjacency
    work = adj_bool.copy()
    in_set = _greedy_decode(work, n, base_prio)
    best_set = in_set.copy()
    best_size = int(in_set.sum())

    # Local search: add / remove / swap, bounded by deadline
    while time.time() < deadline - 5.0:
        improved = False

        # --- Add moves: pick an unselected vertex, drop its selected neighbors ---
        unselected = np.where(~best_set)[0]
        if len(unselected) > 0:
            # Candidate gain = 1 - (# selected neighbors)
            sel_neighbors = best_set[adj_bool[:, unselected]].sum(axis=1) if len(unselected) else np.array([])
            # For each unselected vertex u, count selected neighbors
            cand_gain = 1.0 - np.array([best_set[adj_bool[u]].sum() for u in unselected], dtype=np.float64)
            order_u = np.argsort(-cand_gain, kind='stable')
            for idx in order_u:
                if time.time() > deadline - 5.0:
                    break
                u = int(unselected[idx])
                if best_set[u]:
                    continue
                gain = cand_gain[idx]
                if gain <= 0:
                    break
                # Apply
                new_set = best_set.copy()
                new_set[u] = True
                nbrs = np.where(adj_bool[u])[0]
                new_set[nbrs] = False
                if int(new_set.sum()) > best_size:
                    best_set = new_set
                    best_size = int(best_set.sum())
                    improved = True

        # --- Remove moves: drop a selected vertex that frees more than it loses ---
        selected = np.where(best_set)[0]
        if len(selected) > 0:
            # For each selected v, count unselected neighbors that would become available
            freed = np.array([(~best_set)[adj_bool[v]].sum() for v in selected], dtype=np.float64)
            order_v = np.argsort(-freed, kind='stable')
            for idx in order_v:
                if time.time() > deadline - 5.0:
                    break
                v = int(selected[idx])
                if not best_set[v]:
                    continue
                f = freed[idx]
                if f <= 1:
                    break
                new_set = best_set.copy()
                new_set[v] = False
                if int(new_set.sum()) > best_size:
                    best_set = new_set
                    best_size = int(new_set.sum())
                    improved = True

        # --- Swap moves: replace a selected vertex with an unselected one (net gain) ---
        if len(selected) > 0 and len(unselected) > 0:
            # Sample a subset to keep it fast
            if len(selected) * len(unselected) > 20000:
                sel_sample = np.random.choice(selected, size=min(len(selected), 200), replace=False)
                uns_sample = np.random.choice(unselected, size=min(len(unselected), 200), replace=False)
            else:
                sel_sample = selected
                uns_sample = unselected
            best_swap_gain = 0
            best_swap = None
            for v in sel_sample:
                v = int(v)
                if not best_set[v]:
                    continue
                for u in uns_sample:
                    u = int(u)
                    if best_set[u]:
                        continue
                    if adj_bool[v, u]:
                        continue
                    # Swapping v out, u in: net change = 1 - 1 + (freed_by_v - blocked_by_u)
                    # Freed: unselected neighbors of v that become available (and not neighbors of u)
                    # Blocked: unselected non-neighbors of v that are neighbors of u and become blocked
                    freed_by_v = np.where((~best_set) & adj_bool[v] & (~adj_bool[u]))[0]
                    blocked_by_u = np.where((~best_set) & (~adj_bool[v]) & adj_bool[u] & (~adj_bool[u] | adj_bool[v]))[0]
                    # Simpler: just simulate
                    new_set = best_set.copy()
                    new_set[v] = False
                    new_set[u] = True
                    nbrs_u = np.where(adj_bool[u])[0]
                    new_set[nbrs_u] = False
                    new_size = int(new_set.sum())
                    if new_size > best_size:
                        best_swap = new_set
                        best_size = new_size
                        improved = True
                        break
                if best_swap is not None:
                    best_set = best_swap
                    break

        if not improved:
            break

    # Export membership as scores: selected get high scores (ordered by base priority), others low
    selected = np.where(best_set)[0]
    scores = np.full(n, -1.0, dtype=np.float64)
    if len(selected) > 0:
        # Order selected by base priority (descending) so greedy decoder picks them in that order
        sel_order = np.argsort(-base_prio[selected], kind='stable')
        for rank, s in enumerate(selected[sel_order]):
            scores[int(s)] = float(len(selected) - rank)  # high to low among selected
    return scores
# EVOLVE-BLOCK-END

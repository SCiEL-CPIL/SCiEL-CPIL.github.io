import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115

    n = 40

    # Base structure: 8 disjoint K5s
    base_adj = [0] * n
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u = base + a
                v = base + b
                base_adj[u] |= (1 << v)
                base_adj[v] |= (1 << u)

    existing_edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if (base_adj[u] >> v) & 1:
                existing_edges.add((u, v))

    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    candidate_pairs = [(u, v) for u, v in all_pairs if (u, v) not in existing_edges]

    best_edges_list = []
    best_count = 0

    restart = 0
    while time.time() < deadline:
        restart += 1
        current_adj = base_adj[:]

        random.seed(42 + restart)
        random.shuffle(candidate_pairs)

        # Greedy pass
        for u, v in candidate_pairs:
            if time.time() > deadline - 10:
                break
            if not ((current_adj[u] >> v) & 1):
                if can_add_edge_on(current_adj, u, v):
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)

        # Local search polish: try removing one edge and re-adding others
        if time.time() < deadline - 8:
            current_adj, improved = local_search_polish(current_adj, candidate_pairs, deadline)
            if improved:
                pass  # already updated in-place

        current_count = count_edges_on(current_adj)

        if current_count > best_count:
            best_count = current_count
            best_edges_list = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (current_adj[u] >> v) & 1:
                        best_edges_list.append([u, v])

            if best_count >= 105:
                break

    if not best_edges_list:
        for u in range(n):
            for v in range(u + 1, n):
                if (base_adj[u] >> v) & 1:
                    best_edges_list.append([u, v])

    return best_edges_list


def can_add_edge_on(adj, u, v):
    """
    Check if adding edge (u,v) to the graph represented by adj creates a C6.
    Uses bitmask operations for speed.
    """
    nu_mask = adj[u] & ~(1 << v)
    nv_mask = adj[v] & ~(1 << u)

    a = nu_mask
    while a:
        lb_a = a & (-a)
        ai = lb_a.bit_length() - 1
        a ^= lb_a

        d = nv_mask & ~lb_a
        while d:
            ld_d = d & (-d)
            di = ld_d.bit_length() - 1
            d ^= ld_d

            b_candidates = adj[ai] & ~(1 << u) & ~(1 << v) & ~(1 << di) & ~(1 << ai)
            c_candidates = adj[di] & ~(1 << u) & ~(1 << v) & ~(1 << ai) & ~(1 << di)

            # Check if any neighbor of any b in b_candidates is in c_candidates
            union_neighbors = 0
            bm = b_candidates
            while bm:
                lb_b = bm & (-bm)
                bi = lb_b.bit_length() - 1
                bm ^= lb_b
                union_neighbors |= adj[bi]

                if union_neighbors & c_candidates:
                    return False

    return True


def count_edges_on(adj):
    count = 0
    for i in range(40):
        count += adj[i].bit_count()
    return count // 2


def local_search_polish(adj, candidate_pairs, deadline):
    """
    Try to improve the graph by removing one edge and re-adding multiple edges.
    Returns (adj, improved_flag) where adj is modified in-place.
    """
    n = 40
    improved = False
    best_adj = adj[:]
    best_count = count_edges_on(adj)

    # Build list of existing edges
    existing = []
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                existing.append((u, v))

    # For each existing edge, try removing it and then greedily adding back
    # edges from candidate_pairs (including the removed one)
    for rem_u, rem_v in existing:
        if time.time() > deadline - 5:
            break

        # Remove edge
        adj[rem_u] &= ~(1 << rem_v)
        adj[rem_v] &= ~(1 << rem_u)

        # Now greedily try to add edges, prioritizing the removed edge first
        # Build a list: removed edge first, then other candidates
        temp_candidates = [(rem_u, rem_v)]
        for u, v in candidate_pairs:
            if (u, v) != (rem_u, rem_v) and not ((adj[u] >> v) & 1):
                temp_candidates.append((u, v))

        # Shuffle with a fixed seed based on the removed edge
        random.seed(rem_u * 1000 + rem_v)
        random.shuffle(temp_candidates[1:])  # Keep removed edge first

        added = 0
        for u, v in temp_candidates:
            if time.time() > deadline - 4:
                break
            if not ((adj[u] >> v) & 1):
                if can_add_edge_on(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    added += 1

        current_count = count_edges_on(adj)

        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
            improved = True
        else:
            # Restore original adj
            adj[:] = best_adj[:]

    return adj, improved

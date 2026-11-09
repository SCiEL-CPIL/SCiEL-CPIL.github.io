import time
import random
import numpy as np

N = 40
DEADLINE = 108  # safety margin under 150s kill

def _creates_c6(adj, u, v):
    """
    Returns True iff adding edge (u,v) creates a C6.
    C6: u-a-b-c-d-v, all distinct.
    Precompute union_adj[a] = OR of adj[b] for b in N(a)\{u,v,a} once per a.
    Then for each d: check if (union_adj[a] & c_mask[d]) has a vertex b
    that is actually in b_mask (i.e., b != d). We handle this by checking
    if any b in b_mask has adj[b] & c_mask != 0, but we use the precomputed
    union as a fast filter first.
    """
    nu = adj[u]
    nv = adj[v]
    mask_u = 1 << u
    mask_v = 1 << v

    a_mask = nu & ~mask_v
    while a_mask:
        lsb_a = a_mask & -a_mask
        a = lsb_a.bit_length() - 1
        a_mask ^= lsb_a
        mask_a = 1 << a

        b_base = adj[a] & ~mask_u & ~mask_v & ~mask_a
        if not b_base:
            continue

        # Precompute: for each b in b_base, adj[b]. Store as list.
        # Also compute the full union for fast filtering.
        b_list = []
        bm = b_base
        while bm:
            lsb_b = bm & -bm
            b = lsb_b.bit_length() - 1
            bm ^= lsb_b
            b_list.append((b, adj[b]))

        d_mask = nv & ~mask_u & ~mask_a
        while d_mask:
            lsb_d = d_mask & -d_mask
            d = lsb_d.bit_length() - 1
            d_mask ^= lsb_d
            mask_d = 1 << d

            c_mask = adj[d] & ~mask_u & ~mask_v & ~mask_a & ~mask_d
            if not c_mask:
                continue

            # Check: exists b in b_list with b != d such that adj[b] & c_mask != 0
            for b, adj_b in b_list:
                if b == d:
                    continue
                if adj_b & c_mask:
                    return True

    return False


def _count_edges(adj):
    total = 0
    for i in range(N):
        total += adj[i].bit_count()
    return total // 2


def _add_edge(adj, u, v):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)


def _remove_edge(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)


def _make_all_edges():
    return [(i, j) for i in range(N) for j in range(i + 1, N)]


def _greedy_from_order(adj, order):
    count = 0
    for u, v in order:
        if (adj[u] >> v) & 1:
            continue
        if _creates_c6(adj, u, v):
            continue
        _add_edge(adj, u, v)
        count += 1
    return count


def _local_search(adj, deadline, rng, max_rounds=50):
    """
    Local search: repeatedly try to add any missing edge that keeps C6-free.
    Then try swaps to escape local optima.
    """
    all_edges = _make_all_edges()
    best_count = _count_edges(adj)
    cur = adj[:]

    for rnd in range(max_rounds):
        if time.time() >= deadline:
            break
        improved = False
        # Try all missing edges in deterministic order for consistency
        for u, v in all_edges:
            if time.time() >= deadline:
                break
            if (cur[u] >> v) & 1:
                continue
            if _creates_c6(cur, u, v):
                continue
            _add_edge(cur, u, v)
            improved = True

        cnt = _count_edges(cur)
        if cnt > best_count:
            best_count = cnt
            adj = cur[:]

        if not improved:
            # Try swaps: remove one edge, add another
            swap_improved = False
            present_edges = [(u, v) for u, v in all_edges if (cur[u] >> v) & 1]
            if present_edges:
                rng.shuffle(present_edges)
                for u1, v1 in present_edges[:30]:
                    if time.time() >= deadline:
                        break
                    _remove_edge(cur, u1, v1)
                    added = False
                    for _ in range(20):
                        u2, v2 = rng.choice(all_edges)
                        if (cur[u2] >> v2) & 1:
                            continue
                        if _creates_c6(cur, u2, v2):
                            continue
                        _add_edge(cur, u2, v2)
                        added = True
                        break
                    if added:
                        swap_improved = True
                    else:
                        _add_edge(cur, u1, v1)
                    if swap_improved:
                        break
            if not swap_improved:
                break

    return adj, best_count


def _anneal(adj, deadline, rng):
    """
    Simulated annealing: accept improving moves always, 
    accept neutral with some probability, accept worsening rarely.
    """
    cur = adj[:]
    cur_count = _count_edges(cur)
    best = cur[:]
    best_count = cur_count
    all_edges = _make_all_edges()

    start = time.time()
    total_time = max(1.0, deadline - start)

    while time.time() < deadline:
        elapsed = time.time() - start
        frac = min(1.0, elapsed / total_time)
        temp = max(0.01, 1.0 - frac)

        r = rng.random()
        if r < 0.5:
            # Add random edge
            u, v = rng.choice(all_edges)
            if (cur[u] >> v) & 1:
                continue
            if _creates_c6(cur, u, v):
                continue
            _add_edge(cur, u, v)
            delta = 1
            move = ("add", u, v)
        elif r < 0.8:
            # Swap
            present = [(u, v) for u, v in all_edges if (cur[u] >> v) & 1]
            if not present:
                continue
            u1, v1 = rng.choice(present)
            _remove_edge(cur, u1, v1)
            added = False
            for _ in range(10):
                u2, v2 = rng.choice(all_edges)
                if (cur[u2] >> v2) & 1:
                    continue
                if _creates_c6(cur, u2, v2):
                    continue
                _add_edge(cur, u2, v2)
                added = True
                break
            if not added:
                _add_edge(cur, u1, v1)
                continue
            move = ("swap", (u1, v1), (u2, v2))
            delta = 0
        else:
            # Remove random edge (escape)
            present = [(u, v) for u, v in all_edges if (cur[u] >> v) & 1]
            if not present:
                continue
            u1, v1 = rng.choice(present)
            _remove_edge(cur, u1, v1)
            delta = -1
            move = ("remove", u1, v1)

        new_count = cur_count + delta
        if new_count > best_count:
            best_count = new_count
            best = cur[:]
        cur_count = new_count

        # Accept/reject
        if delta >= 0:
            continue
        # delta < 0: accept with probability temp
        if rng.random() < temp * 0.3:
            continue
        else:
            # Revert
            if move[0] == "add":
                _remove_edge(cur, move[1], move[2])
            elif move[0] == "swap":
                (u1, v1), (u2, v2) = move[1], move[2]
                _remove_edge(cur, u2, v2)
                _add_edge(cur, u1, v1)
            elif move[0] == "remove":
                _add_edge(cur, move[1], move[2])
            cur_count = _count_edges(cur)

    return best, best_count


def _k5_seed():
    """8 disjoint K5s as starting point."""
    adj = [0] * N
    for k in range(8):
        base = k * 5
        for i in range(5):
            for j in range(i + 1, 5):
                _add_edge(adj, base + i, base + j)
    return adj


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    rng = random.Random(42)
    all_edges = _make_all_edges()

    best_adj = [0] * N
    best_count = 0

    def refresh_best(adj):
        nonlocal best_adj, best_count
        c = _count_edges(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]

    # Phase 1: K5 seed + greedy + local search + anneal
    seed = _k5_seed()
    order = all_edges[:]
    rng.shuffle(order)
    _greedy_from_order(seed, order)
    refresh_best(seed)

    seed, c = _local_search(seed, deadline, rng, max_rounds=30)
    refresh_best(seed)

    seed, c = _anneal(seed, deadline, rng)
    refresh_best(seed)

    # Phase 2: Multiple random restarts
    while time.time() < deadline - 5:
        adj = [0] * N
        order = all_edges[:]
        rng.shuffle(order)
        _greedy_from_order(adj, order)
        refresh_best(adj)

        adj, c = _local_search(adj, deadline, rng, max_rounds=20)
        refresh_best(adj)

        if time.time() < deadline - 3:
            adj, c = _anneal(adj, deadline, rng)
            refresh_best(adj)

    # Final polish: systematically try all missing edges on best_adj
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            for u, v in all_edges:
                if time.time() >= deadline - 1:
                    break
                if (best_adj[u] >> v) & 1:
                    continue
                if _creates_c6(best_adj, u, v):
                    continue
                _add_edge(best_adj, u, v)
                best_count += 1
                improved = True

    # Reconstruct edge list
    edges = []
    for i in range(N):
        m = best_adj[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if j > i:
                edges.append([i, j])
            m ^= lsb
    return edges


if __name__ == "__main__":
    construct()

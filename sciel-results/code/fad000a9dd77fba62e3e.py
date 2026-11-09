import time
import random
import numpy as np

N = 40
DEADLINE = 115  # wall-clock seconds (safety margin under 150s kill)

def _creates_c6(adj, u, v):
    """
    Returns True iff adding edge (u, v) creates a C6.
    That happens iff there is a simple path of length 5 between u and v:
        u - a - b - c - d - v
    with all six vertices distinct.
    adj: list of int bitmasks, adj[i] has bit j set iff edge (i, j) present.
    """
    nu = adj[u]
    nv = adj[v]
    forbidden_uv = (1 << u) | (1 << v)

    # Iterate a in N(u), a != v
    a_mask = nu & ~(1 << v)
    while a_mask:
        lsb_a = a_mask & -a_mask
        a = lsb_a.bit_length() - 1
        a_mask ^= lsb_a

        # Iterate d in N(v), d not in {u, a}
        d_mask = nv & ~(1 << u) & ~(1 << a)
        while d_mask:
            lsb_d = d_mask & -d_mask
            d = lsb_d.bit_length() - 1
            d_mask ^= lsb_d

            forbidden = forbidden_uv | (1 << a) | (1 << d)

            # b in N(a), c in N(d), b,c distinct from {u,v,a,d}
            b_mask = adj[a] & ~forbidden
            if not b_mask:
                continue
            c_mask = adj[d] & ~forbidden
            if not c_mask:
                continue

            # Check if any b in b_mask is connected to any c in c_mask
            bm = b_mask
            while bm:
                lsb_b = bm & -bm
                b = lsb_b.bit_length() - 1
                bm ^= lsb_b
                if adj[b] & c_mask:
                    return True

    return False

def _count_edges(adj):
    total = 0
    for i in range(N):
        total += bin(adj[i]).count("1")
    return total // 2

def _add_edge(adj, u, v):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

def _remove_edge(adj, u, v):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)

def _make_all_edges():
    return [(i, j) for i in range(N) for j in range(i + 1, N)]

def _greedy_add(adj, rng, deadline):
    """Greedily add edges in random order, skipping C6-creating ones."""
    all_edges = _make_all_edges()
    rng.shuffle(all_edges)
    for u, v in all_edges:
        if time.time() >= deadline:
            break
        if (adj[u] >> v) & 1:
            continue
        if _creates_c6(adj, u, v):
            continue
        _add_edge(adj, u, v)

def _degree_guided_greedy(adj, rng, deadline):
    """
    Greedy edge addition prioritizing edges that connect high-degree vertices
    or that don't create many near-C6 paths. Uses a scoring heuristic.
    """
    all_edges = _make_all_edges()
    non_edges = [(u, v) for u, v in all_edges if not ((adj[u] >> v) & 1)]
    
    while non_edges and time.time() < deadline:
        # Score each non-edge: prefer those that don't create C6, and among those,
        # prefer high-degree-sum edges
        best_edge = None
        best_score = -1
        for u, v in non_edges:
            if _creates_c6(adj, u, v):
                continue
            du = bin(adj[u]).count("1")
            dv = bin(adj[v]).count("1")
            score = du + dv
            if score > best_score:
                best_score = score
                best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        _add_edge(adj, u, v)
        non_edges.remove(best_edge)

def _local_search(adj, deadline, rng):
    """
    Local search: try to add an edge, or perform swaps to unlock more additions.
    Returns (adj, count).
    """
    all_edges = _make_all_edges()
    best_count = _count_edges(adj)
    no_improve_streak = 0
    
    while time.time() < deadline:
        improved = False
        
        # Pass 1: Try all additions
        for u, v in all_edges:
            if time.time() >= deadline:
                break
            if (adj[u] >> v) & 1:
                continue
            if _creates_c6(adj, u, v):
                continue
            _add_edge(adj, u, v)
            improved = True
        
        # Pass 2: Swaps - remove one edge, try to add two (net +1) or add one (net 0)
        if not improved:
            # Try single swaps: remove e1, add e2
            for u1, v1 in all_edges:
                if time.time() >= deadline:
                    break
                if not ((adj[u1] >> v1) & 1):
                    continue
                _remove_edge(adj, u1, v1)
                added = False
                for u2, v2 in all_edges:
                    if (adj[u2] >> v2) & 1:
                        continue
                    if _creates_c6(adj, u2, v2):
                        continue
                    _add_edge(adj, u2, v2)
                    added = True
                    break
                if not added:
                    _add_edge(adj, u1, v1)
                else:
                    # Now try to add another edge
                    for u3, v3 in all_edges:
                        if (adj[u3] >> v3) & 1:
                            continue
                        if _creates_c6(adj, u3, v3):
                            continue
                        _add_edge(adj, u3, v3)
                        improved = True
                        break
        
        cnt = _count_edges(adj)
        if cnt > best_count:
            best_count = cnt
            no_improve_streak = 0
        else:
            no_improve_streak += 1
            if no_improve_streak >= 3:
                break
    
    return adj, best_count

def _anneal_from(adj, deadline, rng):
    """
    Simulated annealing: accept worsening moves with temperature-based probability.
    Focus on edge additions and 1-for-1 swaps.
    Returns (best_adj, best_count).
    """
    cur = adj[:]
    cur_count = _count_edges(cur)
    best = cur[:]
    best_count = cur_count
    all_edges = _make_all_edges()
    
    start = time.time()
    total_time = max(1e-6, deadline - start)
    
    while time.time() < deadline:
        elapsed = time.time() - start
        frac = min(1.0, elapsed / total_time)
        temp = max(0.01, 1.0 - frac)
        
        r = rng.random()
        if r < 0.6:
            # Add a random non-edge
            for _ in range(20):
                u, v = rng.choice(all_edges)
                if (cur[u] >> v) & 1:
                    continue
                if _creates_c6(cur, u, v):
                    continue
                _add_edge(cur, u, v)
                delta = 1
                move = ("add", u, v)
                break
            else:
                continue
        elif r < 0.9:
            # Swap: remove random edge, add random non-edge
            for _ in range(20):
                u1, v1 = rng.choice(all_edges)
                if not ((cur[u1] >> v1) & 1):
                    continue
                _remove_edge(cur, u1, v1)
                for _ in range(30):
                    u2, v2 = rng.choice(all_edges)
                    if (cur[u2] >> v2) & 1:
                        continue
                    if _creates_c6(cur, u2, v2):
                        continue
                    _add_edge(cur, u2, v2)
                    delta = 0
                    move = ("swap", (u1, v1), (u2, v2))
                    break
                else:
                    _add_edge(cur, u1, v1)
                    break
            else:
                continue
        else:
            # Remove a random edge (escape move)
            u1, v1 = rng.choice(all_edges)
            if not ((cur[u1] >> v1) & 1):
                continue
            _remove_edge(cur, u1, v1)
            delta = -1
            move = ("remove", u1, v1)
        
        new_count = cur_count + delta
        if new_count > best_count:
            best = cur[:]
            best_count = new_count
        cur_count = new_count
        
        # Accept/reject
        if delta > 0:
            continue
        p = 1.0 if delta == 0 else min(1.0, temp * 0.5)
        if rng.random() < p:
            continue
        else:
            # Revert
            if move[0] == "add":
                _remove_edge(cur, move[1], move[2])
            elif move[0] == "swap":
                (u2, v2), (u1, v1) = move[2], move[1]
                _remove_edge(cur, u2, v2)
                _add_edge(cur, u1, v1)
            elif move[0] == "remove":
                _add_edge(cur, move[1], move[2])
            cur_count = _count_edges(cur)
    
    return best, best_count

def _k5_seed_adj():
    """
    Build a C6-free seed from 8 disjoint K5's (40 vertices).
    Returns adj list.
    """
    adj = [0] * N
    for k in range(8):
        base = k * 5
        for i in range(5):
            for j in range(i + 1, 5):
                u = base + i
                v = base + j
                _add_edge(adj, u, v)
    return adj

def _design_seed_adj():
    """
    Build a seed from a combinatorial design: partition 40 vertices into groups
    and connect groups in a C6-free pattern.
    """
    adj = [0] * N
    # Create 8 groups of 5 vertices
    # Connect each vertex to all vertices in 3 other groups (complete bipartite between groups)
    # This creates a structure similar to a complete multipartite graph restricted to avoid C6
    groups = [list(range(k * 5, (k + 1) * 5)) for k in range(8)]
    
    # Connect groups in a cycle pattern: group i connects to groups (i+1)%8, (i+2)%8, (i+3)%8
    for i in range(8):
        for offset in [1, 2, 3]:
            j = (i + offset) % 8
            for u in groups[i]:
                for v in groups[j]:
                    _add_edge(adj, u, v)
    
    # Add internal edges within each group (K5 within each group)
    for k in range(8):
        base = k * 5
        for i in range(5):
            for j in range(i + 1, 5):
                u = base + i
                v = base + j
                _add_edge(adj, u, v)
    
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
    seed = _k5_seed_adj()
    _greedy_add(seed, rng, deadline)
    seed, c = _local_search(seed, deadline, rng)
    refresh_best(seed)
    if time.time() < deadline:
        seed, c = _anneal_from(seed, deadline, rng)
        refresh_best(seed)

    # Phase 2: Design seed + greedy + local search + anneal
    if time.time() < deadline:
        seed2 = _design_seed_adj()
        _greedy_add(seed2, rng, deadline)
        seed2, c = _local_search(seed2, deadline, rng)
        refresh_best(seed2)
        if time.time() < deadline:
            seed2, c = _anneal_from(seed2, deadline, rng)
            refresh_best(seed2)

    # Phase 3: Multiple random-start greedy builds with local search / anneal
    while time.time() < deadline:
        adj = [0] * N
        _greedy_add(adj, rng, deadline)
        adj, c = _local_search(adj, deadline, rng)
        refresh_best(adj)
        if time.time() < deadline:
            adj, c = _anneal_from(adj, deadline, rng)
            refresh_best(adj)

    # Phase 4: Final polish - try to add any remaining edges to best_adj
    if time.time() < deadline:
        polish_adj = best_adj[:]
        non_edges = []
        for u, v in all_edges:
            if not ((polish_adj[u] >> v) & 1):
                du = bin(polish_adj[u]).count("1")
                dv = bin(polish_adj[v]).count("1")
                non_edges.append((du + dv, u, v))
        non_edges.sort(reverse=True)
        for deg_sum, u, v in non_edges:
            if time.time() >= deadline:
                break
            if (polish_adj[u] >> v) & 1:
                continue
            if _creates_c6(polish_adj, u, v):
                continue
            _add_edge(polish_adj, u, v)
        refresh_best(polish_adj)

    # Reconstruct edge list from best_adj
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

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
    uv_mask = (1 << u) | (1 << v)

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

            forbidden = uv_mask | (1 << a) | (1 << d)

            # b in N(a) \ forbidden, c in N(d) \ forbidden
            b_mask = adj[a] & ~forbidden
            if not b_mask:
                continue
            c_mask = adj[d] & ~forbidden
            if not c_mask:
                continue

            # Check if any b in b_mask is connected to any c in c_mask
            # Fast path: if b_mask has few bits, iterate; else check union of adj[b] for b in b_mask
            # Actually, the most efficient is: for each b in b_mask, check adj[b] & c_mask
            # But we can also try: compute union_adj_b = OR of adj[b] for all b in b_mask, then & c_mask
            # This is faster when b_mask has many bits.
            bm = b_mask
            # If b_mask is small (<= 3 bits), iterate directly
            if bm & (bm - 1) == 0:  # single bit
                b = bm.bit_length() - 1
                if adj[b] & c_mask:
                    return True
            elif bm & (bm - 1) & (bm - 1) == 0:  # two bits
                b1 = (bm & -bm).bit_length() - 1
                b2 = (bm ^ (bm & -bm)).bit_length() - 1
                if (adj[b1] | adj[b2]) & c_mask:
                    return True
            else:
                # More bits: iterate but with early exit
                bm2 = bm
                while bm2:
                    lsb_b = bm2 & -bm2
                    b = lsb_b.bit_length() - 1
                    bm2 ^= lsb_b
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

def _random_seed(adj):
    """Add random edges greedily to seed."""
    all_edges = _make_all_edges()
    random.shuffle(all_edges)
    for u, v in all_edges:
        if (adj[u] >> v) & 1:
            continue
        if _creates_c6(adj, u, v):
            continue
        _add_edge(adj, u, v)

def _local_search(adj, max_iter, deadline, rng):
    """
    Local search: greedy fill + systematic 2-opt swaps.
    Returns (best_adj, best_count).
    """
    best_count = _count_edges(adj)
    all_edges = _make_all_edges()
    it = 0
    while time.time() < deadline and it < max_iter:
        it += 1
        improved = False

        # Pass 1: greedy additions in random order
        order = all_edges[:]
        rng.shuffle(order)
        for u, v in order:
            if time.time() >= deadline:
                break
            if (adj[u] >> v) & 1:
                continue
            if _creates_c6(adj, u, v):
                continue
            _add_edge(adj, u, v)
            improved = True

        # Pass 2: systematic 2-opt swaps - try removing each existing edge
        # and see if we can add more edges than we removed
        existing_edges = []
        for i in range(N):
            m = adj[i]
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if j > i:
                    existing_edges.append((i, j))
                m ^= lsb
        
        swap_order = existing_edges[:]
        rng.shuffle(swap_order)
        for e1 in swap_order:
            if time.time() >= deadline:
                break
            u1, v1 = e1
            if not ((adj[u1] >> v1) & 1):
                continue
            _remove_edge(adj, u1, v1)

            # Try to add as many edges as possible after removal
            added = 0
            add_order = all_edges[:]
            rng.shuffle(add_order)
            for u2, v2 in add_order:
                if time.time() >= deadline:
                    break
                if (adj[u2] >> v2) & 1:
                    continue
                if _creates_c6(adj, u2, v2):
                    continue
                _add_edge(adj, u2, v2)
                added += 1
                if added >= 2:
                    break

            if added > 0:
                improved = True
            else:
                # No improvement, restore original edge
                _add_edge(adj, u1, v1)

        # Pass 3: Try 1-remove, 1-add swaps (net zero but may enable future gains)
        if time.time() < deadline:
            existing_edges = []
            for i in range(N):
                m = adj[i]
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    if j > i:
                        existing_edges.append((i, j))
                    m ^= lsb
            
            for e1 in existing_edges:
                if time.time() >= deadline:
                    break
                u1, v1 = e1
                if not ((adj[u1] >> v1) & 1):
                    continue
                _remove_edge(adj, u1, v1)
                
                added_any = False
                add_order = all_edges[:]
                rng.shuffle(add_order)
                for u2, v2 in add_order:
                    if time.time() >= deadline:
                        break
                    if (adj[u2] >> v2) & 1:
                        continue
                    if _creates_c6(adj, u2, v2):
                        continue
                    _add_edge(adj, u2, v2)
                    added_any = True
                    break
                
                if not added_any:
                    _add_edge(adj, u1, v1)
                else:
                    improved = True

        # Pass 4: High-precision polish - try all possible single-edge additions
        # in a systematic order (by vertex index) to catch missed opportunities
        if time.time() < deadline:
            for i in range(N):
                for j in range(i + 1, N):
                    if time.time() >= deadline:
                        break
                    if (adj[i] >> j) & 1:
                        continue
                    if _creates_c6(adj, i, j):
                        continue
                    _add_edge(adj, i, j)
                    improved = True

        # Update best
        cnt = _count_edges(adj)
        if cnt > best_count:
            best_count = cnt
        # If no improvement, stop early
        if not improved:
            break
    return adj, best_count

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

def _fano_seed_adj():
    """
    Build a structured seed using Fano plane incidence.
    Partition 40 vertices into 8 groups of 5.
    Use Fano plane (7 points, 7 lines) to define inter-group connections.
    The 8th group connects to all others in a controlled way.
    This creates a structured bipartite-like inter-group graph that is
    more likely to be C6-free by construction.
    """
    adj = [0] * N
    
    # Fano plane: 7 points {0,1,2,3,4,5,6}, 7 lines
    # Lines: {0,1,2}, {0,3,4}, {0,5,6}, {1,3,5}, {1,4,6}, {2,3,6}, {2,4,5}
    fano_lines = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5),
    ]
    
    # For each group pair, use Fano structure to determine which vertices connect
    # Group g has vertices 5g .. 5g+4
    # For groups i and j (i != j), we want a sparse bipartite connection
    
    # Use the Fano plane to define: for groups corresponding to Fano points,
    # connect vertex a in group i to vertex b in group j based on line membership
    # Simplified: for each pair of groups, add edges in a pattern that avoids C6
    
    # Strategy: for each pair of groups (i, j), add a matching of size 2 or 3
    # that is structured to avoid creating C6s
    for i in range(8):
        for j in range(i + 1, 8):
            # Use (i+j) mod 7 to index into Fano lines for variety
            line_idx = (i + j) % 7
            line = fano_lines[line_idx]
            # Connect vertices based on Fano line pattern
            # Vertex 0 in group i connects to vertices in group j indexed by line
            for a in range(5):
                for b in range(5):
                    # Sparse connection: only connect if (a + b + i + j) % 3 == 0
                    # This gives roughly 1/3 density between groups
                    if (a + b + i * 7 + j) % 3 == 0:
                        u = i * 5 + a
                        v = j * 5 + b
                        _add_edge(adj, u, v)
    
    # Add intra-group edges (K5 within each group)
    for k in range(8):
        base = k * 5
        for i in range(5):
            for j in range(i + 1, 5):
                u = base + i
                v = base + j
                _add_edge(adj, u, v)
    
    return adj

def _anneal_from(adj, deadline, rng):
    """
    Simulated annealing on top of adj:
    - propose add/swap/remove to move toward higher edge count while C6-free.
    Accept improves always; accept neutral/swap with probability to escape.
    Returns best_adj, best_count.
    """
    cur = adj[:]
    cur_count = _count_edges(cur)
    best = cur[:]
    best_count = cur_count
    all_edges = _make_all_edges()

    t = deadline
    min_temp = 0.01
    # Time-based schedule
    start = time.time()
    total = max(1e-6, t - start)

    # Precompute degree for faster edge selection
    while time.time() < t:
        elapsed = time.time() - start
        frac = min(1.0, elapsed / total)
        temp = max(min_temp, 1.0 - frac)

        # Choose move type
        r = rng.random()
        if r < 0.55:
            # Add a random edge
            u, v = rng.choice(all_edges)
            if (cur[u] >> v) & 1:
                continue
            if _creates_c6(cur, u, v):
                continue
            _add_edge(cur, u, v)
            delta = 1
            move = ("add", u, v)
        elif r < 0.85:
            # Swap: remove a random existing edge, try to add a random edge
            # Prefer edges whose removal might unlock many new edges
            e1 = rng.choice(all_edges)
            u1, v1 = e1
            if not ((cur[u1] >> v1) & 1):
                continue
            _remove_edge(cur, u1, v1)
            added = False
            # Try multiple candidates
            for _ in range(20):
                e2 = rng.choice(all_edges)
                u2, v2 = e2
                if (cur[u2] >> v2) & 1:
                    continue
                if _creates_c6(cur, u2, v2):
                    continue
                _add_edge(cur, u2, v2)
                added = True
                move = ("swap", (u1, v1), (u2, v2))
                break
            if not added:
                _add_edge(cur, u1, v1)
                continue
            delta = 0
        else:
            # Remove a random edge (escape)
            e1 = rng.choice(all_edges)
            u1, v1 = e1
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
        # delta <= 0: accept with probability
        p = 1.0 if delta == 0 else min(1.0, temp * 0.5)
        if rng.random() < p:
            continue
        else:
            # revert
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

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    rng = random.Random(12345)
    all_edges = _make_all_edges()

    best_adj = [0] * N
    best_count = 0

    def refresh_best(adj):
        nonlocal best_adj, best_count
        c = _count_edges(adj)
        if c > best_count:
            best_count = c
            best_adj = adj[:]

    # Phase 1: K5 seed + local search + anneal
    seed = _k5_seed_adj()
    # Greedy add from random order - try multiple orderings
    for attempt in range(3):
        order = all_edges[:]
        rng.shuffle(order)
        for u, v in order:
            if (seed[u] >> v) & 1:
                continue
            if _creates_c6(seed, u, v):
                continue
            _add_edge(seed, u, v)
        # Check if we improved
        c = _count_edges(seed)
        if c > best_count:
            best_count = c
            best_adj = seed[:]
    seed, c = _local_search(seed, max_iter=150, deadline=deadline, rng=rng)
    refresh_best(seed)
    if time.time() < deadline:
        seed, c = _anneal_from(seed, deadline, rng)
        refresh_best(seed)

    # Phase 1b: Fano-plane structured seed
    if time.time() < deadline - 10:
        fano_seed = _fano_seed_adj()
        # Greedy add from random order
        for attempt in range(2):
            if time.time() >= deadline - 10:
                break
            order = all_edges[:]
            rng.shuffle(order)
            for u, v in order:
                if time.time() >= deadline - 10:
                    break
                if (fano_seed[u] >> v) & 1:
                    continue
                if _creates_c6(fano_seed, u, v):
                    continue
                _add_edge(fano_seed, u, v)
        c = _count_edges(fano_seed)
        if c > best_count:
            best_count = c
            best_adj = fano_seed[:]
        fano_seed, c = _local_search(fano_seed, max_iter=80, deadline=deadline, rng=rng)
        refresh_best(fano_seed)
        if time.time() < deadline - 5:
            fano_seed, c = _anneal_from(fano_seed, deadline, rng)
            refresh_best(fano_seed)

    # Phase 2: Multiple random-start greedy builds with local search / anneal
    # Use time chunks: each iteration gets a bounded time slice
    while time.time() < deadline:
        # Allocate time for this iteration
        remaining = deadline - time.time()
        if remaining < 5:
            break
        iter_deadline = time.time() + min(remaining, 20)

        # Try a few different random starts
        for start_attempt in range(2):
            if time.time() >= iter_deadline:
                break
            adj = [0] * N
            order = all_edges[:]
            rng.shuffle(order)
            for u, v in order:
                if time.time() >= iter_deadline - 1:
                    break
                if (adj[u] >> v) & 1:
                    continue
                if _creates_c6(adj, u, v):
                    continue
                _add_edge(adj, u, v)
            adj, c = _local_search(adj, max_iter=60, deadline=iter_deadline, rng=rng)
            refresh_best(adj)
            if time.time() < iter_deadline - 2:
                adj, c = _anneal_from(adj, iter_deadline, rng)
                refresh_best(adj)

    # Final polish: try greedy fill on the best solution multiple times
    if time.time() < deadline - 3:
        for polish_attempt in range(8):
            if time.time() >= deadline - 2:
                break
            polish_adj = best_adj[:]
            # Systematic order for final polish
            for i in range(N):
                for j in range(i + 1, N):
                    if time.time() >= deadline - 1:
                        break
                    if (polish_adj[i] >> j) & 1:
                        continue
                    if _creates_c6(polish_adj, i, j):
                        continue
                    _add_edge(polish_adj, i, j)
            refresh_best(polish_adj)
            if _count_edges(polish_adj) <= best_count:
                break

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

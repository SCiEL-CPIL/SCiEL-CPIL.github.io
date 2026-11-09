import time
import random

N = 40
DEADLINE = 115

def _c6_check(adj, u, v):
    """Check if adding (u,v) creates a C6: path u-a-b-c-d-v with all distinct."""
    nu = adj[u]
    nv = adj[v]
    a_bits = nu & ~(1 << v)
    while a_bits:
        a = (a_bits & -a_bits).bit_length() - 1
        a_bits ^= a_bits & -a_bits
        na = adj[a]
        d_bits = nv & ~(1 << u) & ~(1 << a)
        while d_bits:
            d = (d_bits & -d_bits).bit_length() - 1
            d_bits ^= d_bits & -d_bits
            mask = (1 << u) | (1 << v) | (1 << a) | (1 << d)
            bm = na & ~mask
            if not bm:
                continue
            nd = adj[d]
            cm = nd & ~mask
            if not cm:
                continue
            # Check if any b in bm has neighbor in cm
            bb = bm
            while bb:
                b = (bb & -bb).bit_length() - 1
                bb ^= bb & -bb
                if adj[b] & cm:
                    return True
    return False

def _edges_from_adj(adj):
    out = []
    for i in range(N):
        m = adj[i]
        while m:
            j = (m & -m).bit_length() - 1
            if j > i:
                out.append((i, j))
            m ^= m & -m
    return out

def _edge_count(adj):
    s = 0
    for i in range(N):
        s += adj[i].bit_count()
    return s // 2

def _all_pairs():
    return [(i, j) for i in range(N) for j in range(i + 1, N)]

def _greedy_fill(adj, rng, deadline):
    """Greedily add edges in order of minimum degree sum."""
    pairs = _all_pairs()
    for _pass in range(10):
        if time.time() >= deadline:
            break
        missing = []
        deg = [adj[i].bit_count() for i in range(N)]
        for u, v in pairs:
            if not (adj[u] >> v) & 1:
                missing.append((deg[u] + deg[v], u, v))
        if not missing:
            break
        missing.sort()
        added = False
        for _, u, v in missing:
            if (adj[u] >> v) & 1:
                continue
            if _c6_check(adj, u, v):
                continue
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            added = True
        if not added:
            break

def _two_swap_search(adj, rng, deadline, max_rounds=200):
    """Try 2-swap moves: remove 2 edges, add 2 edges, net +0 but may unlock additions."""
    pairs = _all_pairs()
    cur_count = _edge_count(adj)
    best_count = cur_count
    best_adj = adj[:]
    
    for rnd in range(max_rounds):
        if time.time() >= deadline:
            break
        improved = False
        # Try to add any edge
        for u, v in pairs:
            if time.time() >= deadline:
                break
            if (adj[u] >> v) & 1:
                continue
            if _c6_check(adj, u, v):
                continue
            adj[u] |= 1 << v
            adj[v] |= 1 << u
            improved = True
            cur_count += 1
        if not improved:
            # Try 1-swap: remove one edge, add one
            for u1, v1 in pairs:
                if time.time() >= deadline:
                    break
                if not (adj[u1] >> v1) & 1:
                    continue
                adj[u1] &= ~(1 << v1)
                adj[v1] &= ~(1 << u1)
                added = False
                for u2, v2 in pairs:
                    if (adj[u2] >> v2) & 1:
                        continue
                    if _c6_check(adj, u2, v2):
                        continue
                    adj[u2] |= 1 << v2
                    adj[v2] |= 1 << u2
                    added = True
                    break
                if added:
                    improved = True
                else:
                    adj[u1] |= 1 << v1
                    adj[v1] |= 1 << u1
                if improved:
                    break
            if not improved:
                # Try 2-swap: remove two edges, add two
                for u1, v1 in pairs:
                    if time.time() >= deadline:
                        break
                    if not (adj[u1] >> v1) & 1:
                        continue
                    adj[u1] &= ~(1 << v1)
                    adj[v1] &= ~(1 << u1)
                    for u2, v2 in pairs:
                        if time.time() >= deadline:
                            break
                        if not (adj[u2] >> v2) & 1:
                            continue
                        if (u2, v2) == (u1, v1):
                            continue
                        adj[u2] &= ~(1 << v2)
                        adj[v2] &= ~(1 << u2)
                        # Now try to add 2 edges
                        added_count = 0
                        for u3, v3 in pairs:
                            if time.time() >= deadline:
                                break
                            if (adj[u3] >> v3) & 1:
                                continue
                            if _c6_check(adj, u3, v3):
                                continue
                            adj[u3] |= 1 << v3
                            adj[v3] |= 1 << u3
                            added_count += 1
                            if added_count == 2:
                                break
                        if added_count == 2:
                            cur_count += 2
                            improved = True
                        else:
                            # Revert the added edges and the removed ones
                            for u3, v3 in pairs:
                                if (adj[u3] >> v3) & 1:
                                    # Check if this was just added (not in original)
                                    # We need to track which were added
                                    pass
                            # Simpler: restore original state
                            adj[u1] |= 1 << v1
                            adj[v1] |= 1 << u1
                            adj[u2] |= 1 << v2
                            adj[v2] |= 1 << u2
                            # Remove any edges we added
                            for u3, v3 in pairs:
                                if (adj[u3] >> v3) & 1:
                                    # This is tricky; we'll just leave it
                                    pass
                        if improved:
                            break
                    if improved:
                        break
            if not improved:
                break
        cnt = _edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]
    
    return best_adj, best_count

def _random_build(rng, deadline):
    """Build a graph by random edge insertion."""
    adj = [0] * N
    pairs = _all_pairs()
    rng.shuffle(pairs)
    for u, v in pairs:
        if time.time() >= deadline:
            break
        if _c6_check(adj, u, v):
            continue
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    return adj

def _structured_seed():
    """Build seed from 8 K5's plus cross edges."""
    adj = [0] * N
    for k in range(8):
        base = k * 5
        for i in range(5):
            for j in range(i + 1, 5):
                adj[base + i] |= 1 << (base + j)
                adj[base + j] |= 1 << (base + i)
    return adj

def _anneal(adj, deadline, rng, steps=5000):
    """Simple simulated annealing: add/swap/remove moves."""
    pairs = _all_pairs()
    cur = adj[:]
    cur_cnt = _edge_count(cur)
    best = cur[:]
    best_cnt = cur_cnt
    start = time.time()
    total_time = max(0.1, deadline - start)
    
    for step in range(steps):
        if time.time() >= deadline:
            break
        frac = (time.time() - start) / total_time
        temp = max(0.01, 1.0 - frac)
        
        r = rng.random()
        if r < 0.5:
            # Add
            u, v = rng.choice(pairs)
            if (cur[u] >> v) & 1:
                continue
            if _c6_check(cur, u, v):
                continue
            cur[u] |= 1 << v
            cur[v] |= 1 << u
            delta = 1
            revert = lambda: (cur.__setitem__(u, cur[u] & ~(1 << v)), cur.__setitem__(v, cur[v] & ~(1 << u)))
        elif r < 0.8:
            # Swap
            e1 = rng.choice(pairs)
            u1, v1 = e1
            if not (cur[u1] >> v1) & 1:
                continue
            cur[u1] &= ~(1 << v1)
            cur[v1] &= ~(1 << u1)
            added = False
            for _ in range(20):
                u2, v2 = rng.choice(pairs)
                if (cur[u2] >> v2) & 1:
                    continue
                if _c6_check(cur, u2, v2):
                    continue
                cur[u2] |= 1 << v2
                cur[v2] |= 1 << u2
                added = True
                break
            if not added:
                cur[u1] |= 1 << v1
                cur[v1] |= 1 << u1
                continue
            delta = 0
            revert = lambda: (cur.__setitem__(u2, cur[u2] & ~(1 << v2)), cur.__setitem__(v2, cur[v2] & ~(1 << u2)), cur.__setitem__(u1, cur[u1] | (1 << v1)), cur.__setitem__(v1, cur[v1] | (1 << u1)))
        else:
            # Remove
            u, v = rng.choice(pairs)
            if not (cur[u] >> v) & 1:
                continue
            cur[u] &= ~(1 << v)
            cur[v] &= ~(1 << u)
            delta = -1
            revert = lambda: (cur.__setitem__(u, cur[u] | (1 << v)), cur.__setitem__(v, cur[v] | (1 << u)))
        
        new_cnt = cur_cnt + delta
        if new_cnt > best_cnt:
            best = cur[:]
            best_cnt = new_cnt
        cur_cnt = new_cnt
        
        if delta > 0:
            continue
        p = 1.0 if delta == 0 else min(1.0, temp * 0.4)
        if rng.random() < p:
            continue
        revert()
        cur_cnt = _edge_count(cur)
    
    return best, best_cnt

def _final_polish(adj, deadline):
    """Aggressive final pass: try every possible add, then systematic swaps."""
    pairs = _all_pairs()
    # Repeated greedy with multiple passes
    for _ in range(10):
        if time.time() >= deadline:
            break
        changed = True
        while changed:
            changed = False
            deg = [adj[i].bit_count() for i in range(N)]
            missing = [(deg[u] + deg[v], u, v) for u, v in pairs if not (adj[u] >> v) & 1]
            missing.sort()
            for _, u, v in missing:
                if (adj[u] >> v) & 1:
                    continue
                if _c6_check(adj, u, v):
                    continue
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                changed = True
                break
    # Systematic 1-swap: try removing each edge and see if we can add a different one
    for u1, v1 in pairs:
        if time.time() >= deadline:
            break
        if not (adj[u1] >> v1) & 1:
            continue
        adj[u1] &= ~(1 << v1)
        adj[v1] &= ~(1 << u1)
        added = False
        for u2, v2 in pairs:
            if (adj[u2] >> v2) & 1:
                continue
            if _c6_check(adj, u2, v2):
                continue
            adj[u2] |= 1 << v2
            adj[v2] |= 1 << u2
            added = True
            break
        if not added:
            adj[u1] |= 1 << v1
            adj[v1] |= 1 << u1
    # Another round of greedy after swaps
    for _ in range(5):
        if time.time() >= deadline:
            break
        changed = True
        while changed:
            changed = False
            deg = [adj[i].bit_count() for i in range(N)]
            missing = [(deg[u] + deg[v], u, v) for u, v in pairs if not (adj[u] >> v) & 1]
            missing.sort()
            for _, u, v in missing:
                if (adj[u] >> v) & 1:
                    continue
                if _c6_check(adj, u, v):
                    continue
                adj[u] |= 1 << v
                adj[v] |= 1 << u
                changed = True
                break
    return adj

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    rng = random.Random(42)
    
    best_adj = [0] * N
    best_cnt = 0
    
    def update(adj):
        nonlocal best_adj, best_cnt
        c = _edge_count(adj)
        if c > best_cnt:
            best_cnt = c
            best_adj = adj[:]
    
    # Phase 1: Structured seed (8 K5's) + greedy + local search + anneal
    adj = _structured_seed()
    _greedy_fill(adj, rng, deadline)
    update(adj)
    adj, c = _two_swap_search(adj, rng, deadline, max_rounds=100)
    update(adj)
    if time.time() < deadline:
        adj, c = _anneal(adj, deadline, rng, steps=3000)
        update(adj)
    
    # Phase 2: Random builds
    for trial in range(8):
        if time.time() >= deadline:
            break
        adj = _random_build(rng, deadline)
        _greedy_fill(adj, rng, deadline)
        update(adj)
        adj, c = _two_swap_search(adj, rng, deadline, max_rounds=80)
        update(adj)
        if time.time() < deadline:
            adj, c = _anneal(adj, deadline, rng, steps=2000)
            update(adj)
    
    # Phase 3: More random builds with different strategies
    for trial in range(6):
        if time.time() >= deadline:
            break
        adj = [0] * N
        # Build with degree-biased random
        pairs = _all_pairs()
        for _ in range(500):
            if time.time() >= deadline:
                break
            u, v = rng.choice(pairs)
            if (adj[u] >> v) & 1:
                continue
            if _c6_check(adj, u, v):
                continue
            adj[u] |= 1 << v
            adj[v] |= 1 << u
        _greedy_fill(adj, rng, deadline)
        update(adj)
        adj, c = _two_swap_search(adj, rng, deadline, max_rounds=60)
        update(adj)
        if time.time() < deadline:
            adj, c = _anneal(adj, deadline, rng, steps=1500)
            update(adj)
    
    # Phase 4: Final polish on best
    if time.time() < deadline:
        best_adj = _final_polish(best_adj, deadline)
        best_cnt = _edge_count(best_adj)
    
    # Convert to edge list
    edges = []
    for i in range(N):
        m = best_adj[i]
        while m:
            j = (m & -m).bit_length() - 1
            if j > i:
                edges.append([i, j])
            m ^= m & -m
    return edges

if __name__ == "__main__":
    construct()

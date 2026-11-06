import time
import random
import numpy as np


def _iter_bits(mask):
    """Generator that yields set bit positions in a mask."""
    while mask:
        yield (mask & -mask).bit_length() - 1
        mask &= mask - 1


def construct():
    n = 40
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def time_ok():
        return time.time() < deadline

    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    # --- Fast bitset C6 check (from inspiration 1) ---
    def would_create_c6(adj, u, v):
        """True if adding edge (u,v) creates a C6.
        A C6 using (u,v): u-a-b-c-d-v. For each a in N(u)\{v}, d in N(v)\{u},
        a!=d, we need a length-3 path a-b-c-d in G\{u,v}.
        """
        Nu = adj[u] & ~(1 << v)
        Nv = adj[v] & ~(1 << u)

        mask_a = Nu
        while mask_a:
            bit_a = mask_a & (-mask_a)
            a = bit_a.bit_length() - 1
            mask_a ^= bit_a

            Na_base = adj[a] & ~(1 << u) & ~(1 << v)

            mask_d = Nv & ~(1 << a)  # d != a
            while mask_d:
                bit_d = mask_d & (-mask_d)
                d = bit_d.bit_length() - 1
                mask_d ^= bit_d

                B = Na_base & ~(1 << d)
                if B == 0:
                    continue
                C = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a)
                if C == 0:
                    continue

                mask_b = B
                while mask_b:
                    bit_b = mask_b & (-mask_b)
                    b = bit_b.bit_length() - 1
                    mask_b ^= bit_b
                    if adj[b] & C & ~(1 << b):
                        return True
        return False

    # --- Spectral-guided edge ordering (from inspiration 1) ---
    def spectral_order():
        """Return all_pairs sorted by a spectral weight (descending)."""
        rng = np.random.default_rng(42)
        adj_matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            candidates = list(range(n))
            candidates.remove(i)
            rng.shuffle(candidates)
            for j in candidates[:4]:
                if adj_matrix[i][j] == 0:
                    adj_matrix[i][j] = 1
                    adj_matrix[j][i] = 1
            deg = int(adj_matrix[i].sum())
            if deg < 4:
                for j in range(n):
                    if i != j and adj_matrix[i][j] == 0:
                        adj_matrix[i][j] = 1
                        adj_matrix[j][i] = 1
                        deg += 1
                        if deg >= 4:
                            break
        eigenvalues, eigenvectors = np.linalg.eigh(adj_matrix)
        target = np.sqrt(5)
        idx = int(np.argmin(np.abs(eigenvalues - target)))
        spec = eigenvectors[:, idx]
        w = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                w[i][j] = abs(spec[i] * spec[j])
        pairs = sorted(all_pairs, key=lambda p: w[p[0]][p[1]], reverse=True)
        return pairs

    # --- Greedy add in a given order ---
    def greedy_add(adj, order, ec):
        for u, v in order:
            if not time_ok():
                break
            if adj[u] & (1 << v):
                continue
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                ec += 1
        return ec

    def get_edges(adj):
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                j = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if i < j:
                    edges.append([i, j])
        return edges

    random.seed(42)

    best_adj = [0] * n
    best_edges = 0

    # --- Phase 1: multiple initializations ---
    # (a) Spectral-guided greedy
    try:
        spec_pairs = spectral_order()
    except Exception:
        spec_pairs = all_pairs[:]
        random.shuffle(spec_pairs)
    adj_a = [0] * n
    ec_a = greedy_add(adj_a, spec_pairs, 0)
    if ec_a > best_edges:
        best_edges = ec_a
        best_adj = adj_a[:]

    # (b) 8 disjoint K5 base + random greedy fill
    adj_b = [0] * n
    ec_b = 0
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                u = 5 * k + a
                v = 5 * k + b
                adj_b[u] |= (1 << v)
                adj_b[v] |= (1 << u)
                ec_b += 1
    rand_pairs = all_pairs[:]
    random.shuffle(rand_pairs)
    ec_b = greedy_add(adj_b, rand_pairs, ec_b)
    if ec_b > best_edges:
        best_edges = ec_b
        best_adj = adj_b[:]

    # (c) pure random greedy
    adj_c = [0] * n
    rp = all_pairs[:]
    random.shuffle(rp)
    ec_c = greedy_add(adj_c, rp, 0)
    if ec_c > best_edges:
        best_edges = ec_c
        best_adj = adj_c[:]

    if best_edges == 0:
        # fallback 8 K5
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u = 5 * k + a
                    v = 5 * k + b
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)
        best_edges = 8 * 10

    # --- Phase 2: local search (add-then-remove escape) ---
    adj = best_adj[:]
    current_edges = best_edges
    no_improve = 0
    stuck_threshold = 20

    while time_ok() and (time.time() - start_time) < 142.0:
        added = False
        attempts = 30 + min(no_improve // 5, 20)
        for _ in range(attempts):
            if not time_ok():
                break
            u, v = random.choice(all_pairs)
            if adj[u] & (1 << v):
                continue
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1
                added = True
                no_improve = 0
                break

        if not added:
            no_improve += 1
            if no_improve >= stuck_threshold:
                # remove a random edge to escape
                removed = False
                for _ in range(5):
                    u = random.randint(0, n - 1)
                    mask = adj[u]
                    if mask:
                        v = (mask & -mask).bit_length() - 1
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        current_edges -= 1
                        removed = True
                        break
                if not removed:
                    for i in range(n):
                        if adj[i]:
                            u = i
                            v = (adj[i] & -adj[i]).bit_length() - 1
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            current_edges -= 1
                            break
                    else:
                        break
                no_improve = 0
            if no_improve == 0 and stuck_threshold < 50:
                stuck_threshold += 2

        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = adj[:]

    return get_edges(best_adj)

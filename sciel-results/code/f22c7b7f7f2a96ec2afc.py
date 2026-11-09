import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # safety margin before 150s kill

    # ---------- Exact checker (bitmask, early exit) ----------
    def build_masks(n, adj):
        masks = [0] * n
        for u in range(n):
            m = 0
            row = adj[u]
            for v in range(n):
                if row[v]:
                    m |= (1 << v)
            masks[u] = m
        return masks

    def has_k4(n, masks):
        for u in range(n):
            mu = masks[u]
            for v in range(u + 1, n):
                if not (mu & (1 << v)):
                    continue
                common = mu & masks[v]
                common &= ~((1 << (v + 1)) - 1)
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if masks[wi] & (1 << xi):
                            return True
        return False

    def has_ind6(n, masks):
        full_mask = (1 << n) - 1
        comp = [0] * n
        for u in range(n):
            comp[u] = full_mask & ~(masks[u] | (1 << u))
        for u in range(n):
            cu = comp[u]
            for v in range(u + 1, n):
                if not (cu & (1 << v)):
                    continue
                common = cu & comp[v]
                common &= ~((1 << (v + 1)) - 1)
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if comp[wi] & (1 << xi):
                            common2 = cu & comp[v] & comp[wi] & comp[xi]
                            common2 &= ~((1 << (xi + 1)) - 1)
                            y_mask = common2
                            while y_mask:
                                lsb_y = y_mask & -y_mask
                                yi = lsb_y.bit_length() - 1
                                y_mask ^= lsb_y
                                z_mask = common2 & ~((1 << (yi + 1)) - 1)
                                while z_mask:
                                    lsb_z = z_mask & -z_mask
                                    zi = lsb_z.bit_length() - 1
                                    z_mask ^= lsb_z
                                    if comp[yi] & (1 << zi):
                                        return True
        return False

    def is_valid(n, adj):
        masks = build_masks(n, adj)
        if has_k4(n, masks):
            return False
        if has_ind6(n, masks):
            return False
        return True

    def edges_from_adj(n, adj):
        return [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]

    # ---------- Cost: exact count of K4 + Ind6 (soft surrogate) ----------
    def count_k4(n, masks):
        c = 0
        for u in range(n):
            mu = masks[u]
            for v in range(u + 1, n):
                if not (mu & (1 << v)):
                    continue
                common = mu & masks[v]
                common &= ~((1 << (v + 1)) - 1)
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if masks[wi] & (1 << xi):
                            c += 1
        return c

    def count_ind6(n, masks):
        full_mask = (1 << n) - 1
        comp = [0] * n
        for u in range(n):
            comp[u] = full_mask & ~(masks[u] | (1 << u))
        c = 0
        for u in range(n):
            cu = comp[u]
            for v in range(u + 1, n):
                if not (cu & (1 << v)):
                    continue
                common = cu & comp[v]
                common &= ~((1 << (v + 1)) - 1)
                w_mask = common
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if comp[wi] & (1 << xi):
                            common2 = cu & comp[v] & comp[wi] & comp[xi]
                            common2 &= ~((1 << (xi + 1)) - 1)
                            y_mask = common2
                            while y_mask:
                                lsb_y = y_mask & -y_mask
                                yi = lsb_y.bit_length() - 1
                                y_mask ^= lsb_y
                                z_mask = common2 & ~((1 << (yi + 1)) - 1)
                                while z_mask:
                                    lsb_z = z_mask & -z_mask
                                    zi = lsb_z.bit_length() - 1
                                    z_mask ^= lsb_z
                                    if comp[yi] & (1 << zi):
                                        c += 1
        return c

    def cost(n, adj):
        masks = build_masks(n, adj)
        return count_k4(n, masks) + count_ind6(n, masks)

    # ---------- Seeds ----------
    def paley17():
        sq = {x * x % 17 for x in range(1, 17)}
        adj = np.zeros((17, 17), dtype=bool)
        for i in range(17):
            for j in range(i + 1, 17):
                if (j - i) % 17 in sq:
                    adj[i, j] = True
                    adj[j, i] = True
        return adj

    def spectral_seed(n, seed):
        rng = np.random.default_rng(seed)
        # Structured random symmetric matrix (Hadamard/Paley-like mix)
        A = rng.normal(size=(n, n))
        A = (A + A.T) / 2.0
        # Add a bit of structure: low-rank + diagonal
        A += 0.3 * np.eye(n)
        # Eigendecomposition
        w, V = np.linalg.eigh(A)
        # Take top-k eigenvectors, form projection P = V_k V_k^T
        k = max(1, int(0.5 * n))
        Vk = V[:, -k:]
        P = Vk @ Vk.T
        # Binarize: edge if P_ij > threshold (median)
        thr = np.median(P)
        adj = (P > thr)
        np.fill_diagonal(adj, False)
        # Symmetrize
        adj = (adj | adj.T)
        return adj

    def structured_seed(n, seed):
        """Combine Paley blocks + random cross edges."""
        rng = np.random.default_rng(seed)
        p17 = paley17()
        adj = np.zeros((n, n), dtype=bool)
        # Place Paley blocks
        pos = 0
        while pos + 17 <= n:
            adj[pos:pos+17, pos:pos+17] = p17
            pos += 17
        # Random cross edges (tuned density)
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    continue
                # Same-block handled; cross-block random
                if i // 17 != j // 17:
                    if rng.random() < 0.5:
                        adj[i, j] = True
                        adj[j, i] = True
        return adj

    # ---------- Local search with soft objective ----------
    def local_search(n, adj0, time_budget, seed):
        rng = random.Random(seed)
        end_time = time.time() + time_budget
        adj = adj0.copy()
        cur_cost = cost(n, adj)
        if cur_cost == 0:
            return edges_from_adj(n, adj)

        best_cost = cur_cost
        best_adj = adj.copy()
        T = 3.0
        T_min = 0.05
        no_improve = 0
        max_no_improve = 400

        while time.time() < end_time and no_improve < max_no_improve:
            # Pick a random edge
            u = rng.randint(0, n - 1)
            v = rng.randint(0, n - 1)
            if u == v:
                continue
            old = adj[u, v]
            adj[u, v] = not old
            adj[v, u] = not old
            new_cost = cost(n, adj)
            if new_cost < cur_cost:
                cur_cost = new_cost
                no_improve = 0
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_adj = adj.copy()
                    if new_cost == 0:
                        return edges_from_adj(n, adj)
            else:
                delta = new_cost - cur_cost
                if delta > 0 and rng.random() < np.exp(-delta / T):
                    cur_cost = new_cost
                    no_improve += 1
                else:
                    adj[u, v] = old
                    adj[v, u] = old
                    no_improve += 1
            T = max(T_min, T * 0.9995)

        if is_valid(n, best_adj):
            return edges_from_adj(n, best_adj)
        return None

    def try_find(n, time_budget, seed):
        end_time = time.time() + time_budget
        # Try multiple seeds: spectral, structured, random
        seed_list = [
            spectral_seed(n, seed),
            structured_seed(n, seed),
            np.zeros((n, n), dtype=bool),
        ]
        # Fill the random one
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    seed_list[2][i, j] = True
                    seed_list[2][j, i] = True

        # Pick the best initial cost among seeds
        best_seed_adj = None
        best_seed_cost = float('inf')
        for s in seed_list:
            c = cost(n, s)
            if c < best_seed_cost:
                best_seed_cost = c
                best_seed_adj = s
            if c == 0:
                return edges_from_adj(n, s)

        # Local search from best seed
        result = local_search(n, best_seed_adj, time_budget, seed)
        if result is not None:
            return result

        # If time remains, try another seed
        if time.time() < end_time:
            for s in seed_list:
                if s is best_seed_adj:
                    continue
                if time.time() >= end_time:
                    break
                result = local_search(n, s, end_time - time.time(), seed + 1000)
                if result is not None:
                    return result
        return None

    # ---------- Main: grow n ----------
    best_n = 17
    best_edges = edges_from_adj(17, paley17())

    # Verify seed
    if is_valid(17, paley17()):
        best_n = 17
    else:
        # Fallback: random search for small n
        for n in range(10, 17):
            for attempt in range(50):
                adj = np.zeros((n, n), dtype=bool)
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            adj[i, j] = True
                            adj[j, i] = True
                if is_valid(n, adj):
                    best_n = n
                    best_edges = edges_from_adj(n, adj)
                    break
            if best_n >= n:
                break

    # Try to grow from best_n + 1 upward
    for n in range(best_n + 1, 36):
        if time.time() > deadline:
            break
        remaining = deadline - time.time()
        if remaining < 3.0:
            break
        # Budget: more time for smaller n
        if n <= 22:
            budget = min(remaining * 0.4, 6.0)
        elif n <= 28:
            budget = min(remaining * 0.25, 4.0)
        else:
            budget = min(remaining * 0.15, 3.0)

        edges = try_find(n, budget, seed=n * 101 + 7)
        if edges is not None:
            best_n = n
            best_edges = edges
            if n >= 35:
                break

    return {'n': best_n, 'edges': best_edges}

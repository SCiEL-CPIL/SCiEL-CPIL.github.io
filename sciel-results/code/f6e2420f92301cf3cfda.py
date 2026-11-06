import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 120s limit

    n = 40
    full_mask = (1 << n) - 1

    def would_create_c6(adj, u, v):
        """Fast bitmask check: does adding (u,v) create a C6?
        Cycle u-a-b-c-d-v-u with all distinct vertices."""
        a_mask = adj[u] & ~(1 << v)
        d_mask = adj[v] & ~(1 << u)
        while a_mask:
            la = a_mask & -a_mask
            a = la.bit_length() - 1
            a_mask ^= la
            Na = adj[a]
            dd = d_mask & ~la
            while dd:
                ld = dd & -dd
                d = ld.bit_length() - 1
                dd ^= ld
                Nd = adj[d]
                b_mask = Na & ~(1 << u) & ~(1 << v) & ~ld
                c_mask = Nd & ~(1 << u) & ~(1 << v) & ~la
                if not (b_mask & c_mask):
                    continue
                # need b in b_mask, c in c_mask, b != c, edge b-c
                bm = b_mask
                while bm:
                    lb = bm & -bm
                    b = lb.bit_length() - 1
                    bm ^= lb
                    if adj[b] & (c_mask & ~lb):
                        return True
        return False

    def build_adj(edges_set):
        adj = [0] * n
        for u, v in edges_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def find_one_c6(adj):
        """Return one 6-cycle as [u,a,b,c,d,v] or None. Bitmask O(n^4)."""
        for u in range(n):
            nu = adj[u]
            v_mask = (~nu) & ~(1 << u) & full_mask
            while v_mask:
                lv = v_mask & -v_mask
                v = lv.bit_length() - 1
                v_mask ^= lv
                nv = adj[v]
                a_mask = nu
                d_mask = nv
                while a_mask:
                    la = a_mask & -a_mask
                    a = la.bit_length() - 1
                    a_mask ^= la
                    Na = adj[a]
                    dm = d_mask & ~la
                    while dm:
                        ld = dm & -dm
                        d = ld.bit_length() - 1
                        dm ^= ld
                        Nd = adj[d]
                        b_mask = Na & ~(1 << u) & ~lv & ~ld & ~la
                        c_mask = Nd & ~(1 << u) & ~lv & ~la & ~ld
                        if not (b_mask & c_mask):
                            continue
                        bm = b_mask
                        while bm:
                            lb = bm & -bm
                            b = lb.bit_length() - 1
                            bm ^= lb
                            if adj[b] & (c_mask & ~lb):
                                # find c
                                cm = c_mask & ~lb
                                while cm:
                                    lc = cm & -cm
                                    c = lc.bit_length() - 1
                                    cm ^= lc
                                    if adj[b] & lc:
                                        return [u, a, b, c, d, v]
        return None

    def prune_to_c6free(adj, seed_val, time_limit):
        """Remove edges to destroy all C6s, preferring high-degree edges."""
        random.seed(seed_val)
        guard = 0
        while time.time() < time_limit:
            guard += 1
            if guard > 5000:
                break
            cyc = find_one_c6(adj)
            if cyc is None:
                return True
            # pick a random edge on the cycle; prefer high-degree endpoint
            best_e = None
            best_key = -1
            for i in range(6):
                x, y = cyc[i], cyc[(i + 1) % 6]
                key = adj[x].bit_count() + adj[y].bit_count()
                if key > best_key:
                    best_key = key
                    best_e = (x, y)
            # randomize among top to diversify
            r = random.random()
            if r < 0.5:
                best_e = best_e
            else:
                i = random.randint(0, 5)
                best_e = (cyc[i], cyc[(i + 1) % 6])
            x, y = best_e
            if x > y:
                x, y = y, x
            adj[x] &= ~(1 << y)
            adj[y] &= ~(1 << x)
        # verify
        return find_one_c6(adj) is None

    def edges_from_adj(adj):
        es = set()
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    es.add((u, v))
        return es

    def dense_prune_seed(seed_val, p, time_limit):
        """Dense random graph -> prune to C6-free. Returns edge set (may be empty)."""
        random.seed(seed_val)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        ok = prune_to_c6free(adj, seed_val, time_limit)
        if not ok:
            return None
        return edges_from_adj(adj)

    def k5_seed():
        es = set()
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u, v = 5 * k + a, 5 * k + b
                    es.add((u, v))
        return es

    def run_greedy(start_edges, seed_val):
        random.seed(seed_val)
        adj = build_adj(start_edges)
        edges = set(start_edges)
        possible_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] & (1 << j)):
                    possible_edges.append((i, j))
        random.shuffle(possible_edges)
        for u, v in possible_edges:
            if time.time() > deadline - 1:
                return edges
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.add((u, v))
        return edges

    def local_search(edges_set, seed_val):
        random.seed(seed_val)
        adj = build_adj(edges_set)
        current_edges = set(edges_set)

        def regen_nonedges():
            ne = []
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] & (1 << j)):
                        ne.append((i, j))
            random.shuffle(ne)
            return ne

        # Pass 1: add any possible non-edge
        non_edges = regen_nonedges()
        for u, v in non_edges:
            if time.time() > deadline - 1:
                break
            if not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges.add((u, v))

        # Pass 2: swaps (remove 1, add up to 2)
        removable = list(current_edges)
        random.shuffle(removable)
        non_edges = regen_nonedges()
        consecutive_failures = 0
        max_failures = 40
        while time.time() < deadline - 1:
            if consecutive_failures > max_failures:
                break
            if not removable:
                break
            e_rem = removable[random.randint(0, len(removable) - 1)]
            u_rem, v_rem = e_rem
            old_adj = adj[:]
            old_edges = set(current_edges)
            adj[u_rem] &= ~(1 << v_rem)
            adj[v_rem] &= ~(1 << u_rem)
            current_edges.remove(e_rem)
            added = 0
            local_non_edges = non_edges[:]
            random.shuffle(local_non_edges)
            for u, v in local_non_edges:
                if added >= 2:
                    break
                if time.time() > deadline - 1:
                    break
                if (u, v) in current_edges:
                    continue
                if not would_create_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    current_edges.add((u, v))
                    added += 1
            if added >= 2:
                consecutive_failures = 0
                removable = list(current_edges)
                random.shuffle(removable)
                non_edges = regen_nonedges()
            else:
                adj = old_adj
                current_edges = old_edges
                consecutive_failures += 1
        return current_edges

    def polish(edges_set, seed_val):
        """Final high-precision polish: greedy + local search on the best graph."""
        if time.time() > deadline - 2:
            return edges_set
        e = run_greedy(edges_set, seed_val)
        if time.time() < deadline - 2:
            e = local_search(e, seed_val + 7)
        return e

    def perturb_and_refine(edges_set, seed_val, num_remove):
        random.seed(seed_val)
        perturbed = set(edges_set)
        removable_list = list(perturbed)
        random.shuffle(removable_list)
        for i in range(min(num_remove, len(removable_list))):
            perturbed.remove(removable_list[i])
        edges = run_greedy(perturbed, seed_val + 1000)
        if time.time() < deadline - 4:
            edges = local_search(edges, seed_val + 2000)
        return edges

    best_edges = set()
    best_count = 0
    seed_counter = 42
    restart_count = 0

    # Phase 1: diverse restarts. New: dense-prune seeds give a strong, near-optimal
    # starting point for the shared greedy+local-search polish.
    while time.time() < deadline - 18:
        strategy = restart_count % 4
        if strategy == 0:
            start_edges = k5_seed()
            seed = start_edges
        elif strategy == 1:
            start_edges = set()
            seed = start_edges
        elif strategy == 2:
            start_edges = set()
            random.seed(seed_counter)
            for _ in range(30):
                u, v = random.sample(range(n), 2)
                if u > v:
                    u, v = v, u
                start_edges.add((u, v))
            seed = start_edges
        else:
            # Dense random init -> prune to C6-free (strong seed)
            p = 0.35 + 0.1 * (restart_count % 3)
            seed = dense_prune_seed(seed_counter + 5000, p, deadline - 18)
            if seed is None:
                seed = k5_seed()

        edges = run_greedy(seed, seed_counter)
        if time.time() < deadline - 12:
            edges = local_search(edges, seed_counter + 1000)

        count = len(edges)
        if count > best_count:
            best_count = count
            best_edges = set(edges)

        seed_counter += 1
        restart_count += 1

    # Phase 2: perturb the best graph to escape local optima
    while time.time() < deadline - 4:
        if not best_edges:
            break
        num_to_remove = random.randint(3, 8)
        edges = perturb_and_refine(best_edges, seed_counter, num_to_remove)
        count = len(edges)
        if count > best_count:
            best_count = count
            best_edges = set(edges)
        seed_counter += 1
        restart_count += 1

    # Phase 3: final polish of the best graph
    if time.time() < deadline - 2:
        polished = polish(best_edges, seed_counter)
        if len(polished) > best_count:
            best_edges = polished

    result = [[u, v] for u, v in best_edges]
    return result

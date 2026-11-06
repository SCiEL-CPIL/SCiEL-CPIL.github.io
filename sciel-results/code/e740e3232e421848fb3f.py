import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # safety margin before 150s kill
    n = 40

    def would_create_c6(adj, u, v):
        """Return True if adding edge (u,v) creates a C6.
        A new C6 must use (u,v): need a path u-a-b-c-d-v of length 5
        with a,b,c,d distinct and none equal to u or v.
        """
        Nu = adj[u]
        Nv = adj[v]

        a_mask = Nu & ~(1 << v)
        while a_mask:
            lsb_a = a_mask & -a_mask
            a = lsb_a.bit_length() - 1
            a_mask ^= lsb_a

            d_mask = Nv & ~(1 << u) & ~(1 << a)
            while d_mask:
                lsb_d = d_mask & -d_mask
                d = lsb_d.bit_length() - 1
                d_mask ^= lsb_d

                Na = adj[a]
                Nd = adj[d]

                b_mask = Na & ~(1 << u) & ~(1 << v) & ~(1 << d)
                c_mask = Nd & ~(1 << u) & ~(1 << v) & ~(1 << a)

                b_temp = b_mask
                while b_temp:
                    lsb_b = b_temp & -b_temp
                    b = lsb_b.bit_length() - 1
                    b_temp ^= lsb_b
                    # c must be in N(d) ∩ N(b), not in {u,v,a,b}
                    if adj[b] & c_mask & ~(1 << b):
                        return True
        return False

    def build_adj(edge_set):
        adj = [0] * n
        for u, v in edge_set:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def non_edge_list(adj):
        out = []
        for i in range(n):
            mi = adj[i]
            for j in range(i + 1, n):
                if not (mi >> j) & 1:
                    out.append((i, j))
        return out

    def greedy_fill(adj, edge_set, seed_val):
        random.seed(seed_val)
        non_edges = non_edge_list(adj)
        random.shuffle(non_edges)
        for u, v in non_edges:
            if time.time() > deadline - 2.0:
                break
            if not (adj[u] >> v) & 1 and not would_create_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_set.add((u, v))
        return adj, edge_set

    def local_search(adj, edge_set, seed_val):
        random.seed(seed_val)

        # Pass 1: greedy add
        adj, edge_set = greedy_fill(adj, edge_set, seed_val)

        # Pass 2: 1-out-2-in swaps (remove 1 edge, try to add 2)
        removable = list(edge_set)
        random.shuffle(removable)
        non_edges = non_edge_list(adj)
        random.shuffle(non_edges)

        consecutive_failures = 0
        max_failures = 40

        while time.time() < deadline - 2.0:
            if consecutive_failures > max_failures:
                break
            if not removable:
                break

            e_rem = removable[random.randint(0, len(removable) - 1)]
            u_rem, v_rem = e_rem

            old_adj = adj[:]
            old_edges = set(edge_set)

            adj[u_rem] &= ~(1 << v_rem)
            adj[v_rem] &= ~(1 << u_rem)
            edge_set.remove(e_rem)

            added = 0
            local_non = non_edges[:]
            random.shuffle(local_non)
            for u, v in local_non:
                if added >= 2:
                    break
                if time.time() > deadline - 2.0:
                    break
                if (u, v) in edge_set:
                    continue
                if not would_create_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_set.add((u, v))
                    added += 1

            if added >= 2:
                consecutive_failures = 0
                removable = list(edge_set)
                random.shuffle(removable)
                non_edges = non_edge_list(adj)
                random.shuffle(non_edges)
            else:
                adj = old_adj
                edge_set = old_edges
                consecutive_failures += 1

        return adj, edge_set

    def verify_no_c6(adj):
        """Full check: no 6-cycle anywhere. Returns True if C6-free."""
        for u in range(n):
            Nu = adj[u]
            a_mask = Nu
            while a_mask:
                lsb_a = a_mask & -a_mask
                a = lsb_a.bit_length() - 1
                a_mask ^= lsb_a
                # v must be > u (to avoid double counting) and in N(a)
                v_mask = adj[a] & (~((1 << (u + 1)) - 1))
                while v_mask:
                    lsb_v = v_mask & -v_mask
                    v = lsb_v.bit_length() - 1
                    v_mask ^= lsb_v
                    # b in N(v)\{u,a}, c in N(a)\{u,v}, need b~c, b!=c, b,c not in {u,a,v}
                    b_mask = adj[v] & ~(1 << u) & ~(1 << a)
                    c_mask = adj[a] & ~(1 << u) & ~(1 << v)
                    b_temp = b_mask
                    while b_temp:
                        lsb_b = b_temp & -b_temp
                        b = lsb_b.bit_length() - 1
                        b_temp ^= lsb_b
                        if adj[b] & c_mask & ~(1 << b):
                            return False
        return True

    def make_k5_seed():
        s = set()
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u, v = 5 * k + a, 5 * k + b
                    s.add((u, v))
        return s

    best_edges = set()
    best_count = 0

    def consider(adj, edge_set):
        nonlocal best_edges, best_count
        if len(edge_set) > best_count:
            if verify_no_c6(adj):
                best_count = len(edge_set)
                best_edges = set(edge_set)

    seed_counter = 42
    restart_count = 0

    # Phase 1: multi-restart (alternating K5 seed and empty seed)
    while time.time() < deadline - 12.0:
        if restart_count % 2 == 0:
            start_edges = make_k5_seed()
        else:
            start_edges = set()

        adj = build_adj(start_edges)
        edge_set = set(start_edges)

        adj, edge_set = local_search(adj, edge_set, seed_counter)
        consider(adj, edge_set)

        seed_counter += 1
        restart_count += 1

    # Phase 2: perturbation of the best graph (shake & re-fill)
    while time.time() < deadline - 2.0:
        if not best_edges:
            break
        random.seed(seed_counter)
        perturbed = set(best_edges)
        num_remove = random.randint(6, 12)
        rem_list = list(perturbed)
        random.shuffle(rem_list)
        for i in range(min(num_remove, len(rem_list))):
            perturbed.discard(rem_list[i])

        adj = build_adj(perturbed)
        edge_set = set(perturbed)
        adj, edge_set = local_search(adj, edge_set, seed_counter + 5000)
        consider(adj, edge_set)
        seed_counter += 1

    # Build the final edge list
    result = []
    for u, v in best_edges:
        result.append([u, v])
    return result

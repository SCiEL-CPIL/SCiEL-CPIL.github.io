import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin under 120s

    n = 40

    def elapsed_ok():
        return time.time() < deadline

    # ---------- Fast C6 check using NumPy ----------
    def has_c6(adj):
        """adj: (n,n) int array 0/1. Returns True if any C6 exists."""
        # C6: u-v-w-x-y-z-u  (6 distinct vertices, 6 edges)
        # Equivalent: exists u,v,w,x,y,z all distinct with
        # adj[u,v]=adj[v,w]=adj[w,x]=adj[x,y]=adj[y,z]=adj[z,u]=1
        # Efficient approach: for each pair (u,v) that is an edge,
        # check if there's a path of length 4 from u to v in G \ {edge uv}.
        # Path of length 4: u-a-b-c-v, all distinct.
        # We can use matrix powers but that's overkill. Instead:
        # For each edge (u,v), remove it, then check if dist(u,v) <= 4 in remaining graph
        # and there's a path of exactly length 4 (or we can just check length 4 existence).
        # Actually simpler: a C6 through edge (u,v) means a path u-a-b-c-v of length 4
        # with all vertices distinct. Let's check for each edge (u,v):
        #   exists a in N(u)\{v}, c in N(v)\{u}, a!=c,
        #   exists b in N(a)∩N(c), b not in {u,v,a,c}
        # This is the same logic as before but we can vectorize the inner part.

        # For speed, let's use the following approach:
        # For each pair (a,c) with a!=c:
        #   common = N(a) & N(c)  (candidates for b)
        #   For each b in common:
        #     u in N(a)\{b,c}, v in N(c)\{b,a}, u!=v, adj[u][v]=1
        #     Actually we need u-v edge AND u in N(a), v in N(c), u!=v, u!=b,c, v!=b,a
        # This is getting complex. Let me just use the original bitset logic but with
        # early termination and NumPy for the final verification.

        # Actually, for the local search we need a FAST check. Let me use the bitset
        # approach but only for the specific edge being added/removed.

        return False  # placeholder, we'll use the incremental check in local search

    # ---------- Bitmask-based fast C6 creation check ----------
    def creates_c6_bitmask(adj_bits, u, v):
        """adj_bits: list of int bitmasks. Check if adding (u,v) creates a C6."""
        mask_uv = (1 << u) | (1 << v)
        nu = adj_bits[u] & ~(1 << v)
        nv = adj_bits[v] & ~(1 << u)
        a = nu
        while a:
            lb = a & (-a)
            ai = lb.bit_length() - 1
            a ^= lb
            d = nv & ~lb
            while d:
                ld = d & (-d)
                di = ld.bit_length() - 1
                d ^= ld
                bmask = adj_bits[ai] & ~lb & ~ld & ~mask_uv
                b = bmask
                while b:
                    lb2 = b & (-b)
                    bi = lb2.bit_length() - 1
                    b ^= lb2
                    cmask = adj_bits[bi] & adj_bits[di] & ~lb & ~lb2 & ~ld & ~mask_uv
                    if cmask:
                        return True
        return False

    def count_edges_bitmask(adj_bits):
        return sum(adj_bits[u].bit_count() for u in range(n)) // 2

    # ---------- Seed: 8 disjoint K5s ----------
    def make_seed():
        """Create 8 disjoint K5s on vertices 0..39."""
        adj = [0] * n
        for k in range(8):
            base = k * 5
            for i in range(5):
                for j in range(i + 1, 5):
                    u = base + i
                    v = base + j
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    # ---------- Greedy completion on a given adj ----------
    def greedy_complete(adj_bits, seed):
        """Add edges in random order if they don't create C6."""
        random.seed(seed)
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(pairs)
        for (u, v) in pairs:
            if time.time() >= deadline - 2:
                break
            if not (adj_bits[u] >> v) & 1:
                if not creates_c6_bitmask(adj_bits, u, v):
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
        return adj_bits

    # ---------- Simulated Annealing local search ----------
    def local_search(adj_bits, seed, max_iter=50000):
        """
        Local search: try to add edges, and if stuck, remove a random edge and re-add.
        This helps escape local optima.
        """
        random.seed(seed)
        best_adj = adj_bits[:]
        best_cnt = count_edges_bitmask(adj_bits)
        current_adj = adj_bits[:]
        current_cnt = best_cnt

        # Build list of current edges
        edges = set()
        for u in range(n):
            m = current_adj[u]
            for v in range(u + 1, n):
                if (m >> v) & 1:
                    edges.add((u, v))

        # Non-edges
        all_pairs = set((u, v) for u in range(n) for v in range(u + 1, n))
        non_edges = all_pairs - edges

        for it in range(max_iter):
            if time.time() >= deadline - 2:
                break

            # Try to add a random non-edge
            if non_edges:
                u, v = random.choice(list(non_edges))
                if not creates_c6_bitmask(current_adj, u, v):
                    # Accept the addition
                    current_adj[u] |= (1 << v)
                    current_adj[v] |= (1 << u)
                    current_cnt += 1
                    edges.add((u, v))
                    non_edges.discard((u, v))
                    if current_cnt > best_cnt:
                        best_cnt = current_cnt
                        best_adj = current_adj[:]
                else:
                    # Can't add; try a remove-readd move
                    if edges and random.random() < 0.3:
                        # Remove a random edge
                        ru, rv = random.choice(list(edges))
                        current_adj[ru] &= ~(1 << rv)
                        current_adj[rv] &= ~(1 << ru)
                        current_cnt -= 1
                        edges.discard((ru, rv))
                        non_edges.add((ru, rv))

                        # Now try to add several random non-edges
                        added_any = False
                        attempts = 0
                        while non_edges and attempts < 20:
                            attempts += 1
                            au, av = random.choice(list(non_edges))
                            if not creates_c6_bitmask(current_adj, au, av):
                                current_adj[au] |= (1 << av)
                                current_adj[av] |= (1 << au)
                                current_cnt += 1
                                edges.add((au, av))
                                non_edges.discard((au, av))
                                added_any = True

                        if not added_any and current_cnt < best_cnt - 2:
                            # Revert: put back the removed edge
                            current_adj[ru] |= (1 << rv)
                            current_adj[rv] |= (1 << ru)
                            current_cnt += 1
                            edges.add((ru, rv))
                            non_edges.discard((ru, rv))

        return best_adj, best_cnt

    # ---------- Main: multiple strategies ----------
    best_adj = None
    best_cnt = 0

    # Strategy 1: Greedy from seed with different orderings
    for s in range(5):
        if not elapsed_ok():
            break
        seed_adj = make_seed()
        completed = greedy_complete(seed_adj, 1000 + s)
        cnt = count_edges_bitmask(completed)
        if cnt > best_cnt:
            best_cnt = cnt
            best_adj = completed[:]

    # Strategy 2: Local search from the best greedy result
    if elapsed_ok() and best_adj is not None:
        ls_adj, ls_cnt = local_search(best_adj[:], 42, max_iter=30000)
        if ls_cnt > best_cnt:
            best_cnt = ls_cnt
            best_adj = ls_adj[:]

    # Strategy 3: More greedy restarts with different seeds
    for s in range(10):
        if not elapsed_ok():
            break
        seed_adj = make_seed()
        # Perturb: remove a few random edges from seed
        random.seed(9999 + s)
        for _ in range(random.randint(0, 5)):
            u, v = random.sample(range(n), 2)
            if (seed_adj[u] >> v) & 1:
                seed_adj[u] &= ~(1 << v)
                seed_adj[v] &= ~(1 << u)
        completed = greedy_complete(seed_adj, 7777 + s)
        cnt = count_edges_bitmask(completed)
        if cnt > best_cnt:
            best_cnt = cnt
            best_adj = completed[:]

    # Strategy 4: Local search from empty graph greedy
    if elapsed_ok():
        empty_adj = [0] * n
        completed = greedy_complete(empty_adj, 5555)
        cnt = count_edges_bitmask(completed)
        if cnt > best_cnt:
            best_cnt = cnt
            best_adj = completed[:]
        # Local search on this
        if elapsed_ok():
            ls_adj, ls_cnt = local_search(completed[:], 123, max_iter=20000)
            if ls_cnt > best_cnt:
                best_cnt = ls_cnt
                best_adj = ls_adj[:]

    # Final local search on the best found
    if elapsed_ok() and best_adj is not None:
        ls_adj, ls_cnt = local_search(best_adj[:], 2024, max_iter=20000)
        if ls_cnt > best_cnt:
            best_cnt = ls_cnt
            best_adj = ls_adj[:]

    # Build edge list from best_adj
    if best_adj is None:
        best_adj = make_seed()

    edges = []
    for u in range(n):
        m = best_adj[u]
        for v in range(u + 1, n):
            if (m >> v) & 1:
                edges.append([u, v])

    return edges

import random
import time
import math

def construct():
    deadline = time.time() + 118  # Safety margin (evaluator kills at 150s)
    n = 40
    FULL = (1 << n) - 1

    def adj_to_edges(adj):
        edges = []
        for i in range(n):
            bits = adj[i]
            while bits:
                lsb = bits & -bits
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                bits ^= lsb
        return edges

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def creates_c6_fast(adj, u, v):
        """
        Fast C6 check: is there a path u-a-b-c-d-v of length 5?
        Uses bitmask operations for speed.
        """
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)

        if nu == 0 or nv == 0:
            return False

        # For each a in N(u), for each d in N(v)\{a},
        # check if there's a path a-b-c-d of length 3 in G-{u,v}
        bits_a = nu
        while bits_a:
            lsb_a = bits_a & -bits_a
            a = lsb_a.bit_length() - 1
            bits_a ^= lsb_a

            # d must be in N(v) but not a, and not u or v
            bits_d = nv & ~(1 << a)
            if bits_d == 0:
                continue

            # Precompute: for each d, we need b in N(a)\{u,v,a,d} and c in N(d)\{u,v,a,d}
            # such that b~c in G-{u,v}
            bits_d_cur = bits_d
            while bits_d_cur:
                lsb_d = bits_d_cur & -bits_d_cur
                d = lsb_d.bit_length() - 1
                bits_d_cur ^= lsb_d

                # b candidates: neighbors of a, excluding u, v, a, d
                b_mask = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                if b_mask == 0:
                    continue

                # c candidates: neighbors of d, excluding u, v, a, d
                c_mask = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                if c_mask == 0:
                    continue

                # For each b in b_mask, check if adj[b] intersects c_mask (excluding b itself)
                bits_b = b_mask
                while bits_b:
                    lsb_b = bits_b & -bits_b
                    b = lsb_b.bit_length() - 1
                    bits_b ^= lsb_b

                    # c must be in N(b) ∩ c_mask, c ≠ b
                    if adj[b] & c_mask & ~(1 << b):
                        return True

        return False

    def is_c6_free(adj):
        """Full validation: check that no C6 exists in the graph."""
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    if creates_c6_fast(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        return False
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return True

    def get_missing_edges(adj):
        missing = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    missing.append((u, v))
        return missing

    def get_existing_edges(adj):
        existing = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    existing.append((u, v))
        return existing

    def greedy_construct(rng):
        """Greedy edge addition in random order."""
        adj = [0] * n
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_edges)

        for u, v in all_edges:
            if not creates_c6_fast(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def local_search(adj, rng):
        """
        Greedily add valid edges until no more can be added.
        Uses multiple passes with random ordering.
        """
        improved = True
        while improved:
            improved = False
            missing = get_missing_edges(adj)
            if not missing:
                break
            rng.shuffle(missing)
            for u, v in missing:
                if not creates_c6_fast(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
        return adj

    def edge_swap_local_search(adj, rng, max_iters=50):
        """
        Try to swap one edge for another to increase edge count.
        Remove edge (u,v), then try to add a different edge (x,y) that was previously blocked.
        This can help escape local optima.
        """
        current_m = count_edges(adj)
        best_m = current_m
        best_adj = adj[:]

        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break

            existing = get_existing_edges(adj)
            if not existing:
                break

            # Pick a random edge to remove
            u, v = rng.choice(existing)
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)

            # Now try to add as many edges as possible
            adj = local_search(adj, rng)
            m = count_edges(adj)

            if m > best_m:
                best_m = m
                best_adj = adj[:]
            else:
                # Revert: restore the removed edge and re-run local search
                # Actually, just restore the edge and continue
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                adj = local_search(adj, rng)

        return best_adj

    rng = random.Random(42)

    best_adj = None
    best_m = 0

    # Phase 1: Multiple random greedy constructions
    phase1_deadline = time.time() + 8
    while time.time() < phase1_deadline:
        adj = greedy_construct(rng)
        adj = local_search(adj, rng)
        m = count_edges(adj)
        if m > best_m:
            best_m = m
            best_adj = adj[:]

    current_adj = best_adj[:]
    current_m = best_m

    # Phase 2: Iterated Local Search with varied perturbations
    while time.time() < deadline - 3:
        perturbed = current_adj[:]

        # Choose perturbation type
        pert_type = rng.random()

        if pert_type < 0.4:
            # Remove 2-6 random edges
            num_removals = rng.randint(2, 6)
            existing = get_existing_edges(perturbed)
            if existing:
                rng.shuffle(existing)
                for i in range(min(num_removals, len(existing))):
                    u, v = existing[i]
                    perturbed[u] &= ~(1 << v)
                    perturbed[v] &= ~(1 << u)

        elif pert_type < 0.7:
            # Remove 1-3 edges and add 1-2 random missing edges
            num_removals = rng.randint(1, 3)
            existing = get_existing_edges(perturbed)
            if existing:
                rng.shuffle(existing)
                for i in range(min(num_removals, len(existing))):
                    u, v = existing[i]
                    perturbed[u] &= ~(1 << v)
                    perturbed[v] &= ~(1 << u)

            missing = get_missing_edges(perturbed)
            if missing:
                num_adds = rng.randint(1, 2)
                rng.shuffle(missing)
                for i in range(min(num_adds, len(missing))):
                    u, v = missing[i]
                    if not creates_c6_fast(perturbed, u, v):
                        perturbed[u] |= (1 << v)
                        perturbed[v] |= (1 << u)

        else:
            # Remove 4-12 edges (larger perturbation)
            num_removals = rng.randint(4, 12)
            existing = get_existing_edges(perturbed)
            if existing:
                rng.shuffle(existing)
                for i in range(min(num_removals, len(existing))):
                    u, v = existing[i]
                    perturbed[u] &= ~(1 << v)
                    perturbed[v] &= ~(1 << u)

        # Local search on perturbed
        perturbed = local_search(perturbed, rng)
        m = count_edges(perturbed)

        # Acceptance criterion
        if m >= best_m:
            if m > best_m:
                best_m = m
                best_adj = perturbed[:]
            current_adj = perturbed[:]
            current_m = m
        else:
            # Accept worse solutions with some probability (simulated annealing style)
            # Probability decreases as the gap increases
            gap = best_m - m
            if gap <= 1:
                accept_prob = 0.3
            elif gap <= 2:
                accept_prob = 0.1
            else:
                accept_prob = 0.02

            if rng.random() < accept_prob:
                current_adj = perturbed[:]
                current_m = m
            # Else: keep current_adj as is (perturb from it next time)

    # Final validation
    if not is_c6_free(best_adj):
        adj = greedy_construct(rng)
        if is_c6_free(adj):
            best_adj = adj
        else:
            # Fallback: return empty graph
            return []

    return adj_to_edges(best_adj)

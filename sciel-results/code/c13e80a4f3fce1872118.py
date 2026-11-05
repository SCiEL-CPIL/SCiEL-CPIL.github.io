"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    def time_up():
        return (time.time() - start_time) > DEADLINE

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def compute_centrality_order(adj, n):
        """Use power iteration to get a vertex ordering by eigenvector centrality."""
        # Initialize with degree
        vec = [0.0] * n
        for i in range(n):
            vec[i] = float(adj[i].bit_count())
        # Normalize
        norm = sum(vec)
        if norm > 0:
            vec = [v / norm for v in vec]
        # Power iteration (a few steps is enough for ordering)
        for _ in range(10):
            new_vec = [0.0] * n
            for i in range(n):
                mask = adj[i]
                s = 0.0
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    s += vec[j]
                    mask ^= lsb
                new_vec[i] = s
            norm = sum(new_vec)
            if norm > 0:
                new_vec = [v / norm for v in new_vec]
            vec = new_vec
        # Return vertices sorted by centrality descending
        order = list(range(n))
        order.sort(key=lambda v: vec[v], reverse=True)
        return order

    def max_clique_bk(adj, n, active_mask, vertex_order):
        """Bron-Kerbosch with coloring to find max clique in subgraph induced by active_mask."""
        if active_mask == 0:
            return []

        best_size = [0]
        best_clique = []

        # Get list of active vertices sorted by centrality
        verts = []
        m = active_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            verts.append(idx)
            m ^= lsb

        if not verts:
            return []

        # Use centrality ordering to sort vertices
        order_map = {v: i for i, v in enumerate(vertex_order)}
        verts.sort(key=lambda v: order_map.get(v, n))

        def color_sort(P_mask):
            """Greedy coloring of vertices in P_mask, return list of (color, vertex)."""
            pm = P_mask
            verts_p = []
            while pm:
                lsb = pm & -pm
                idx = lsb.bit_length() - 1
                verts_p.append(idx)
                pm ^= lsb

            if not verts_p:
                return []

            # Sort by degree in P_mask descending
            verts_p.sort(key=lambda v: (adj[v] & P_mask).bit_count(), reverse=True)

            v_color = {}
            result = []
            for v in verts_p:
                used = set()
                neigh = adj[v] & P_mask
                nm = neigh
                while nm:
                    lsb = nm & -nm
                    u = lsb.bit_length() - 1
                    if u in v_color:
                        used.add(v_color[u])
                    nm ^= lsb
                c = 1
                while c in used:
                    c += 1
                v_color[v] = c
                result.append((c, v))

            result.sort(key=lambda x: x[0])
            return result

        def bk(R, P, X):
            nonlocal best_size, best_clique

            if P == 0 and X == 0:
                if len(R) > best_size[0]:
                    best_size[0] = len(R)
                    best_clique = list(R)
                return

            if time_up():
                return

            colored = color_sort(P)
            for color, v in reversed(colored):
                if len(R) + color <= best_size[0]:
                    break

                R.append(v)
                bk(R, P & adj[v], X & adj[v])
                R.pop()
                P ^= (1 << v)
                X |= (1 << v)

                if time_up():
                    return

        bk([], active_mask, 0)
        return best_clique

    def solve_instance(n, p, seed):
        adj = generate_graph(n, p, seed)
        rem = adj[:]  # remaining edges as bitmasks
        cliques = []

        # Compute centrality ordering once
        vertex_order = compute_centrality_order(adj, n)

        # Compute initial active mask
        active = 0
        for i in range(n):
            if rem[i]:
                active |= (1 << i)

        while active and not time_up():
            # Find max clique in remaining graph
            clique = max_clique_bk(rem, n, active, vertex_order)

            if not clique:
                # Fallback: find any edge
                found = False
                m = active
                while m and not found:
                    lsb = m & -m
                    u = lsb.bit_length() - 1
                    nb = rem[u] & active
                    if nb:
                        lsb2 = nb & -nb
                        v = lsb2.bit_length() - 1
                        clique = [u, v]
                        found = True
                    m ^= lsb
                if not found:
                    break

            cliques.append(clique)

            # Remove clique edges from rem
            for u in clique:
                for v in clique:
                    if u != v:
                        rem[u] &= ~(1 << v)

            # Update active mask
            active = 0
            for i in range(n):
                if rem[i]:
                    active |= (1 << i)

        # Fallback: cover remaining edges as pairs
        if active:
            for u in range(n):
                nb = rem[u]
                while nb:
                    lsb = nb & -nb
                    v = lsb.bit_length() - 1
                    if u < v:
                        cliques.append([u, v])
                        rem[u] &= ~(1 << v)
                        rem[v] &= ~(1 << u)
                    nb ^= lsb

        return cliques

    c1 = solve_instance(32, 0.6, 601)
    c2 = solve_instance(64, 0.55, 602)

    return [c1, c2]
# EVOLVE-BLOCK-END

<<<<<<< SEARCH
def _max_independent_set_size_fast(adj, n, limit=10):
    """
    Fast exact max independent set using bitset branch and bound.
    Returns min(actual_alpha, limit) - i.e., if alpha >= limit, returns limit.
    """
    best = [0]
    full_mask = (1 << n) - 1
    adj_local = adj

    def _greedy_color_bound(available):
        """Upper bound: greedy clique cover of G[available]."""
        unassigned = available
        k = 0
        while unassigned:
            k += 1
            lsb = unassigned & -unassigned
            v = lsb.bit_length() - 1
            unassigned ^= lsb
            # Build a clique containing v in G[available]
            clique = 1 << v
            remaining = unassigned & adj_local[v]
            while remaining:
                lsb_u = remaining & -remaining
                u = lsb_u.bit_length() - 1
                remaining ^= lsb_u
                clique |= lsb_u
                remaining &= adj_local[u]
            unassigned &= ~clique
        return k

    def _bnb(available, current_size):
        if not available:
            if current_size > best[0]:
                best[0] = current_size
            return

        if current_size + _bit_count(available) <= best[0]:
            return

        bound = _greedy_color_bound(available)
        if current_size + bound <= best[0]:
            return

        # Pick vertex with max degree in G[available]
        v_mask = available
        max_deg = -1
        v = -1
        while v_mask:
            lsb = v_mask & -v_mask
            u = lsb.bit_length() - 1
            v_mask ^= lsb
            deg = _bit_count(adj_local[u] & available)
            if deg > max_deg:
                max_deg = deg
                v = u

        if v == -1:
            return

        # Branch 1: Include v
        _bnb(available & ~(1 << v) & ~adj_local[v], current_size + 1)

        # Branch 2: Exclude v
        _bnb(available & ~(1 << v), current_size)

    _bnb(full_mask, 0)
    return min(best[0], limit)
=======
def _max_independent_set_size_fast(adj, n, limit=10):
    """
    Fast exact max independent set using bitset branch and bound.
    Returns min(actual_alpha, limit) - i.e., if alpha >= limit, returns limit.
    """
    best = [0]
    full_mask = (1 << n) - 1
    adj_local = adj

    def _color_bound(available):
        """
        Upper bound on independence number via greedy coloring of G[available].
        Each color class is an independent set in G = clique in complement.
        We greedily partition available vertices into cliques of the complement
        (i.e., independent sets of G), which gives an upper bound on alpha.
        """
        unassigned = available
        k = 0
        while unassigned:
            k += 1
            # Greedily build a clique in complement = independent set in G
            # Pick first vertex, then add vertices not adjacent in G (adjacent in complement)
            lsb = unassigned & -unassigned
            v = lsb.bit_length() - 1
            unassigned ^= lsb
            # Vertices in unassigned that are NOT adjacent to v in G (can be in same color class)
            non_neighbors = unassigned & ~adj_local[v]
            # Greedily fill the color class
            color_class = 1 << v
            while non_neighbors:
                lsb_u = non_neighbors & -non_neighbors
                u = lsb_u.bit_length() - 1
                non_neighbors ^= lsb_u
                color_class |= lsb_u
                # Remove u's G-neighbors from remaining candidates
                non_neighbors &= ~adj_local[u]
            unassigned &= ~color_class
        return k

    def _bnb(available, current_size):
        if not available:
            if current_size > best[0]:
                best[0] = current_size
            return

        if current_size + _bit_count(available) <= best[0]:
            return

        bound = _color_bound(available)
        if current_size + bound <= best[0]:
            return

        # Pick vertex with max degree in G[available] for better pruning
        v_mask = available
        max_deg = -1
        v = -1
        while v_mask:
            lsb = v_mask & -v_mask
            u = lsb.bit_length() - 1
            v_mask ^= lsb
            deg = _bit_count(adj_local[u] & available)
            if deg > max_deg:
                max_deg = deg
                v = u

        if v == -1:
            return

        # Branch 1: Include v in independent set
        _bnb(available & ~(1 << v) & ~adj_local[v], current_size + 1)

        # Branch 2: Exclude v
        _bnb(available & ~(1 << v), current_size)

    _bnb(full_mask, 0)
    return min(best[0], limit)
>>>>>>> REPLACE


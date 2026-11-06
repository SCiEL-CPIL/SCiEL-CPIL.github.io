import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Leave 10s margin for safety

    best_n = 37
    best_edges = None

    # Seed graph: Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    seed_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    best_edges = seed_edges

    def _find_clique_bitmask(n, adj_masks, k):
        """Find a clique of size k using bitmask operations. Returns list of vertices or None."""
        if k == 0:
            return []
        if k == 1:
            return [0]
        if k == 2:
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj_masks[i] >> j) & 1:
                        return [i, j]
            return None
        if k == 3:
            for i in range(n):
                mask_i = adj_masks[i]
                mask_j_candidates = mask_i
                while mask_j_candidates:
                    j = (mask_j_candidates & -mask_j_candidates).bit_length() - 1
                    mask_j_candidates &= mask_j_candidates - 1
                    mask_k_candidates = mask_i & adj_masks[j]
                    if mask_k_candidates:
                        k = (mask_k_candidates & -mask_k_candidates).bit_length() - 1
                        return [i, j, k]
            return None
        if k == 4:
            for i in range(n):
                mask_i = adj_masks[i]
                mask_j_candidates = mask_i
                while mask_j_candidates:
                    j = (mask_j_candidates & -mask_j_candidates).bit_length() - 1
                    mask_j_candidates &= mask_j_candidates - 1
                    common_ij = mask_i & adj_masks[j]
                    mask_k_candidates = common_ij
                    while mask_k_candidates:
                        k = (mask_k_candidates & -mask_k_candidates).bit_length() - 1
                        mask_k_candidates &= mask_k_candidates - 1
                        common_ijk = common_ij & adj_masks[k]
                        if common_ijk:
                            l = (common_ijk & -common_ijk).bit_length() - 1
                            return [i, j, k, l]
            return None
        if k == 5:
            for v in range(n):
                mask_v = adj_masks[v]
                mask_u_candidates = mask_v
                while mask_u_candidates:
                    u = (mask_u_candidates & -mask_u_candidates).bit_length() - 1
                    mask_u_candidates &= mask_u_candidates - 1
                    common_vu = mask_v & adj_masks[u]
                    mask_w_candidates = common_vu
                    while mask_w_candidates:
                        w = (mask_w_candidates & -mask_w_candidates).bit_length() - 1
                        mask_w_candidates &= mask_w_candidates - 1
                        common_vuw = common_vu & adj_masks[w]
                        mask_x_candidates = common_vuw
                        while mask_x_candidates:
                            x = (mask_x_candidates & -mask_x_candidates).bit_length() - 1
                            mask_x_candidates &= mask_x_candidates - 1
                            common_vuwx = common_vuw & adj_masks[x]
                            if common_vuwx:
                                y = (common_vuwx & -common_vuwx).bit_length() - 1
                                return [v, u, w, x, y]
            return None
        return None

    def _find_independent_set_bitmask(n, adj_masks, k):
        """Find an independent set of size k (clique in complement)."""
        comp_masks = [0] * n
        full_mask = (1 << n) - 1
        for i in range(n):
            comp_masks[i] = full_mask & ~adj_masks[i] & ~(1 << i)
        return _find_clique_bitmask(n, comp_masks, k)

    def _build_adj_masks(n, edges):
        adj_masks = [0] * n
        for u, v in edges:
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        return adj_masks

    def _is_valid(n, adj_masks):
        """Check if graph is (5,5)-free."""
        if _find_clique_bitmask(n, adj_masks, 5):
            return False
        if _find_independent_set_bitmask(n, adj_masks, 5):
            return False
        return True

    # Try to find valid graphs for n from 45 down to 38
    for n in range(45, 37, -1):
        if time.time() > deadline:
            break

        # Multiple restarts with different strategies
        for restart in range(200):
            if time.time() > deadline:
                break

            # Choose edge probability based on restart
            # For (5,5)-free, we want density around 0.5
            # But we can try different densities
            if restart % 3 == 0:
                p = 0.5
            elif restart % 3 == 1:
                p = 0.45
            else:
                p = 0.55

            edges = set()
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        edges.add((i, j))

            adj_masks = _build_adj_masks(n, edges)

            # Repair loop
            for step in range(1000):
                if time.time() > deadline:
                    break

                # Check for K5
                k5 = _find_clique_bitmask(n, adj_masks, 5)
                if k5:
                    # Remove a random edge from the clique
                    i, j = random.sample(k5, 2)
                    if i > j:
                        i, j = j, i
                    if (i, j) in edges:
                        edges.remove((i, j))
                        adj_masks[i] &= ~(1 << j)
                        adj_masks[j] &= ~(1 << i)
                    continue

                # Check for independent set of size 5
                indep = _find_independent_set_bitmask(n, adj_masks, 5)
                if indep:
                    # Add a random edge within the independent set
                    i, j = random.sample(indep, 2)
                    if i > j:
                        i, j = j, i
                    if (i, j) not in edges:
                        edges.add((i, j))
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
                    continue

                # No violations found - valid graph
                if n > best_n:
                    best_n = n
                    best_edges = list(edges)
                break

            if best_n >= n:
                break

    return best_n, best_edges

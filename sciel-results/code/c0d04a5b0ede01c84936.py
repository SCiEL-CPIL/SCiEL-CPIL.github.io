import time

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 85

    # Bitmask representation for fast adjacency
    def is_valid_mask(adj_mask, deg):
        # For each edge (u,v), check if there exist s in N(u)\{v} and t in N(v)\{u} with s!=t and deg[s]==deg[t]
        # Group vertices by degree
        deg_groups = {}
        for i in range(n):
            deg_groups.setdefault(deg[i], []).append(i)
        
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_mask[u] >> v) & 1:
                    continue
                # s must be neighbor of u, not v, not equal to v
                s_candidates = adj_mask[u] & ~(1 << v) & ~((1 << n) - 1)
                # t must be neighbor of v, not u, not equal to u
                t_candidates = adj_mask[v] & ~(1 << u) & ~((1 << n) - 1)
                # Also s != t (handled by checking pairs)
                # For each degree class, check if both s_candidates and t_candidates have vertices in that class
                for d, verts in deg_groups.items():
                    mask_d = 0
                    for x in verts:
                        mask_d |= (1 << x)
                    s_in_d = s_candidates & mask_d
                    t_in_d = t_candidates & mask_d
                    if s_in_d and t_in_d:
                        # Need s != t: if s_in_d has only one bit and it's the same as the only bit in t_in_d, skip
                        if s_in_d == t_in_d and (s_in_d & (s_in_d - 1)) == 0:
                            continue
                        return False
        return True

    def get_degrees_mask(adj_mask):
        return [bin(adj_mask[i]).count('1') for i in range(n)]

    def count_edges_mask(adj_mask):
        return sum(bin(adj_mask[i]).count('1') for i in range(n)) // 2

    # Start with K_{7,8}
    part_a = list(range(7))
    part_b = list(range(7, 15))

    adj_mask = [0] * n
    for u in part_a:
        for v in part_b:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)

    best_adj = adj_mask[:]
    best_deg = get_degrees_mask(best_adj)
    best_edges = count_edges_mask(best_adj)

    # All possible edges
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]

    # Greedy: try adding edges, ordered by potential degree-difference benefit
    for _ in range(300):
        if time.time() > deadline:
            break
        improved = False
        for (u, v) in all_pairs:
            if time.time() > deadline:
                break
            if (best_adj[u] >> v) & 1:
                continue
            # Try adding edge (u, v)
            best_adj[u] |= (1 << v)
            best_adj[v] |= (1 << u)
            new_deg = get_degrees_mask(best_adj)
            if is_valid_mask(best_adj, new_deg):
                best_deg = new_deg
                best_edges = count_edges_mask(best_adj)
                improved = True
            else:
                best_adj[u] &= ~(1 << v)
                best_adj[v] &= ~(1 << u)
        if not improved:
            break

    # Swap search: remove one edge, add another
    for _ in range(30):
        if time.time() > deadline:
            break
        improved = False
        for i, (u1, v1) in enumerate(all_pairs):
            if time.time() > deadline:
                break
            if not (best_adj[u1] >> v1) & 1:
                continue
            for (u2, v2) in all_pairs[i + 1:]:
                if time.time() > deadline:
                    break
                if (best_adj[u2] >> v2) & 1:
                    continue
                # Try swap
                best_adj[u1] &= ~(1 << v1)
                best_adj[v1] &= ~(1 << u1)
                best_adj[u2] |= (1 << v2)
                best_adj[v2] |= (1 << u2)
                new_deg = get_degrees_mask(best_adj)
                new_edges = count_edges_mask(best_adj)
                if new_edges > best_edges and is_valid_mask(best_adj, new_deg):
                    best_edges = new_edges
                    improved = True
                else:
                    # Revert
                    best_adj[u1] |= (1 << v1)
                    best_adj[v1] |= (1 << u1)
                    best_adj[u2] &= ~(1 << v2)
                    best_adj[v2] &= ~(1 << u2)
        if not improved:
            break

    # Try 2-swap: remove two edges, add two edges
    for _ in range(10):
        if time.time() > deadline:
            break
        improved = False
        present = [(u, v) for (u, v) in all_pairs if (best_adj[u] >> v) & 1]
        absent = [(u, v) for (u, v) in all_pairs if not (best_adj[u] >> v) & 1]
        for i in range(len(present)):
            if time.time() > deadline:
                break
            (u1, v1) = present[i]
            for j in range(i + 1, len(present)):
                if time.time() > deadline:
                    break
                (u2, v2) = present[j]
                for a in range(len(absent)):
                    if time.time() > deadline:
                        break
                    (u3, v3) = absent[a]
                    for b in range(a + 1, len(absent)):
                        if time.time() > deadline:
                            break
                        (u4, v4) = absent[b]
                        # Remove (u1,v1) and (u2,v2), add (u3,v3) and (u4,v4)
                        best_adj[u1] &= ~(1 << v1)
                        best_adj[v1] &= ~(1 << u1)
                        best_adj[u2] &= ~(1 << v2)
                        best_adj[v2] &= ~(1 << u2)
                        best_adj[u3] |= (1 << v3)
                        best_adj[v3] |= (1 << u3)
                        best_adj[u4] |= (1 << v4)
                        best_adj[v4] |= (1 << u4)
                        new_deg = get_degrees_mask(best_adj)
                        new_edges = count_edges_mask(best_adj)
                        if new_edges > best_edges and is_valid_mask(best_adj, new_deg):
                            best_edges = new_edges
                            improved = True
                        else:
                            # Revert
                            best_adj[u1] |= (1 << v1)
                            best_adj[v1] |= (1 << u1)
                            best_adj[u2] |= (1 << v2)
                            best_adj[v2] |= (1 << u2)
                            best_adj[u3] &= ~(1 << v3)
                            best_adj[v3] &= ~(1 << u3)
                            best_adj[u4] &= ~(1 << v4)
                            best_adj[v4] &= ~(1 << u4)
                if improved:
                    # Rebuild present/absent lists
                    present = [(u, v) for (u, v) in all_pairs if (best_adj[u] >> v) & 1]
                    absent = [(u, v) for (u, v) in all_pairs if not (best_adj[u] >> v) & 1]
                    break
            if improved:
                present = [(u, v) for (u, v) in all_pairs if (best_adj[u] >> v) & 1]
                absent = [(u, v) for (u, v) in all_pairs if not (best_adj[u] >> v) & 1]
                break
        if not improved:
            break

    # Build result
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])

    return edges

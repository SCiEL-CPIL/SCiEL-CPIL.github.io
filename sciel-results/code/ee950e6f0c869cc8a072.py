import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110

    # Precompute all 24 permutations of [0,1,2,3]
    perms24 = list(itertools.permutations(range(4)))

    # Precompute the 70 ways to split 8 positions into two sets of 4
    # For each, store (A_indices, B_indices)
    splits8 = []
    for A_idx in itertools.combinations(range(8), 4):
        B_idx = tuple(i for i in range(8) if i not in A_idx)
        splits8.append((A_idx, B_idx))

    # For a given 8-vertex tuple (v0..v7) and a split (A_idx, B_idx),
    # and a permutation p of B positions, check if Q3 pattern matches.
    # Pattern: A[i] must connect to specific B[p[j]] positions.
    # Required connections (row in A, col in B):
    # A[0]: B[0], B[1], B[3]
    # A[1]: B[0], B[1], B[2]
    # A[2]: B[0], B[2], B[3]
    # A[3]: B[1], B[2], B[3]
    #
    # For permutation p, A[i] must connect to B[p[0]], B[p[1]], B[p[3]] etc.
    # We can precompute for each permutation p, which B-indices each A-row needs.

    # Precompute: for each perm p, the set of (a_row, b_col) pairs needed
    # a_row is 0-3, b_col is the actual index in B (0-3)
    # Pattern edges: (0,0),(0,1),(0,3),(1,0),(1,1),(1,2),(2,0),(2,2),(2,3),(3,1),(3,2),(3,3)
    pattern_edges = [(0,0),(0,1),(0,3),(1,0),(1,1),(1,2),(2,0),(2,2),(2,3),(3,1),(3,2),(3,3)]

    # For each permutation p, the required (a_row, b_actual_idx) pairs
    # where b_actual_idx = p[b_pattern_idx]
    perm_required = []
    for p in perms24:
        req = []
        for (ar, bp) in pattern_edges:
            req.append((ar, p[bp]))
        perm_required.append(tuple(sorted(req)))

    # Initial graph: K_{3,13}
    edges = set()
    for u in range(3):
        for v in range(3, 16):
            edges.add((u, v))

    adj_mask = [0] * 16
    for (u, v) in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    def contains_q3_fast(adj_mask, verts):
        """Check if 8 vertices contain Q3. verts is a tuple of 8 vertex ids."""
        # Try all 70 splits and 24 perms for each orientation
        for (A_idx, B_idx) in splits8:
            A = [verts[i] for i in A_idx]
            B = [verts[i] for i in B_idx]

            # Orientation 1: A = even parity, B = odd parity
            for req in perm_required:
                ok = True
                for (ar, bi) in req:
                    if not (adj_mask[A[ar]] >> B[bi]) & 1:
                        ok = False
                        break
                if ok:
                    return True

            # Orientation 2: A = odd parity, B = even parity
            # Swap roles: B rows, A cols
            for req in perm_required:
                ok = True
                for (ar, bi) in req:
                    if not (adj_mask[B[ar]] >> A[bi]) & 1:
                        ok = False
                        break
                if ok:
                    return True

        return False

    # Precompute all 8-vertex subsets containing a given edge (u,v)
    # There are C(14,6) = 3003 such subsets
    all_8_with_edge = {}
    for (u, v) in itertools.combinations(range(16), 2):
        remaining = [x for x in range(16) if x != u and x != v]
        all_8_with_edge[(u, v)] = list(itertools.combinations(remaining, 6))

    def is_q3_free(adj_mask):
        """Check entire graph for Q3. Only needed for final verification."""
        for combo in itertools.combinations(range(16), 8):
            if contains_q3_fast(adj_mask, combo):
                return False
        return True

    # Local search: try adding, removing, or swapping edges
    all_pairs = set(itertools.combinations(range(16), 2))
    current_edges = set(edges)
    current_adj = adj_mask[:]
    best_edges = set(current_edges)
    best_adj = current_adj[:]
    best_count = len(current_edges)

    def try_add_edge(adj_mask, edge_set, u, v):
        """Try adding edge (u,v). Return True if safe."""
        edge_set.add((u, v))
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

        for combo in all_8_with_edge[(u, v)]:
            verts = (u, v) + combo
            if contains_q3_fast(adj_mask, verts):
                edge_set.remove((u, v))
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
                return False
        return True

    def try_remove_edge(adj_mask, edge_set, u, v):
        """Remove edge (u,v). Always safe."""
        edge_set.remove((u, v))
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)

    # Phase 1: Greedy addition
    non_edges = list(all_pairs - current_edges)

    for _round in range(50):
        if time.time() > deadline - 30:
            break
        changed = False
        # Sort by degree sum, descending
        non_edges.sort(key=lambda e: bin(current_adj[e[0]]).count('1') + bin(current_adj[e[1]]).count('1'), reverse=True)
        for e in non_edges:
            if time.time() > deadline - 30:
                break
            if e in current_edges:
                continue
            u, v = e
            if try_add_edge(current_adj, current_edges, u, v):
                changed = True
                if len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = set(current_edges)
                    best_adj = current_adj[:]
        if not changed:
            break

    # Phase 2: Local search with add/remove/swap
    random.seed(42)
    no_improve_count = 0
    max_no_improve = 200

    while time.time() < deadline - 5 and no_improve_count < max_no_improve:
        # Pick a random move type
        move_type = random.random()
        improved = False

        if move_type < 0.5:
            # Try adding a random non-edge
            non_edges_now = [e for e in all_pairs if e not in current_edges]
            if not non_edges_now:
                no_improve_count += 1
                continue
            # Weight by degree sum
            weights = [bin(current_adj[e[0]]).count('1') + bin(current_adj[e[1]]).count('1') + 1 for e in non_edges_now]
            e = random.choices(non_edges_now, weights=weights, k=1)[0]
            u, v = e
            if try_add_edge(current_adj, current_edges, u, v):
                if len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = set(current_edges)
                    best_adj = current_adj[:]
                    improved = True
                else:
                    # Accept with some probability (simulated annealing-like)
                    if random.random() < 0.1:
                        improved = True
                    else:
                        try_remove_edge(current_adj, current_edges, u, v)
            no_improve_count += 0 if improved else 1

        elif move_type < 0.8:
            # Try removing a random edge (to enable other additions)
            if not current_edges:
                no_improve_count += 1
                continue
            e = random.choice(list(current_edges))
            u, v = e
            try_remove_edge(current_adj, current_edges, u, v)
            # Try to add back something
            non_edges_now = [x for x in all_pairs if x not in current_edges]
            if non_edges_now:
                weights = [bin(current_adj[x[0]]).count('1') + bin(current_adj[x[1]]).count('1') + 1 for x in non_edges_now]
                for _ in range(3):
                    e2 = random.choices(non_edges_now, weights=weights, k=1)[0]
                    if try_add_edge(current_adj, current_edges, e2[0], e2[1]):
                        break
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = set(current_edges)
                best_adj = current_adj[:]
                improved = True
            no_improve_count += 0 if improved else 1

        else:
            # Swap: remove one edge, add another
            if not current_edges:
                no_improve_count += 1
                continue
            e_remove = random.choice(list(current_edges))
            u1, v1 = e_remove
            try_remove_edge(current_adj, current_edges, u1, v1)

            non_edges_now = [x for x in all_pairs if x not in current_edges]
            if non_edges_now:
                weights = [bin(current_adj[x[0]]).count('1') + bin(current_adj[x[1]]).count('1') + 1 for x in non_edges_now]
                e_add = random.choices(non_edges_now, weights=weights, k=1)[0]
                if try_add_edge(current_adj, current_edges, e_add[0], e_add[1]):
                    if len(current_edges) > best_count:
                        best_count = len(current_edges)
                        best_edges = set(current_edges)
                        best_adj = current_adj[:]
                        improved = True
                else:
                    # Re-add the removed edge
                    current_edges.add((u1, v1))
                    current_adj[u1] |= (1 << v1)
                    current_adj[v1] |= (1 << u1)
            else:
                # Re-add
                current_edges.add((u1, v1))
                current_adj[u1] |= (1 << v1)
                current_adj[v1] |= (1 << u1)

            no_improve_count += 0 if improved else 1

    # Final: try greedy addition on best solution
    final_edges = set(best_edges)
    final_adj = best_adj[:]
    non_edges_final = [e for e in all_pairs if e not in final_edges]
    non_edges_final.sort(key=lambda e: bin(final_adj[e[0]]).count('1') + bin(final_adj[e[1]]).count('1'), reverse=True)

    for e in non_edges_final:
        if time.time() > deadline - 2:
            break
        u, v = e
        if try_add_edge(final_adj, final_edges, u, v):
            pass

    return [list(e) for e in final_edges]

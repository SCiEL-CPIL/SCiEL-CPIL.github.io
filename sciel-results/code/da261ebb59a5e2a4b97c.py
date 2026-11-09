import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110

    # Precompute all 24 permutations of [0,1,2,3]
    perms24 = list(itertools.permutations(range(4)))

    # Precompute the 70 ways to split 8 positions into two sets of 4
    splits8 = []
    for A_idx in itertools.combinations(range(8), 4):
        B_idx = tuple(i for i in range(8) if i not in A_idx)
        splits8.append((A_idx, B_idx))

    # Pattern edges for Q3 bipartite: (a_row, b_col) pairs
    pattern_edges = [(0,0),(0,1),(0,3),(1,0),(1,1),(1,2),(2,0),(2,2),(2,3),(3,1),(3,2),(3,3)]

    # For each permutation p, the required (a_row, b_actual_idx) pairs
    perm_required = []
    for p in perms24:
        req = []
        for (ar, bp) in pattern_edges:
            req.append((ar, p[bp]))
        perm_required.append(tuple(sorted(req)))

    def contains_q3_fast(adj_mask, verts):
        """Check if 8 vertices contain Q3. verts is a tuple of 8 vertex ids."""
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
    all_8_with_edge = {}
    for (u, v) in itertools.combinations(range(16), 2):
        remaining = [x for x in range(16) if x != u and x != v]
        all_8_with_edge[(u, v)] = list(itertools.combinations(remaining, 6))

    all_pairs = set(itertools.combinations(range(16), 2))

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

    # Strategy: Start with K_{8,8} (bipartite between {0..7} and {8..15}),
    # then greedily add intra-partition edges and some inter-partition edges.
    #
    # Phase 1: Build K_{8,8} and verify it's Q3-free (it is, since Q3 is bipartite
    # with parts of size 4+4, and K_{8,8} has no 3-regular subgraph on 8 vertices
    # that forms Q3... actually we need to check).
    #
    # Actually, K_{8,8} DOES contain Q3 as a subgraph! We need to be more careful.
    #
    # New strategy: Start with a known Q3-free structure and build up.
    # Use the greedy approach but with better initialization.

    # Initialize with K_{3,13} which is Q3-free
    edges = set()
    for u in range(3):
        for v in range(3, 16):
            edges.add((u, v))

    adj_mask = [0] * 16
    for (u, v) in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    current_edges = set(edges)
    current_adj = adj_mask[:]
    best_edges = set(current_edges)
    best_adj = current_adj[:]
    best_count = len(current_edges)

    # Phase 1: Greedy addition - try all non-edges sorted by degree sum
    non_edges = list(all_pairs - current_edges)

    for _round in range(100):
        if time.time() > deadline - 30:
            break
        changed = False
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

    # Phase 2: Simulated annealing-style local search
    random.seed(42)
    no_improve_count = 0
    max_no_improve = 300
    temperature = 1.0
    cooling = 0.995

    while time.time() < deadline - 5 and no_improve_count < max_no_improve:
        move_type = random.random()
        improved = False

        if move_type < 0.5:
            # Try adding a random non-edge
            non_edges_now = [e for e in all_pairs if e not in current_edges]
            if not non_edges_now:
                no_improve_count += 1
                continue
            weights = [bin(current_adj[e[0]]).count('1') + bin(current_adj[e[1]]).count('1') + 1 for e in non_edges_now]
            e = random.choices(non_edges_now, weights=weights, k=1)[0]
            u, v = e
            if try_add_edge(current_adj, current_edges, u, v):
                if len(current_edges) > best_count:
                    best_count = len(current_edges)
                    best_edges = set(current_edges)
                    best_adj = current_adj[:]
                    improved = True
                    no_improve_count = 0
                else:
                    # Accept with probability based on temperature (always accept since edge count doesn't decrease)
                    improved = True
                    no_improve_count = 0
            else:
                no_improve_count += 1

        elif move_type < 0.8:
            # Try removing a random edge then adding back
            if not current_edges:
                no_improve_count += 1
                continue
            e = random.choice(list(current_edges))
            u, v = e
            try_remove_edge(current_adj, current_edges, u, v)
            # Try to add back something different
            non_edges_now = [x for x in all_pairs if x not in current_edges]
            if non_edges_now:
                weights = [bin(current_adj[x[0]]).count('1') + bin(current_adj[x[1]]).count('1') + 1 for x in non_edges_now]
                for _ in range(5):
                    e2 = random.choices(non_edges_now, weights=weights, k=1)[0]
                    if try_add_edge(current_adj, current_edges, e2[0], e2[1]):
                        break
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = set(current_edges)
                best_adj = current_adj[:]
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1

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
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    # Re-add the removed edge
                    current_edges.add((u1, v1))
                    current_adj[u1] |= (1 << v1)
                    current_adj[v1] |= (1 << u1)
                    no_improve_count += 1
            else:
                current_edges.add((u1, v1))
                current_adj[u1] |= (1 << v1)
                current_adj[v1] |= (1 << u1)
                no_improve_count += 1

        temperature *= cooling

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

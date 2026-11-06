import random
import time
from itertools import combinations

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0

    # --- Fast K3,3 check ---
    def has_k33(adj):
        """Check if graph with bitmask adj[] contains K3,3.
        K3,3 exists iff there exist 3 vertices u,v,w with >=3 common neighbours.
        """
        for u in range(n):
            au = adj[u]
            for v in range(u + 1, n):
                common_uv = au & adj[v]
                if common_uv.bit_count() < 3:
                    continue
                for w in range(v + 1, n):
                    if (common_uv & adj[w]).bit_count() >= 3:
                        return True
        return False

    def is_add_valid(adj, a, b):
        """Check if adding edge (a,b) keeps graph K3,3-free.
        Only need to check triples that include a or b, since only those change.
        Optimisation: only pairs (x,y) that are both neighbours of b (for triples with a)
        or both neighbours of a (for triples with b) can create a new common neighbour.
        """
        na = adj[a]
        nb = adj[b]

        # Triples (a, x, y): new common neighbour is b iff x,y in N(b)
        # Old common count must be < 3; new count = old + 1 if b in N(x)∩N(y)
        # So we only need to check pairs (x,y) in N(b)
        # and verify that (na & adj[x] & adj[y]).bit_count() >= 2 (so new count >= 3)
        nb_set = nb & ~(1 << a)  # neighbours of b, excluding a
        nb_list = []
        m = nb_set
        while m:
            lsb = m & -m
            nb_list.append(lsb.bit_length() - 1)
            m -= lsb

        for i in range(len(nb_list)):
            x = nb_list[i]
            ax = adj[x]
            for j in range(i + 1, len(nb_list)):
                y = nb_list[j]
                ay = adj[y]
                # Common neighbours of a, x, y before adding (a,b)
                # b is NOT yet a neighbour of a, so common = na & ax & ay
                old_common = (na & ax & ay).bit_count()
                # After adding (a,b), b becomes a common neighbour of (a,x,y)
                # since b in N(a) (new), b in N(x), b in N(y)
                # So new count = old_common + 1
                if old_common >= 2:
                    return False

        # Triples (b, x, y): new common neighbour is a iff x,y in N(a)
        na_set = na & ~(1 << b)
        na_list = []
        m = na_set
        while m:
            lsb = m & -m
            na_list.append(lsb.bit_length() - 1)
            m -= lsb

        for i in range(len(na_list)):
            x = na_list[i]
            ax = adj[x]
            for j in range(i + 1, len(na_list)):
                y = na_list[j]
                ay = adj[y]
                old_common = (nb & ax & ay).bit_count()
                if old_common >= 2:
                    return False

        return True

    def edge_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def get_non_edges(adj):
        non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    non_edges.append((u, v))
        return non_edges

    def get_all_edges(adj):
        all_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    all_edges.append((u, v))
        return all_edges

    def greedy_add(adj, rng):
        """Exhaustively add any valid non-edge."""
        changed = True
        while changed:
            changed = False
            non_edges = get_non_edges(adj)
            rng.shuffle(non_edges)
            for u, v in non_edges:
                if is_add_valid(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    changed = True
        return adj

    # --- Deterministic strong starting point: 5x4 grid construction ---
    def grid_start():
        """
        Construct a K3,3-free graph using a 5x4 grid structure.
        Vertices 0..19 arranged as 5 rows x 4 columns.
        Connect vertices in the same row or same column.
        This gives a grid graph which is K3,3-free for small grids.
        Actually, let's use a different construction:
        Take 5 groups of 4 vertices. Connect all vertices between groups except
        a specific pattern to avoid K3,3.
        
        Better: Use the complement of a 5-partite construction.
        Let's use: vertices partitioned into 5 sets of 4.
        Within each set: no edges (independent set).
        Between sets: connect all except a 1-factor pattern.
        This is a complete 5-partite graph minus a matching-like structure.
        
        Actually, let's just start with a random dense graph and let local search do the work.
        The key insight from the inspiration is that the fast is_add_valid enables more search.
        """
        rng = random.Random(42)
        adj = [0] * n
        # Start with a moderately dense random graph
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < 0.5:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        # Remove K3,3 violations by removing edges
        # This is a repair step
        for _ in range(100):
            if not has_k33(adj):
                break
            # Find a K3,3 and remove a random edge from it
            found = False
            for u in range(n):
                for v in range(u + 1, n):
                    common_uv = adj[u] & adj[v]
                    if common_uv.bit_count() < 3:
                        continue
                    for w in range(v + 1, n):
                        common_uvw = common_uv & adj[w]
                        if common_uvw.bit_count() >= 3:
                            # Found K3,3: {u,v,w} with >=3 common neighbours
                            # Remove a random edge incident to one of u,v,w
                            candidates = []
                            for z in [u, v, w]:
                                for nb in range(n):
                                    if z != nb and (adj[z] & (1 << nb)):
                                        candidates.append((z, nb))
                            if candidates:
                                a, b = rng.choice(candidates)
                                adj[a] &= ~(1 << b)
                                adj[b] &= ~(1 << a)
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
        return adj

    def local_search(adj, rng):
        """Local search: greedy add + swap moves to escape local optima."""
        adj = adj[:]
        current_edges = edge_count(adj)
        best_local_edges = current_edges
        best_local_adj = adj[:]

        # First, greedily add valid edges
        adj = greedy_add(adj, rng)
        current_edges = edge_count(adj)
        if current_edges > best_local_edges:
            best_local_edges = current_edges
            best_local_adj = adj[:]

        no_improve = 0
        while time.time() < deadline - 5:
            moved = False

            # Try to add edges: exhaustive pass
            non_edges = get_non_edges(adj)
            if non_edges:
                rng.shuffle(non_edges)
                for u, v in non_edges:
                    if is_add_valid(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        current_edges += 1
                        moved = True
                        if current_edges > best_local_edges:
                            best_local_edges = current_edges
                            best_local_adj = adj[:]

            if not moved:
                # Try 1-swap: remove (a,b), add (c,d)
                all_edges = get_all_edges(adj)
                non_edges = get_non_edges(adj)
                if all_edges and non_edges:
                    for _ in range(100):
                        if time.time() > deadline - 5:
                            break
                        a, b = rng.choice(all_edges)
                        c, d = rng.choice(non_edges)
                        if (c, d) == (a, b) or (c, d) == (b, a):
                            continue
                        # Remove (a,b)
                        bm2 = adj[:]
                        bm2[a] &= ~(1 << b)
                        bm2[b] &= ~(1 << a)
                        # Try add (c,d)
                        if is_add_valid(bm2, c, d):
                            bm2[c] |= (1 << d)
                            bm2[d] |= (1 << c)
                            if not has_k33(bm2):
                                adj = bm2
                                current_edges = edge_count(adj)
                                moved = True
                                if current_edges > best_local_edges:
                                    best_local_edges = current_edges
                                    best_local_adj = adj[:]
                                break

            if not moved:
                # Try 2-swap: remove (a,b), add (c,d) and (e,f)
                all_edges = get_all_edges(adj)
                non_edges = get_non_edges(adj)
                if len(all_edges) >= 1 and len(non_edges) >= 2:
                    for _ in range(50):
                        if time.time() > deadline - 5:
                            break
                        a, b = rng.choice(all_edges)
                        c, d = rng.choice(non_edges)
                        e, f = rng.choice(non_edges)
                        # Ensure distinct
                        if (c, d) == (e, f) or (c, d) == (f, e):
                            continue
                        if (c, d) == (a, b) or (c, d) == (b, a):
                            continue
                        if (e, f) == (a, b) or (e, f) == (b, a):
                            continue
                        # Remove (a,b)
                        bm2 = adj[:]
                        bm2[a] &= ~(1 << b)
                        bm2[b] &= ~(1 << a)
                        # Try add (c,d)
                        if not is_add_valid(bm2, c, d):
                            continue
                        bm2[c] |= (1 << d)
                        bm2[d] |= (1 << c)
                        # Try add (e,f)
                        if not is_add_valid(bm2, e, f):
                            continue
                        bm2[e] |= (1 << f)
                        bm2[f] |= (1 << e)
                        if not has_k33(bm2):
                            adj = bm2
                            current_edges = edge_count(adj)
                            moved = True
                            if current_edges > best_local_edges:
                                best_local_edges = current_edges
                                best_local_adj = adj[:]
                            break

            if not moved:
                no_improve += 1
                if no_improve >= 3:
                    break
            else:
                no_improve = 0

        return best_local_adj

    best_adj = None
    best_count = 0

    # Multiple restarts with different strategies
    for seed in range(60):
        if time.time() >= deadline - 10:
            break
        
        rng = random.Random(seed)
        
        if seed == 0:
            # Use grid start
            adj = grid_start()
        elif seed < 10:
            # Dense random start
            adj = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if rng.random() < 0.6:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            # Repair K3,3 violations
            for _ in range(100):
                if not has_k33(adj):
                    break
                found = False
                for u in range(n):
                    for v in range(u + 1, n):
                        common_uv = adj[u] & adj[v]
                        if common_uv.bit_count() < 3:
                            continue
                        for w in range(v + 1, n):
                            common_uvw = common_uv & adj[w]
                            if common_uvw.bit_count() >= 3:
                                candidates = []
                                for z in [u, v, w]:
                                    for nb in range(n):
                                        if z != nb and (adj[z] & (1 << nb)):
                                            candidates.append((z, nb))
                                if candidates:
                                    a, b = rng.choice(candidates)
                                    adj[a] &= ~(1 << b)
                                    adj[b] &= ~(1 << a)
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        break
                if not found:
                    break
        else:
            # Sparse random start
            adj = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if rng.random() < 0.3:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)

        adj = local_search(adj, rng)
        cnt = edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = adj[:]

    # Final greedy add pass
    if best_adj is not None:
        final_rng = random.Random(12345)
        best_adj = greedy_add(best_adj, final_rng)
        best_count = edge_count(best_adj)

    if best_adj is None:
        # Fallback: K2,18
        adj = [0] * n
        for a in range(2):
            for b in range(2, 20):
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
        best_adj = adj

    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
    return edges

import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    n = 18

    def popcount(x):
        return bin(x).count('1')

    def get_triangles(adj):
        tris = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    C = adj[u] & adj[v]
                    temp = C
                    while temp:
                        w = (temp & -temp).bit_length() - 1
                        if w > v:
                            if adj[v] & (1 << w):
                                tris.append((1 << u) | (1 << v) | (1 << w))
                        temp &= temp - 1
        return tris

    def max_triangle_free_size(tris):
        """Exact branch-and-bound: largest set of vertices containing no triangle.
        Uses suffix-OR pruning: if the OR of all triangles' vertex masks that
        still have a vertex >= pos is 0, no triangle can be formed from pos onward,
        so we can immediately accept all remaining vertices."""
        if not tris:
            return n
        m = len(tris)
        # suffix_or[i] = OR of (t_mask) for tris[i:]
        suffix_or = [0] * (m + 1)
        for i in range(m - 1, -1, -1):
            suffix_or[i] = suffix_or[i + 1] | tris[i]

        best = 0

        def bt(pos, chosen):
            nonlocal best
            # If suffix_or[pos] == 0, no triangle exists among vertices >= pos,
            # so we can take all remaining vertices.
            if suffix_or[pos] == 0:
                val = popcount(chosen) + (n - pos)
                if val > best:
                    best = val
                return
            if pos == n:
                return
            # Upper bound: chosen so far + all remaining vertices
            if popcount(chosen) + (n - pos) <= best:
                return
            # Option 1: include vertex pos
            # Check if including pos completes any triangle:
            # a triangle is completed if there exists t such that t & ~(1<<pos) has exactly 2 bits
            # and both are in chosen, OR t has 3 bits all >= pos with pos in it and other 2 in chosen.
            # Simpler: for each triangle t that contains pos, check if the other two vertices are in chosen.
            blocked = False
            for i in range(m):
                t = tris[i]
                if t & (1 << pos):
                    other = t & ~(1 << pos)
                    if other & ~chosen == 0:
                        # both other vertices are in chosen -> would form triangle
                        blocked = True
                        break
            if not blocked:
                bt(pos + 1, chosen | (1 << pos))
            # Option 2: exclude vertex pos
            bt(pos + 1, chosen)

        bt(0, 0)
        return best

    def can_add_edge_no_k4(adj, u, v):
        if adj[u] & (1 << v):
            return False
        C = adj[u] & adj[v]
        temp = C
        while temp:
            w = (temp & -temp).bit_length() - 1
            if adj[w] & C:
                return False
            temp &= temp - 1
        return True

    def random_greedy_construction(rng):
        adj = [0] * n
        possible_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
        rng.shuffle(possible_edges)
        for u, v in possible_edges:
            if can_add_edge_no_k4(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def local_search(adj, current_val, rng, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            tris = get_triangles(adj)
            # Try adding edges
            for u in range(n):
                if time.time() >= time_limit:
                    break
                for v in range(u+1, n):
                    if time.time() >= time_limit:
                        break
                    if not (adj[u] & (1 << v)):
                        if can_add_edge_no_k4(adj, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            new_tris = get_triangles(adj)
                            new_val = max_triangle_free_size(new_tris)
                            if new_val < current_val:
                                current_val = new_val
                                improved = True
                                tris = new_tris
                            else:
                                adj[u] ^= (1 << v)
                                adj[v] ^= (1 << u)
            # Try removing edges
            tris = get_triangles(adj)
            for u in range(n):
                if time.time() >= time_limit:
                    break
                for v in range(u+1, n):
                    if time.time() >= time_limit:
                        break
                    if adj[u] & (1 << v):
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                        new_tris = get_triangles(adj)
                        new_val = max_triangle_free_size(new_tris)
                        if new_val < current_val:
                            current_val = new_val
                            improved = True
                            tris = new_tris
                        else:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
        return adj, current_val

    best_val = n
    best_adj = None

    rng = random.Random(42)

    # Try multiple random constructions
    num_constructions = 200
    for i in range(num_constructions):
        if time.time() > deadline - 20:
            break

        adj = random_greedy_construction(rng)
        triangles = get_triangles(adj)
        val = max_triangle_free_size(triangles)

        if val < best_val:
            best_val = val
            best_adj = adj[:]

        # Local search on this construction
        remaining = deadline - time.time()
        if remaining > 10:
            time_limit = time.time() + min(remaining - 10, 15)
            adj, val = local_search(adj, val, rng, time_limit)

            if val < best_val:
                best_val = val
                best_adj = adj[:]

        if best_val <= 7:
            break

    # Also try the tripartite construction as a baseline
    parts = [range(0, 6), range(6, 12), range(12, 18)]
    adj = [0] * n
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    adj[a] |= (1 << b)
                    adj[b] |= (1 << a)

    triangles = get_triangles(adj)
    val = max_triangle_free_size(triangles)

    if val < best_val:
        best_val = val
        best_adj = adj[:]

    # Final local search if time permits
    if time.time() < deadline - 10:
        time_limit = deadline - 5
        best_adj, best_val = local_search(best_adj, best_val, rng, time_limit)

    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])

    return edges

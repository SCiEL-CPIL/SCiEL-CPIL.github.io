import time
import random

def construct():
    start = time.time()
    DEADLINE = 140.0

    def time_left():
        return DEADLINE - time.time()

    def path3_exists(adj, n, u, v):
        """Check if there's a path u-a-b-c-v of length 3 (4 edges) with all 5 vertices distinct.
        If yes, adding edge (u,v) creates a C5.
        """
        nu = adj[u]
        nv = adj[v]
        # a in N(u), a != v; c in N(v), c != u; a != c
        # b in N(a) ∩ N(c), b not in {u,v,a,c}
        a_mask = nu & ~(1 << v)
        c_mask = nv & ~(1 << u)
        temp_a = a_mask
        while temp_a:
            a_bit = temp_a & (-temp_a)
            a = a_bit.bit_length() - 1
            temp_a &= temp_a - 1
            temp_c = c_mask & ~a_bit
            while temp_c:
                c_bit = temp_c & (-temp_c)
                c = c_bit.bit_length() - 1
                temp_c &= temp_c - 1
                common = adj[a] & adj[c]
                common &= ~(1 << u) & ~(1 << v) & ~a_bit & ~c_bit
                if common:
                    return True
        return False

    def is_c5_free(adj, n):
        """Check if graph has no C5."""
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    if path3_exists(adj, n, u, v):
                        return False
        return True

    def alpha_ge(adj, n, target):
        """Check if independence number >= target using branch and bound."""
        def dfs(candidates, count):
            if count >= target:
                return True
            if candidates == 0:
                return False
            if count + candidates.bit_count() < target:
                return False
            if time.time() > DEADLINE:
                return False

            # Pick vertex with max degree in candidates
            best_v = -1
            best_deg = -1
            best_v_bit = 0
            temp = candidates
            while temp:
                v_bit = temp & (-temp)
                v = v_bit.bit_length() - 1
                deg = (candidates & adj[v]).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
                    best_v_bit = v_bit
                temp &= temp - 1

            if best_v < 0:
                return False

            # Branch 1: include best_v
            new_cands = candidates & ~adj[best_v] & ~best_v_bit
            if dfs(new_cands, count + 1):
                return True

            # Branch 2: exclude best_v
            if dfs(candidates & ~best_v_bit, count):
                return True

            return False

        return dfs((1 << n) - 1, 0)

    # Fallback: 8 disjoint K4s (n=32, alpha=8, C5-free)
    def make_8k4():
        n = 32
        edges = []
        for b in range(8):
            base = 4 * b
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([base + i, base + j])
        return n, edges

    best_n, best_edges = make_8k4()

    # Strategy 1: Random greedy edge addition on n=35
    # Start empty, add edges that don't create C5, check alpha at end
    for seed in range(300):
        if time.time() > DEADLINE - 20:
            break
        random.seed(seed)
        n = 35
        adj = [0] * n
        edge_list = []

        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_edges)

        for u, v in all_edges:
            if time.time() > DEADLINE - 15:
                break
            if path3_exists(adj, n, u, v):
                continue
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edge_list.append([u, v])

        # Check if alpha < 9
        if not alpha_ge(adj, n, 9):
            best_n = n
            best_edges = edge_list
            break

    # Strategy 2: Start from 8 K4s, try to extend by modifying structure
    if best_n < 35:
        for seed in range(100):
            if time.time() > DEADLINE - 20:
                break
            random.seed(seed)

            # Build 8 K4s
            n = 32
            adj = [0] * n
            edge_list = []
            for b in range(8):
                base = 4 * b
                for i in range(4):
                    for j in range(i + 1, 4):
                        adj[base + i] |= (1 << (base + j))
                        adj[base + j] |= (1 << (base + i))
                        edge_list.append([base + i, base + j])

            # Try adding vertices 32, 33, 34
            for new_n in range(33, 36):
                if time.time() > DEADLINE - 15:
                    break
                new_v = new_n - 1
                adj.append(0)

                success = False
                for attempt in range(1000):
                    if time.time() > DEADLINE - 10:
                        break

                    # Connect new_v to at most 1 vertex per K4
                    neighbors = []
                    for b in range(8):
                        base = 4 * b
                        if random.random() < 0.75:
                            pick = base + random.randint(0, 3)
                            neighbors.append(pick)

                    for nb in neighbors:
                        adj[new_v] |= (1 << nb)
                        adj[nb] |= (1 << new_v)

                    # Check C5: only need to check edges involving new_v
                    c5_ok = True
                    for nb in neighbors:
                        if path3_exists(adj, new_n, new_v, nb):
                            c5_ok = False
                            break

                    if not c5_ok:
                        for nb in neighbors:
                            adj[new_v] &= ~(1 << nb)
                            adj[nb] &= ~(1 << new_v)
                        continue

                    if alpha_ge(adj, new_n, 9):
                        for nb in neighbors:
                            adj[new_v] &= ~(1 << nb)
                            adj[nb] &= ~(1 << new_v)
                        continue

                    success = True
                    for nb in neighbors:
                        edge_list.append([new_v, nb])
                    break

                if not success:
                    adj.pop()
                    break

            if len(adj) > best_n:
                if is_c5_free(adj, len(adj)) and not alpha_ge(adj, len(adj), 9):
                    best_n = len(adj)
                    best_edges = edge_list[:]
                    if best_n >= 35:
                        break

    return {'n': best_n, 'edges': best_edges}

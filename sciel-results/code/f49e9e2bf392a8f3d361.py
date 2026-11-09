"""C4-free graph construction maximizing n with min_degree >= n-39."""
import time
import random
import math

def construct():
    deadline = time.time() + 135.0

    def verify(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        for i in range(n):
            if bin(adj[i]).count('1') < n - 39:
                return False
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    common = adj[u] & adj[v]
                    common &= ~(1 << u)
                    common &= ~(1 << v)
                    if bin(common).count('1') >= 2:
                        return False
        return True

    def try_anneal(n, min_deg, time_limit):
        def full_c4_count(adj):
            cnt = 0
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u] & (1 << v):
                        common = adj[u] & adj[v]
                        common &= ~(1 << u)
                        common &= ~(1 << v)
                        c = bin(common).count('1')
                        if c >= 2:
                            cnt += c * (c - 1) // 2
            return cnt

        def get_min_deg(adj):
            return min(bin(adj[i]).count('1') for i in range(n))

        for restart in range(50):
            if time.time() > time_limit:
                break

            rng = random.Random(restart * 1000003 + 7)

            # Initialize: random graph with degree around min_deg + margin
            target_deg = min_deg + rng.randint(1, 4)
            adj = [0] * n
            for u in range(n):
                candidates = list(range(n))
                candidates.remove(u)
                rng.shuffle(candidates)
                for v in candidates[:target_deg]:
                    if not (adj[u] & (1 << v)):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)

            c4c = full_c4_count(adj)
            md = get_min_deg(adj)
            score = -100000 * c4c + md
            best_adj = adj[:]
            best_score = score

            if c4c == 0 and md >= min_deg:
                edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if adj[u] & (1 << v):
                            edges.append([u, v])
                return edges

            T = 5.0
            T_min = 0.005
            cooling = 0.997
            max_steps = 100000

            for step in range(max_steps):
                if time.time() > time_limit:
                    break

                if step % 1000 == 0:
                    T *= cooling
                    if T < T_min:
                        T = T_min

                u = rng.randint(0, n - 1)
                v = rng.randint(0, n - 1)
                if u == v:
                    continue

                is_add = not (adj[u] & (1 << v))

                # Compute delta in C4 count
                common = adj[u] & adj[v]
                common &= ~(1 << u)
                common &= ~(1 << v)
                c = bin(common).count('1')

                if is_add:
                    delta_c4 = c * (c - 1) // 2
                else:
                    delta_c4 = -c * (c - 1) // 2

                new_c4c = c4c + delta_c4
                # Compute new min degree
                # Track degrees to make this O(1)
                new_md = get_min_deg(adj)
                # If we add edge, degrees of u and v increase by 1
                # If we remove, decrease by 1
                # For now just recompute (O(n) = O(47), acceptable)

                new_score = -100000 * new_c4c + new_md
                delta = new_score - score

                accept = False
                if delta >= 0:
                    accept = True
                else:
                    try:
                        if rng.random() < math.exp(delta / T):
                            accept = True
                    except OverflowError:
                        accept = False

                if accept:
                    if is_add:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    c4c = new_c4c
                    score = new_score
                    if score > best_score:
                        best_score = score
                        best_adj = adj[:]
                    if c4c == 0 and new_md >= min_deg:
                        edges = []
                        for uu in range(n):
                            for vv in range(uu + 1, n):
                                if adj[uu] & (1 << vv):
                                    edges.append([uu, vv])
                        return edges

            # Check best found in this restart
            c4c_best = full_c4_count(best_adj)
            md_best = get_min_deg(best_adj)
            if c4c_best == 0 and md_best >= min_deg:
                edges = []
                for uu in range(n):
                    for vv in range(uu + 1, n):
                        if best_adj[uu] & (1 << vv):
                            edges.append([uu, vv])
                return edges

        return None

    for n in [47, 46]:
        min_deg = n - 39
        time_limit = deadline - 5.0
        edges = try_anneal(n, min_deg, time_limit)
        if edges is not None:
            if verify(n, edges):
                return {'n': n, 'edges': edges}

    return {'n': 1, 'edges': []}

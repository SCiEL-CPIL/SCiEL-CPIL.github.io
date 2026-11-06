import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n = 18

    def edges_from_adj(adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges

    def has_k4(adj):
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if not ((common >> c) & 1):
                        continue
                    remaining = common & adj[c] & ~((1 << (c + 1)) - 1)
                    if remaining:
                        return True
        return False

    def get_triangles(adj):
        triangles = []
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if (common >> c) & 1:
                        triangles.append((1 << a) | (1 << b) | (1 << c))
        return triangles

    def max_triangle_free_size(triangles):
        if not triangles:
            return n

        tri_for_vertex = [[] for _ in range(n)]
        for idx, t in enumerate(triangles):
            for v in range(n):
                if (t >> v) & 1:
                    tri_for_vertex[v].append(idx)

        vertex_order = sorted(range(n), key=lambda v: len(tri_for_vertex[v]), reverse=True)
        pos_to_vertex = vertex_order

        best_found = 0

        def dfs(pos, mask, size):
            nonlocal best_found
            remaining = n - pos
            if size + remaining <= best_found:
                return
            if pos == n:
                if size > best_found:
                    best_found = size
                return

            v = pos_to_vertex[pos]

            # Option 1: Don't include v
            dfs(pos + 1, mask, size)

            # Option 2: Include v
            new_mask = mask | (1 << v)
            valid = True
            for t_idx in tri_for_vertex[v]:
                t = triangles[t_idx]
                others = t & ~(1 << v)
                if (mask & others) == others:
                    valid = False
                    break

            if valid:
                dfs(pos + 1, new_mask, size + 1)

        dfs(0, 0, 0)
        return best_found

    def remove_k4(adj):
        for _ in range(300):
            found = False
            for a in range(n):
                for b in range(a + 1, n):
                    if not ((adj[a] >> b) & 1):
                        continue
                    common = adj[a] & adj[b]
                    for c in range(b + 1, n):
                        if not ((common >> c) & 1):
                            continue
                        remaining = common & adj[c] & ~((1 << (c + 1)) - 1)
                        if remaining:
                            d_bit = remaining & (-remaining)
                            d = d_bit.bit_length() - 1
                            # Remove the edge that's part of the most K4s - try (c,d) first
                            adj[c] &= ~(1 << d)
                            adj[d] &= ~(1 << c)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                break
        return not has_k4(adj)

    def make_k666():
        adj = [0] * n
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        for i in range(3):
            for j in range(i + 1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
        return adj

    def make_random_graph(density):
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def local_search(adj, current_score, time_limit=30):
        best_adj = adj[:]
        best_score = current_score
        adj = adj[:]
        temperature = 2.0
        cooling = 0.998
        iterations = 0

        while time.time() < start_time + time_limit:
            if best_score <= 8:
                break

            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i

            old_adj = adj[:]

            if (adj[i] >> j) & 1:
                adj[i] &= ~(1 << j)
                adj[j] &= ~(1 << i)
            else:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

            if has_k4(adj):
                adj = old_adj
                temperature *= cooling
                iterations += 1
                continue

            new_triangles = get_triangles(adj)
            new_score = max_triangle_free_size(new_triangles)

            delta = new_score - current_score

            if delta <= 0 or random.random() < min(1.0, math.exp(-delta / max(temperature, 0.01))):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = adj[:]
                    temperature = 2.0  # Reset temperature on improvement
            else:
                adj = old_adj

            temperature *= cooling
            iterations += 1

        return best_adj, best_score

    random.seed(42)

    candidates = []

    # 1. K_{6,6,6}
    candidates.append(make_k666())

    # 2. C5 blow-up: 5 parts of sizes 4,4,4,3,3
    sizes = [4, 4, 4, 3, 3]
    adj_c5 = [0] * n
    parts_c5 = []
    start = 0
    for s in sizes:
        parts_c5.append(list(range(start, start + s)))
        start += s
    for i in range(5):
        for j in [(i + 1) % 5, (i + 4) % 5]:
            for a in parts_c5[i]:
                for b in parts_c5[j]:
                    adj_c5[a] |= (1 << b)
                    adj_c5[b] |= (1 << a)
    candidates.append(adj_c5)

    # 3. 4-partite: sizes 5,5,4,4 with complete 4-partite
    sizes4 = [5, 5, 4, 4]
    adj_4p = [0] * n
    parts_4 = []
    start = 0
    for s in sizes4:
        parts_4.append(list(range(start, start + s)))
        start += s
    for i in range(4):
        for j in range(i + 1, 4):
            for a in parts_4[i]:
                for b in parts_4[j]:
                    adj_4p[a] |= (1 << b)
                    adj_4p[b] |= (1 << a)
    candidates.append(adj_4p)

    # 4. 5-partite: sizes 4,4,4,3,3 complete 5-partite
    sizes5 = [4, 4, 4, 3, 3]
    adj_5p = [0] * n
    parts_5 = []
    start = 0
    for s in sizes5:
        parts_5.append(list(range(start, start + s)))
        start += s
    for i in range(5):
        for j in range(i + 1, 5):
            for a in parts_5[i]:
                for b in parts_5[j]:
                    adj_5p[a] |= (1 << b)
                    adj_5p[b] |= (1 << a)
    candidates.append(adj_5p)

    # 5. 6-partite: sizes 3,3,3,3,3,3 complete 6-partite
    adj_6p = [0] * n
    parts_6 = []
    start = 0
    for s in range(6):
        parts_6.append(list(range(start, start + 3)))
        start += 3
    for i in range(6):
        for j in range(i + 1, 6):
            for a in parts_6[i]:
                for b in parts_6[j]:
                    adj_6p[a] |= (1 << b)
                    adj_6p[b] |= (1 << a)
    candidates.append(adj_6p)

    # 6. Random dense graphs
    for _ in range(50):
        if time.time() > deadline - 60:
            break
        density = 0.3 + random.random() * 0.6
        adj = make_random_graph(density)
        if remove_k4(adj):
            candidates.append(adj)

    # Evaluate all candidates
    best_adj = None
    best_score = 18

    for adj in candidates:
        if time.time() > deadline - 30:
            break
        triangles = get_triangles(adj)
        score = max_triangle_free_size(triangles)
        if score < best_score:
            best_score = score
            best_adj = adj[:]

    if best_adj is None:
        best_adj = candidates[0]
        best_score = max_triangle_free_size(get_triangles(best_adj))

    # Local search from best candidate
    best_adj, best_score = local_search(best_adj, best_score, time_limit=50)

    # Multiple restarts if time remains
    remaining = deadline - 5 - time.time()
    if remaining > 10:
        for restart in range(5):
            if time.time() > deadline - 5:
                break
            density = 0.3 + random.random() * 0.6
            adj = make_random_graph(density)
            if not remove_k4(adj):
                continue

            triangles = get_triangles(adj)
            score = max_triangle_free_size(triangles)

            if score < best_score:
                best_adj, best_score = local_search(adj, score, time_limit=min(20, remaining - 5))
            else:
                new_adj, new_score = local_search(adj, score, time_limit=min(15, remaining - 5))
                if new_score < best_score:
                    best_adj = new_adj
                    best_score = new_score

    return edges_from_adj(best_adj)

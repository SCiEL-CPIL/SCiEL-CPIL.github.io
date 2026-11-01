import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = 115.0
    
    # Fallback: LUW graph construction
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def get_k5():
        # K5: 5 vertices, 10 edges, 4-regular
        edges = []
        for i in range(5):
            for j in range(i + 1, 5):
                edges.append((i, j))
        return edges

    def get_adj(edges, n):
        adj = [[] for _ in range(n)]
        for idx, (u, v) in enumerate(edges):
            adj[u].append((v, idx))
            adj[v].append((u, idx))
        return adj

    def enumerate_non_backtracking_closed_walks(adj, n, min_len, max_len):
        """Enumerate all non-backtracking closed walks of length in [min_len, max_len]."""
        walks = set()
        for start in range(n):
            stack = [(start, -1, ())]
            while stack:
                v, prev, path = stack.pop()
                L = len(path)
                if L >= min_len and v == start:
                    walks.add(path)
                    continue
                if L < max_len:
                    for w, eidx in adj[v]:
                        if w != prev:
                            stack.append((w, v, path + (eidx,)))
        return list(walks)

    def verify_girth(n, edges, g):
        """Verify no cycle shorter than g exists."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        for start in range(n):
            stack = [(start, -1, 0)]
            while stack:
                v, prev, depth = stack.pop()
                if depth >= g:
                    continue
                if v == start and depth > 0:
                    return False
                for w in adj[v]:
                    if w != prev:
                        stack.append((w, v, depth + 1))
        return True

    if k == 4 and g == 10:
        p_edges = get_k5()
        p_n = 5
        p_adj = get_adj(p_edges, p_n)
        
        closed_walks = enumerate_non_backtracking_closed_walks(p_adj, p_n, 3, 9)
        walk_seqs = closed_walks
        
        m = 64
        random.seed(42)
        best_voltages = None
        
        walk_edge_counts = []
        for seq in walk_seqs:
            counts = [0] * len(p_edges)
            for e in seq:
                counts[e] += 1
            walk_edge_counts.append(counts)
        
        def count_bad(m, voltages):
            count = 0
            for counts in walk_edge_counts:
                s = 0
                for i, c in enumerate(counts):
                    if c:
                        s += c * voltages[i]
                if s % m == 0:
                    count += 1
            return count

        voltages = [random.randint(1, m - 1) for _ in range(len(p_edges))]
        best_count = count_bad(m, voltages)
        
        max_iters = 300000
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            if best_count == 0:
                best_voltages = voltages
                break
            
            eidx = random.randint(0, len(p_edges) - 1)
            old_val = voltages[eidx]
            
            found_improvement = False
            for _ in range(5):
                new_val = random.randint(1, m - 1)
                if new_val == old_val:
                    continue
                voltages[eidx] = new_val
                new_count = count_bad(m, voltages)
                if new_count < best_count:
                    best_count = new_count
                    found_improvement = True
                    break
            if not found_improvement:
                voltages[eidx] = old_val
                
            if iteration % 2000 == 0 and iteration > 0:
                voltages = [random.randint(1, m - 1) for _ in range(len(p_edges))]
                best_count = count_bad(m, voltages)

        if best_count == 0:
            best_voltages = voltages

        if best_voltages is not None:
            n = p_n * m
            edges = []
            for idx, (u, v) in enumerate(p_edges):
                w = best_voltages[idx]
                for g_val in range(m):
                    src = u * m + g_val
                    dst = v * m + ((g_val + w) % m)
                    if src < dst:
                        edges.append((src, dst))
            
            return n, edges

    return luw_construct(k, g)

"""Construct a 3-regular graph with girth at least g on as few vertices as possible."""
import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 138  # safety margin before 150s kill

    def elapsed():
        return time.time() - start_time

    def verify_girth_fast(n, edges, min_girth):
        """Verify girth >= min_girth using non-backtracking BFS from each vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        for src in range(n):
            visited = {}
            queue = deque()
            for nb in adj[src]:
                queue.append((nb, src, 1))
                visited[(nb, src)] = 1

            while queue:
                node, prev, dist = queue.popleft()
                if dist >= min_girth:
                    continue
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    key = (nb, node)
                    if key in visited:
                        if visited[key] + dist < min_girth:
                            return False
                    else:
                        visited[key] = dist
                        queue.append((nb, node, dist + 1))
        return True

    def find_short_cycles(n, adj, min_girth):
        """Find all cycles of length < min_girth using non-backtracking BFS."""
        cycles = []
        for src in range(n):
            visited = {}
            queue = deque()
            for nb in adj[src]:
                queue.append((nb, src, 1, [src, nb]))
                visited[(nb, src)] = (1, [src, nb])

            while queue:
                node, prev, dist, path = queue.popleft()
                if dist >= min_girth:
                    continue
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    key = (nb, node)
                    if key in visited:
                        if visited[key][0] + dist < min_girth:
                            # Found a cycle
                            cycle = path + [nb]
                            # Canonical form for dedup
                            canon = frozenset(zip(cycle, cycle[1:] + [cycle[0]]))
                            cycles.append((canon, cycle))
                    else:
                        new_path = path + [nb]
                        visited[key] = (dist, new_path)
                        queue.append((nb, node, dist + 1, new_path))
        return cycles

    def local_search_3reg(n, min_girth, max_time):
        """
        Generate random 3-regular graphs and use 2-switches to eliminate short cycles.
        A 2-switch: if edges (a,b) and (c,d) exist and a!=c, a!=d, b!=c, b!=d,
        and (a,c) and (b,d) don't exist, we can swap to (a,c) and (b,d).
        This preserves 3-regularity.
        """
        best = None
        best_girth = 0

        def generate_random_3reg(n):
            if n % 2 == 1 or n < 4:
                return None
            stubs = [i for i in range(n) for _ in range(3)]
            for attempt in range(50):
                random.shuffle(stubs)
                edges = []
                edge_set = set()
                valid = True
                for i in range(0, len(stubs), 2):
                    u, v = stubs[i], stubs[i + 1]
                    if u == v:
                        valid = False
                        break
                    key = (min(u, v), max(u, v))
                    if key in edge_set:
                        valid = False
                        break
                    edge_set.add(key)
                    edges.append((u, v))
                if valid:
                    return edges
            return None

        def get_adj(n, edges):
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            return adj

        def get_edge_set(edges):
            return set((min(u,v), max(u,v)) for u, v in edges)

        def compute_girth(n, adj):
            """Compute exact girth using non-backtracking BFS."""
            min_found = float('inf')
            for src in range(n):
                visited = {}
                queue = deque()
                for nb in adj[src]:
                    queue.append((nb, src, 1))
                    visited[(nb, src)] = 1
                while queue:
                    node, prev, dist = queue.popleft()
                    if dist >= min_found:
                        continue
                    for nb in adj[node]:
                        if nb == prev:
                            continue
                        key = (nb, node)
                        if key in visited:
                            if visited[key] + dist < min_found:
                                min_found = visited[key] + dist
                        else:
                            visited[key] = dist
                            queue.append((nb, node, dist + 1))
            return min_found if min_found < float('inf') else 10**9

        def do_2switch(edges, edge_set, n, adj):
            """
            Try to find and apply a 2-switch that reduces the number of short cycles.
            Returns True if a switch was made.
            """
            # Find all edges
            edge_list = list(edges)
            random.shuffle(edge_list)

            # For each pair of edges, check if a 2-switch is valid and beneficial
            for i in range(len(edge_list)):
                a, b = edge_list[i]
                for j in range(i + 1, len(edge_list)):
                    c, d = edge_list[j]
                    # Check no shared vertices
                    if a == c or a == d or b == c or b == d:
                        continue
                    # Check new edges don't already exist
                    if (min(a,c), max(a,c)) in edge_set or (min(b,d), max(b,d)) in edge_set:
                        continue
                    # Check the other pairing too
                    if (min(a,d), max(a,d)) in edge_set or (min(b,c), max(b,c)) in edge_set:
                        continue

                    # Try the swap: remove (a,b),(c,d) add (a,c),(b,d)
                    new_e1 = (min(a,c), max(a,c))
                    new_e2 = (min(b,d), max(b,d))

                    # Remove old, add new
                    edge_set.discard((a,b))
                    edge_set.discard((c,d))
                    edge_set.add(new_e1)
                    edge_set.add(new_e2)

                    # Update adjacency
                    adj[a].remove(b)
                    adj[b].remove(a)
                    adj[c].remove(d)
                    adj[d].remove(c)
                    adj[a].append(c)
                    adj[c].append(a)
                    adj[b].append(d)
                    adj[d].append(b)

                    # Update edges list (in-place modification)
                    edge_list[i] = new_e1
                    edge_list[j] = new_e2

                    return True

            return False

        def count_short_cycles(n, adj, min_girth):
            """Count number of distinct short cycles (length < min_girth)."""
            cycles = set()
            for src in range(n):
                visited = {}
                queue = deque()
                for nb in adj[src]:
                    queue.append((nb, src, 1, [src, nb]))
                    visited[(nb, src)] = (1, [src, nb])
                while queue:
                    node, prev, dist, path = queue.popleft()
                    if dist >= min_girth:
                        continue
                    for nb in adj[node]:
                        if nb == prev:
                            continue
                        key = (nb, node)
                        if key in visited:
                            if visited[key][0] + dist < min_girth:
                                cycle = tuple(path + [nb])
                                # Canonical: rotate to smallest vertex
                                min_idx = cycle.index(min(cycle))
                                canon = cycle[min_idx:] + cycle[:min_idx]
                                # Also try reverse
                                rev = tuple(canon[::-1])
                                cycles.add(min(canon, rev))
                        else:
                            new_path = path + [nb]
                            visited[key] = (dist, new_path)
                            queue.append((nb, node, dist + 1, new_path))
            return len(cycles)

        # Main local search loop
        for n in range(512, 560, 2):
            if elapsed() > max_time:
                break

            max_starts = 5
            for start_idx in range(max_starts):
                if elapsed() > max_time:
                    break

                edges = generate_random_3reg(n)
                if edges is None:
                    continue

                adj = get_adj(n, edges)
                edge_set = get_edge_set(edges)

                # Compute initial girth
                girth = compute_girth(n, adj)
                if girth >= min_girth:
                    return (n, edges)

                # Local search: try to increase girth via 2-switches
                max_iterations = 5000
                no_improve_count = 0

                for it in range(max_iterations):
                    if elapsed() > max_time:
                        break

                    current_girth = compute_girth(n, adj)
                    if current_girth >= min_girth:
                        return (n, list(edge_set))

                    # Find a 2-switch that increases girth
                    edge_list = list(edge_set)
                    random.shuffle(edge_list)

                    found_improvement = False
                    for i in range(len(edge_list)):
                        a, b = edge_list[i]
                        for j in range(i + 1, len(edge_list)):
                            c, d = edge_list[j]
                            if a == c or a == d or b == c or b == d:
                                continue
                            new_e1 = (min(a,c), max(a,c))
                            new_e2 = (min(b,d), max(b,d))
                            if new_e1 in edge_set or new_e2 in edge_set:
                                continue

                            # Apply switch
                            edge_set.discard((a,b))
                            edge_set.discard((c,d))
                            edge_set.add(new_e1)
                            edge_set.add(new_e2)

                            adj[a].remove(b)
                            adj[b].remove(a)
                            adj[c].remove(d)
                            adj[d].remove(c)
                            adj[a].append(c)
                            adj[c].append(a)
                            adj[b].append(d)
                            adj[d].append(b)

                            # Check if girth improved
                            new_girth = compute_girth(n, adj)
                            if new_girth > current_girth:
                                found_improvement = True
                                no_improve_count = 0
                                break
                            else:
                                # Revert
                                edge_set.discard(new_e1)
                                edge_set.discard(new_e2)
                                edge_set.add((a,b))
                                edge_set.add((c,d))

                                adj[a].remove(c)
                                adj[c].remove(a)
                                adj[b].remove(d)
                                adj[d].remove(b)
                                adj[a].append(b)
                                adj[b].append(a)
                                adj[c].append(d)
                                adj[d].append(c)

                        if found_improvement:
                            break

                    if not found_improvement:
                        no_improve_count += 1
                        if no_improve_count > 10:
                            break

        return best

    # Moore bound for k=3, even girth g: n >= 2 * sum_{i=0}^{g/2-1} 2^i = 2*(2^{g/2}-1)
    moore_bound = 2 * ((1 << (g // 2)) - 1)
    if moore_bound < 512:
        moore_bound = 512

    # Strategy: local search with 2-switches to eliminate short cycles
    result = local_search_3reg(moore_bound, g, deadline - start_time - 5)
    if result is not None:
        return result

    # Fallback: LUW construction (large but guaranteed valid)
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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

"""Construct a k-regular graph of girth at least g, minimizing vertex count."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    def time_left():
        return deadline - time.time()

    # Moore bound for lower bound on n
    if g % 2 == 0:
        moore = k + k * (k - 1) ** (g // 2 - 1)
    else:
        if k == 2:
            moore = g
        else:
            h = (g - 1) // 2
            moore = 1 + k * ((k - 1) ** h - 1) // (k - 2) if k > 2 else g

    best_n = None
    best_edges = None

    def verify_girth(n, edges, min_girth):
        """Verify no cycle shorter than min_girth using non-backtracking walks from every vertex."""
        if n == 0:
            return True
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        # Check simple: no self-loops, no duplicate edges
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            key = (min(u, v), max(u, v))
            if key in edge_set:
                return False
            edge_set.add(key)
        # Non-backtracking walk check: for each directed edge (u,v),
        # find shortest closed non-backtracking walk starting with (u,v).
        # A cycle of length L < min_girth exists iff some NB walk closes with length < min_girth.
        max_depth = min_girth - 1  # we only need to check up to this depth
        for start_u in range(n):
            for start_v in adj[start_u]:
                # State: (current_node, prev_node)
                # BFS in the non-backtracking graph
                visited = {}  # (node, prev) -> distance
                q = deque()
                q.append((start_v, start_u, 1))
                visited[(start_v, start_u)] = 1
                found_cycle = False
                while q:
                    node, prev, dist = q.popleft()
                    if dist >= max_depth:
                        continue
                    for nxt in adj[node]:
                        if nxt == prev:
                            continue
                        if nxt == start_u and dist + 1 < min_girth:
                            # Found a cycle of length dist + 1
                            if dist + 1 < min_girth:
                                return False
                        new_state = (nxt, node)
                        if new_state not in visited:
                            visited[new_state] = dist + 1
                            q.append((nxt, node, dist + 1))
        return True

    def generate_random_regular(n, k, seed=None):
        """Generate a random k-regular graph on n vertices using configuration model + deduplication."""
        if seed is not None:
            random.seed(seed)
        stubs = [i for i in range(n) for _ in range(k)]
        random.shuffle(stubs)
        adj = [[] for _ in range(n)]
        edge_set = set()
        valid = True
        for i in range(n * k // 2):
            u, v = stubs[2 * i], stubs[2 * i + 1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edge_set:
                valid = False
                break
            edge_set.add(key)
            adj[u].append(v)
            adj[v].append(u)
        if not valid:
            return None
        edges = list(edge_set)
        return n, edges

    def local_search_improve(n, edges, min_girth, max_time=5.0):
        """Try to improve the graph by edge swaps that maintain regularity and girth."""
        start_ls = time.time()
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Build edge lookup
        edge_set = set()
        for u, v in edges:
            edge_set.add((min(u, v), max(u, v)))
        
        improved = False
        for _ in range(200):
            if time.time() - start_ls > max_time:
                break
            if time_left() < 3:
                break
            
            # Try a random 2-opt style swap: remove edges (a,b) and (c,d), add (a,c) and (b,d)
            # or remove (a,b) and (c,d), add (a,d) and (b,c)
            e1 = random.choice(list(edge_set))
            a, b = e1
            # Find another edge not sharing vertices with e1
            candidates = [e for e in edge_set if e != e1 and a not in e and b not in e]
            if not candidates:
                continue
            e2 = random.choice(candidates)
            c, d = e2
            
            # Try swap 1: (a,c) and (b,d)
            new_e1 = (min(a, c), max(a, c))
            new_e2 = (min(b, d), max(b, d))
            if new_e1 != e1 and new_e2 != e2 and new_e1 not in edge_set and new_e2 not in edge_set and a != c and b != d:
                # Remove old edges, add new
                edge_set.discard(e1)
                edge_set.discard(e2)
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
                
                # Verify girth
                if verify_girth(n, list(edge_set), min_girth):
                    improved = True
                else:
                    # Revert
                    edge_set.discard(new_e1)
                    edge_set.discard(new_e2)
                    edge_set.add(e1)
                    edge_set.add(e2)
                    adj[a].remove(c)
                    adj[c].remove(a)
                    adj[b].remove(d)
                    adj[d].remove(b)
                    adj[a].append(b)
                    adj[b].append(a)
                    adj[c].append(d)
                    adj[d].append(c)
        
        return list(edge_set)

    def try_explicit_construction():
        """The Lazebnik-Ustimenko-Woldar construction."""
        q = k
        if (g - 5) % 2 == 1:
            d = g - 5
        else:
            d = g - 4
        if d < 3:
            d = 3

        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            def add(x, y): return x ^ y
            def sub(x, y): return x ^ y
            def mul(x, y): return table[x][y]
        else:
            def add(x, y): return (x + y) % q
            def sub(x, y): return (x - y) % q
            def mul(x, y): return x * y % q

        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Step 1: Try explicit construction
    if time_left() > 10:
        try:
            n_exp, edges_exp = try_explicit_construction()
            if verify_girth(n_exp, edges_exp, g):
                best_n = n_exp
                best_edges = edges_exp
        except Exception:
            pass

    # Step 2: Try random regular graphs
    if best_n is None or time_left() > 5:
        start_n = max(moore, k + 1)
        if best_n is not None:
            upper_n = best_n
        else:
            upper_n = start_n * 3

        lo = max(start_n, moore)
        hi = upper_n if upper_n else lo * 5

        for n_try in range(lo, hi + 1, max(1, (hi - lo) // 20)):
            if time_left() < 5:
                break
            for attempt in range(50):
                if time_left() < 5:
                    break
                result = generate_random_regular(n_try, k, seed=random.randint(0, 2**31))
                if result is not None:
                    n_r, edges_r = result
                    if verify_girth(n_r, edges_r, g):
                        if best_n is None or n_r < best_n:
                            best_n = n_r
                            best_edges = edges_r
                        break

    # Step 3: Local search improvement on best found
    if best_n is not None and time_left() > 5:
        improved_edges = local_search_improve(best_n, best_edges, g, max_time=min(10.0, time_left() - 5))
        if improved_edges:
            best_edges = improved_edges

    if best_n is None:
        n_exp, edges_exp = try_explicit_construction()
        return n_exp, edges_exp

    return best_n, best_edges
# EVOLVE-BLOCK-END

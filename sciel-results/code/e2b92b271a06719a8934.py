import time
import random
import sys
from collections import deque

def _is_4cycle(edges_set, a, b, c, d):
    # a-b-c-d-a all edges present
    return ((a,b) in edges_set or (b,a) in edges_set) and \
           ((b,c) in edges_set or (c,b) in edges_set) and \
           ((c,d) in edges_set or (d,c) in edges_set) and \
           ((d,a) in edges_set or (a,d) in edges_set)

def construct(k, g):
    deadline = time.time() + 140  # safety margin before 150s kill
    # For the task, k=4, g=10. Generalize but focus on k=4.
    if k != 4:
        # Fallback: use a simple approach for other k (not expected in this task)
        return _construct_fallback(k, g)

    girth_target = g  # need girth >= 10
    # We'll work with adjacency lists and a set of edges (canonical (min,max))
    
    def make_random_4reg(n):
        # Configuration model: 2n half-edges, pair them randomly, ensure simple
        for attempt in range(200):
            half = list(range(2 * n))
            # Each vertex i has half-edges 2i, 2i+1
            # Pair 2n half-edges randomly
            random.shuffle(half)
            adj = [set() for _ in range(n)]
            ok = True
            for i in range(0, 2 * n, 2):
                u = half[i] // 2
                v = half[i + 1] // 2
                if u == v:
                    ok = False
                    break
                if v in adj[u]:
                    ok = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            if ok:
                # Check all degree 4
                if all(len(adj[i]) == 4 for i in range(n)):
                    return adj
        return None

    def adj_to_edges(adj, n):
        edges = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.add((u, v))
        return edges

    def find_short_cycles(adj, n, max_len):
        # Find all cycles of length <= max_len using BFS from each vertex,
        # tracking the path to reconstruct cycles.
        # Returns list of cycles (list of vertices, first==last not included).
        cycles = []
        seen_cycle_keys = set()
        for start in range(n):
            # BFS with parent tracking, depth up to max_len
            # State: (node, prev, depth, path_tuple)
            # To avoid exponential blowup, limit and use visited (node, prev) at depth
            visited = {}  # (node, prev) -> depth
            queue = deque()
            for nb in adj[start]:
                queue.append((nb, start, 1, (start, nb)))
                visited[(nb, start)] = 1
            while queue:
                node, prev, depth, path = queue.popleft()
                if depth >= max_len:
                    continue
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    new_depth = depth + 1
                    if nb == start and new_depth >= 3:
                        # Found a cycle: path + start
                        cyc = list(path)
                        # Canonical form: rotate so min vertex first, direction consistent
                        cyc_key = _canonical_cycle(cyc)
                        if cyc_key not in seen_cycle_keys:
                            seen_cycle_keys.add(cyc_key)
                            cycles.append(cyc)
                        continue
                    if (nb, node) in visited:
                        continue
                    visited[(nb, node)] = new_depth
                    queue.append((nb, node, new_depth, path + (nb,)))
        return cycles

    def _canonical_cycle(cyc):
        # cyc is a list of distinct vertices forming a cycle (len >= 3)
        m = min(cyc)
        idx = cyc.index(m)
        # Two directions
        fwd = tuple(cyc[idx:] + cyc[:idx])
        bwd = tuple([cyc[idx]] + list(reversed(cyc[:idx])))
        return min(fwd, bwd)

    def count_short_cycles(adj, n, max_len):
        return len(find_short_cycles(adj, n, max_len))

    def two_switch(adj, n, e1, e2, variant):
        # e1=(a,b), e2=(c,d), disjoint. variant 0: add (a,c),(b,d); variant 1: add (a,d),(b,c)
        a, b = e1
        c, d = e2
        if variant == 0:
            new1, new2 = (a, c), (b, d)
        else:
            new1, new2 = (a, d), (b, c)
        # Check not creating loops or duplicates
        for (x, y) in (new1, new2):
            if x == y:
                return False
            if y in adj[x]:
                return False
        # Also ensure we're not just undoing (i.e., new edges not equal to removed)
        if set([frozenset(new1), frozenset(new2)]) == set([frozenset(e1), frozenset(e2)]):
            return False
        # Apply
        adj[a].discard(b); adj[b].discard(a)
        adj[c].discard(d); adj[d].discard(c)
        for (x, y) in (new1, new2):
            adj[x].add(y); adj[y].add(x)
        return True

    def undo_two_switch(adj, e1, e2, variant):
        a, b = e1
        c, d = e2
        if variant == 0:
            new1, new2 = (a, c), (b, d)
        else:
            new1, new2 = (a, d), (b, c)
        for (x, y) in (new1, new2):
            adj[x].discard(y); adj[y].discard(x)
        adj[a].add(b); adj[b].add(a)
        adj[c].add(d); adj[d].add(c)

    # Start with a reasonably large n (4-regular). Larger n -> easier to have high girth.
    # Known: 320 vertices works. Let's start around 250-320.
    best_adj = None
    best_score = None
    best_n = None

    start_n = 260
    for trial_n in [start_n, start_n + 20, start_n + 40, start_n + 60, start_n + 80]:
        if time.time() > deadline - 20:
            break
        adj = make_random_4reg(trial_n)
        if adj is None:
            continue
        # Evaluate
        score = count_short_cycles(adj, trial_n, girth_target)
        if best_score is None or score < best_score:
            best_score = score
            best_adj = [set(s) for s in adj]
            best_n = trial_n
        if best_score == 0:
            break

    if best_adj is None:
        # Fallback: return a known-good structured graph (Lazebnik) if we can't do better
        return _construct_fallback(k, g)

    adj = best_adj
    n = best_n

    # Local search: hill-climbing with 2-switches to reduce short-cycle count
    current_score = best_score
    # Precompute all edges for switch candidates
    def get_edges(adj, n):
        es = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    es.append((u, v))
        return es

    # Simulated annealing / hill climbing
    T = 5.0
    T_min = 0.01
    cooling = 0.9995
    no_improve = 0
    max_no_improve = 200000
    step = 0
    while time.time() < deadline - 15:
        step += 1
        edges = get_edges(adj, n)
        if len(edges) < 2:
            break
        # Pick two random non-adjacent edges
        e1 = random.choice(edges)
        e2 = random.choice(edges)
        a, b = e1
        c, d = e2
        # Must be disjoint and not share a vertex
        if len({a, b, c, d}) < 4:
            continue
        variant = random.randint(0, 1)
        # Check validity
        new1, new2 = ((a, c), (b, d)) if variant == 0 else ((a, d), (b, c))
        valid = True
        for (x, y) in (new1, new2):
            if x == y or y in adj[x]:
                valid = False
                break
        if not valid:
            continue
        # Apply
        two_switch(adj, n, e1, e2, variant)
        new_score = count_short_cycles(adj, n, girth_target)
        delta = new_score - current_score
        if delta <= 0 or random.random() < pow(2.718281828, -delta / max(T, 1e-9)):
            current_score = new_score
            no_improve = 0
            if new_score == 0:
                break
        else:
            undo_two_switch(adj, e1, e2, variant)
            no_improve += 1
        T = max(T * cooling, T_min)
        if no_improve > max_no_improve:
            break

    # Final verification: exact girth via non-backtracking walks from every vertex
    def verify_girth(adj, n, target_girth):
        # Non-backtracking BFS from each vertex; if any returns to start in < target_girth steps, fail.
        for start in range(n):
            # State: (node, prev)
            dist = {}
            queue = deque()
            for nb in adj[start]:
                queue.append((nb, start))
                dist[(nb, start)] = 1
            while queue:
                node, prev = queue.popleft()
                d = dist[(node, prev)]
                if d >= target_girth:
                    continue
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if nb == start:
                        # cycle of length d+1
                        if d + 1 < target_girth:
                            return False
                        continue
                    key = (nb, node)
                    if key in dist:
                        continue
                    dist[key] = d + 1
                    queue.append((nb, node))
        return True

    if verify_girth(adj, n, girth_target):
        # Success! Build edge list
        edge_list = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edge_list.append((u, v))
        return n, edge_list

    # If local search didn't reach girth target, try a few more restarts with different n
    for trial_n in [n - 20, n + 20, n + 40, n + 60]:
        if time.time() > deadline - 15:
            break
        if trial_n < 2 * k:
            continue
        adj2 = make_random_4reg(trial_n)
        if adj2 is None:
            continue
        # Quick local search
        sc = count_short_cycles(adj2, trial_n, girth_target)
        T2 = 3.0
        for _ in range(50000):
            if time.time() > deadline - 15:
                break
            edges2 = get_edges(adj2, trial_n)
            if len(edges2) < 2:
                break
            e1 = random.choice(edges2)
            e2 = random.choice(edges2)
            a, b = e1
            c, d = e2
            if len({a, b, c, d}) < 4:
                continue
            variant = random.randint(0, 1)
            new1, new2 = ((a, c), (b, d)) if variant == 0 else ((a, d), (b, c))
            valid = True
            for (x, y) in (new1, new2):
                if x == y or y in adj2[x]:
                    valid = False
                    break
            if not valid:
                continue
            two_switch(adj2, trial_n, e1, e2, variant)
            ns = count_short_cycles(adj2, trial_n, girth_target)
            delta = ns - sc
            if delta <= 0 or random.random() < pow(2.718281828, -delta / max(T2, 1e-9)):
                sc = ns
                if sc == 0:
                    break
            else:
                undo_two_switch(adj2, e1, e2, variant)
            T2 = max(T2 * 0.9999, 0.01)
        if sc == 0 and verify_girth(adj2, trial_n, girth_target):
            edge_list = []
            for u in range(trial_n):
                for v in adj2[u]:
                    if u < v:
                        edge_list.append((u, v))
            return trial_n, edge_list

    # If we still don't have a verified girth-10 graph, return the best we have
    # (it may not satisfy girth, but the evaluator will reject; better to return
    #  a valid 4-regular graph than crash). Actually, we must return a VALID graph.
    # Let's return the Lazebnik fallback which is guaranteed valid (girth >= 10 for d chosen).
    return _construct_fallback(k, g)


def _construct_fallback(k, g):
    # Lazebnik-Ustimenko-Woldar CD(d, q) as in seed — guaranteed girth >= d+5 for odd d.
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

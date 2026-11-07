import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin for 90s limit

    n = 21
    edges = P['edges']
    edge_set = set()
    adj = defaultdict(set)
    for u, v in edges:
        edge_set.add((min(u, v), max(u, v)))
        adj[u].add(v)
        adj[v].add(u)

    m = len(edges)
    # Normalize edge list
    norm_edges = []
    for u, v in edges:
        norm_edges.append((min(u, v), max(u, v)))
    # Map normalized edge -> index
    edge_to_idx = {}
    for i, (u, v) in enumerate(norm_edges):
        edge_to_idx[(u, v)] = i

    def edge_idx(u, v):
        return edge_to_idx.get((min(u, v), max(u, v)), -1)

    def time_ok():
        return time.time() < deadline

    def try_ilp(max_cycles, extra_cuts=None):
        """
        ILP: x[e, c] in {0,1} for each edge e and cycle c.
        Constraints:
          - Each vertex has degree exactly 2 in each cycle (sum of incident x = 2)
          - Each edge used in at most one cycle (sum_c x[e,c] <= 1)
          - Subtour elimination: for each cycle c and each non-trivial subset S of vertices,
            sum of edges in S for cycle c >= 2 (if |S| >= 2) ... actually for a Hamiltonian cycle
            we need: for every proper subset S (2 <= |S| <= n-2), edges of cycle c inside S >= 2
            OR equivalently the cycle must be connected.
        Objective: maximize sum of x[e,c]
        """
        import numpy as np
        from scipy.optimize import LinearConstraint, milp, Bounds

        if not time_ok():
            return None

        K = max_cycles
        # Variables: x[e, c] for e in range(m), c in range(K)
        # Total variables: m * K
        # Index: var(e, c) = e * K + c

        def var(e, c):
            return e * K + c

        nvars = m * K

        # Objective: maximize sum x[e,c] -> minimize -sum
        c_obj = np.full(nvars, -1.0)

        # Bounds: 0 <= x <= 1
        lb = np.zeros(nvars)
        ub = np.ones(nvars)
        bounds = Bounds(lb, ub)

        # Integrality: all binary
        integrality = np.ones(nvars)

        # Constraints:
        # 1. For each vertex v and cycle c: sum of x[e,c] over edges e incident to v == 2
        # 2. For each edge e: sum_c x[e,c] <= 1
        # 3. Subtour elimination cuts (added iteratively)

        # Build constraint matrix rows
        A_rows = []
        lb_rows = []
        ub_rows = []

        # Degree constraints: for each vertex v, cycle c: sum_{e incident to v} x[e,c] == 2
        for v in range(n):
            incident = [e for e in range(m) if norm_edges[e][0] == v or norm_edges[e][1] == v]
            for c in range(K):
                row = np.zeros(nvars)
                for e in incident:
                    row[var(e, c)] = 1.0
                A_rows.append(row)
                lb_rows.append(2.0)
                ub_rows.append(2.0)

        # Edge capacity: for each edge e: sum_c x[e,c] <= 1
        for e in range(m):
            row = np.zeros(nvars)
            for c in range(K):
                row[var(e, c)] = 1.0
            A_rows.append(row)
            lb_rows.append(0.0)
            ub_rows.append(1.0)

        # Subtour elimination: for each cycle c and subset S (2 <= |S| <= n-2):
        # sum of x[e,c] for e with both endpoints in S >= 2
        # This is exponential, so we add lazily. Start with a few obvious ones.
        # Actually, for a 2-regular connected graph on n vertices, it's a Hamiltonian cycle.
        # A 2-regular graph may be a union of cycles. We need to eliminate subtours.
        # We'll add cuts iteratively.

        def add_subtour_cuts(cycles_found):
            """Add subtour elimination cuts based on found cycles."""
            nonlocal A_rows, lb_rows, ub_rows
            for c in range(K):
                cycle = cycles_found[c]
                if cycle is None:
                    continue
                S = set(cycle)
                if len(S) < 2 or len(S) >= n:
                    continue
                # Edges with both endpoints in S
                row = np.zeros(nvars)
                count = 0
                for e in range(m):
                    u, v = norm_edges[e]
                    if u in S and v in S:
                        row[var(e, c)] = 1.0
                        count += 1
                # For a valid Hamiltonian cycle, if S is a proper subset, 
                # the number of cycle edges inside S must be 0 or >= 2.
                # Since degree is 2 at each vertex in S, if any edge is in S, 
                # we need at least 2 edges (to connect). Actually for a cycle passing through S,
                # if it enters S it must leave, so edges in S are even.
                # For subtour elimination: if S is a proper subset and the cycle is restricted to S,
                # that's a subtour. We need: sum >= 2 OR sum == 0.
                # Since we can't do OR in LP, we use: if the solution has a subtour on S,
                # we add cut: sum_{e in S} x[e,c] >= 2 (forcing the cycle to leave S)
                # But this is only valid if the subtour is a proper subset.
                # Actually the standard subtour elimination cut is:
                # sum_{e in delta(S)} x[e,c] >= 2 for any 2 <= |S| <= n-2
                # where delta(S) is the set of edges with exactly one endpoint in S.
                # Wait, that's for TSP. For our case (2-factor that must be a single cycle),
                # the cut is: for any proper subset S with 2 <= |S| <= n-2,
                # the number of cycle edges crossing the cut delta(S) must be >= 2.
                # Because a single cycle crossing from S to V\S must cross at least twice.
                row2 = np.zeros(nvars)
                for e in range(m):
                    u, v = norm_edges[e]
                    if (u in S) != (v in S):  # exactly one endpoint in S
                        row2[var(e, c)] = 1.0
                A_rows.append(row2)
                lb_rows.append(2.0)
                ub_rows.append(np.inf)

        # Build initial constraint matrix
        if A_rows:
            A = np.array(A_rows)
            lb_arr = np.array(lb_rows)
            ub_arr = np.array(ub_rows)
        else:
            A = np.empty((0, nvars))
            lb_arr = np.empty(0)
            ub_arr = np.empty(0)

        # Solve iteratively with subtour elimination
        for iteration in range(20):
            if not time_ok():
                return None

            try:
                constraint = LinearConstraint(A, lb_arr, ub_arr)
                res = milp(
                    c=c_obj,
                    constraints=constraint,
                    integrality=integrality,
                    bounds=bounds,
                    options={'time_limit': 40, 'mip_rel_gap': 0.0}
                )
            except Exception:
                return None

            if not res.success:
                return None

            if res.x is None:
                return None

            # Extract solution
            x = res.x
            # Check if all binary
            cycles = []
            valid = True
            for c in range(K):
                # Find edges assigned to cycle c
                cycle_edges = []
                for e in range(m):
                    if abs(x[var(e, c)] - 1.0) < 1e-6:
                        cycle_edges.append(e)
                if len(cycle_edges) != n:
                    valid = False
                    break
                # Build the cycle from edges
                cycle_adj = defaultdict(list)
                for e in cycle_edges:
                    u, v = norm_edges[e]
                    cycle_adj[u].append(v)
                    cycle_adj[v].append(u)
                # Check each vertex has degree 2
                for v in range(n):
                    if len(cycle_adj[v]) != 2:
                        valid = False
                        break
                if not valid:
                    break
                # Trace the cycle
                start = 0
                path = [start]
                prev = None
                current = start
                for _ in range(n):
                    next_v = None
                    for nb in cycle_adj[current]:
                        if nb != prev:
                            next_v = nb
                            break
                    if next_v is None:
                        valid = False
                        break
                    path.append(next_v)
                    prev, current = current, next_v
                if path[-1] != path[0]:
                    valid = False
                    break
                path = path[:-1]  # Remove duplicate
                if len(set(path)) != n:
                    valid = False
                    break
                cycles.append(path)
            
            if not valid:
                return None

            # Check for subtours: for each cycle, verify it's a single cycle on all n vertices
            # If all cycles are Hamiltonian, we're done
            all_hamiltonian = True
            for c in range(K):
                if len(cycles[c]) != n or len(set(cycles[c])) != n:
                    all_hamiltonian = False
                    break
            
            if all_hamiltonian:
                return cycles

            # Add subtour cuts
            added = False
            for c in range(K):
                cycle = cycles[c]
                S = set(cycle)
                if 2 <= len(S) <= n - 2:
                    # Add cut for this subset
                    row2 = np.zeros(nvars)
                    for e in range(m):
                        u, v = norm_edges[e]
                        if (u in S) != (v in S):
                            row2[var(e, c)] = 1.0
                    A_rows.append(row2)
                    lb_rows.append(2.0)
                    ub_rows.append(np.inf)
                    added = True
            
            if not added:
                break

            # Rebuild matrix
            A = np.array(A_rows)
            lb_arr = np.array(lb_rows)
            ub_arr = np.array(ub_rows)

        return None

    def try_backtracking(max_cycles):
        """Fallback: find edge-disjoint Hamiltonian cycles via backtracking."""
        if not time_ok():
            return None

        used_edges = set()
        cycles = []

        for c in range(max_cycles):
            if not time_ok():
                break
            cycle = _find_ham_cycle_bt(set(range(n)), used_edges, adj, deadline)
            if cycle is None:
                # Try random restarts
                found = False
                for attempt in range(50):
                    if not time_ok():
                        break
                    cycle = _find_ham_cycle_bt_random(set(range(n)), used_edges, adj, deadline, seed=c*1000+attempt)
                    if cycle is not None:
                        found = True
                        break
                if not found:
                    return cycles if cycles else None
            for i in range(n):
                u = cycle[i]
                v = cycle[(i + 1) % n]
                used_edges.add((min(u, v), max(u, v)))
            cycles.append(cycle)

        return cycles

    def _find_ham_cycle_bt(vertices, used_edges, adj, deadline):
        """Backtracking Hamiltonian cycle finder with pruning."""
        k = len(vertices)
        if k < 3:
            return None

        # Precompute available neighbors
        avail = {}
        for v in vertices:
            avail[v] = [u for u in adj[v] if u in vertices and (min(u, v), max(u, v)) not in used_edges]
            if len(avail[v]) < 2:
                return None

        # Choose start vertex with minimum degree
        start = min(vertices, key=lambda v: len(avail[v]))
        remaining = set(vertices) - {start}
        path = [start]

        def backtrack(path, remaining):
            if not time_ok():
                return None
            if len(path) == k:
                last = path[-1]
                edge = (min(last, start), max(last, start))
                if edge not in used_edges and last in adj[start]:
                    return path[:]
                return None

            last = path[-1]
            candidates = []
            for v in remaining:
                if v in adj[last]:
                    edge = (min(last, v), max(last, v))
                    if edge not in used_edges:
                        # Pruning: check if v can still connect to remaining
                        future_opts = len([w for w in avail[v] if w != last and w in remaining or (w == start and len(path) == k - 1)])
                        if future_opts >= 1:  # at least one option
                            candidates.append((future_opts, v))

            candidates.sort()

            for _, v in candidates:
                if not time_ok():
                    return None
                edge = (min(last, v), max(last, v))
                if edge not in used_edges:
                    path.append(v)
                    remaining.remove(v)
                    result = backtrack(path, remaining)
                    if result is not None:
                        return result
                    path.pop()
                    remaining.add(v)
            return None

        return backtrack(path, remaining)

    def _find_ham_cycle_bt_random(vertices, used_edges, adj, deadline, seed=0):
        """Randomized backtracking Hamiltonian cycle finder."""
        k = len(vertices)
        if k < 3:
            return None

        rng = random.Random(seed)

        for attempt in range(30):
            if not time_ok():
                return None

            avail = {}
            valid = True
            for v in vertices:
                avail[v] = [u for u in adj[v] if u in vertices and (min(u, v), max(u, v)) not in used_edges]
                if len(avail[v]) < 2:
                    valid = False
                    break
            if not valid:
                return None

            start = rng.choice(list(vertices))
            remaining = set(vertices) - {start}
            path = [start]

            def backtrack(path, remaining):
                if not time_ok():
                    return None
                if len(path) == k:
                    last = path[-1]
                    edge = (min(last, start), max(last, start))
                    if edge not in used_edges and last in adj[start]:
                        return path[:]
                    return None

                last = path[-1]
                candidates = []
                for v in remaining:
                    if v in adj[last]:
                        edge = (min(last, v), max(last, v))
                        if edge not in used_edges:
                            future_opts = len([w for w in avail[v] if w != last and (w in remaining or (w == start and len(path) == k - 1))])
                            if future_opts >= 1:
                                candidates.append((future_opts, v))

                if candidates:
                    min_opts = min(c[0] for c in candidates)
                    constrained = [c for c in candidates if c[0] <= min_opts + 1]
                    rng.shuffle(constrained)
                    candidates = constrained

                for _, v in candidates:
                    if not time_ok():
                        return None
                    edge = (min(last, v), max(last, v))
                    if edge not in used_edges:
                        path.append(v)
                        remaining.remove(v)
                        result = backtrack(path, remaining)
                        if result is not None:
                            return result
                        path.pop()
                        remaining.add(v)
                return None

            result = backtrack(path, remaining)
            if result is not None:
                return result

        return None

    # Try ILP first for 3 cycles
    best_cycles = None

    for max_c in [3, 2, 1]:
        if not time_ok():
            break
        result = try_ilp(max_c)
        if result is not None and len(result) >= max_c:
            best_cycles = result[:max_c]
            break

    # If ILP failed, try backtracking
    if best_cycles is None or len(best_cycles) < 3:
        for max_c in [3, 2, 1]:
            if not time_ok():
                break
            result = try_backtracking(max_c)
            if result is not None and len(result) >= max_c and (best_cycles is None or len(result) > len(best_cycles)):
                best_cycles = result[:max_c]
                break

    # Return best result
    if best_cycles:
        result = []
        for cycle in best_cycles:
            result.append([int(v) for v in cycle])
        return result
    else:
        # Fallback: single cycle
        cycle = _find_ham_cycle_bt(set(range(n)), set(), adj, deadline)
        if cycle:
            return [[int(v) for v in cycle]]
        return [list(range(n))]

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

import random
import time
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    if k != 3:
        # Fallback to LUW for other k
        return _luw_graph(k, g)

    # Petersen Graph
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # Outer
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # Inner
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # Spokes
    ]
    n_base = 10
    adj = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))

    # Generate all closed non-backtracking walks of length 3..g-1 in the Petersen graph
    walks = []
    for start_v in range(n_base):
        # BFS state: (current_v, prev_edge, length, path)
        queue = [(start_v, -1, 0, [])]
        while queue:
            curr_v, prev_e, length, path = queue.pop(0)
            if length >= g:
                continue
            if length >= 3 and curr_v == start_v:
                walks.append(path)
            for next_v, e_id in adj[curr_v]:
                if e_id == prev_e:
                    continue
                u, v = edges[e_id]
                sign = 1 if u == curr_v else -1
                new_path = path + [(e_id, sign)]
                queue.append((next_v, e_id, length + 1, new_path))

    # Deduplicate
    unique_walks = set()
    for w in walks:
        unique_walks.add(tuple(w))
    walks = list(unique_walks)

    # Precompute coefficients for each walk
    walk_coeffs = []
    for w in walks:
        coeffs = [0] * len(edges)
        for e_id, sign in w:
            coeffs[e_id] += sign
        # Only keep walks with non-trivial coefficients
        if any(c != 0 for c in coeffs):
            walk_coeffs.append(coeffs)

    # Sort walks by number of non-zero coefficients (fewer = faster to check)
    walk_coeffs.sort(key=lambda c: sum(1 for x in c if x != 0))

    # For each walk, store list of (edge_idx, coeff) for non-zero entries
    walk_sparse = []
    for coeffs in walk_coeffs:
        sparse = [(i, c) for i, c in enumerate(coeffs) if c != 0]
        walk_sparse.append(sparse)

    def check_girth(volts, m):
        """Check if voltage assignment has no short cycles. Returns True if valid."""
        for sparse in walk_sparse:
            s = 0
            for e_id, c in sparse:
                s += c * volts[e_id]
            if s % m == 0:
                return False
        return True

    def is_connected(volts, m):
        """Check if the lift is connected (gcd of all volts and m is 1)."""
        g_val = m
        for v in volts:
            g_val = math.gcd(g_val, v)
            if g_val == 1:
                break
        return g_val == 1

    def build_lift_edges(volts, m):
        """Build the edge list for the voltage lift."""
        lift_edges = []
        for i, (u, v) in enumerate(edges):
            vol = volts[i]
            for t in range(m):
                u_lift = u * m + t
                v_lift = v * m + (t + vol) % m
                lift_edges.append((u_lift, v_lift))
        return lift_edges

    best_n = float('inf')
    best_edges = None

    # Strategy: Try m from Moore bound upward
    # Moore bound for (3,16) is 512, so m >= 52 (since n = 10*m)
    # But we can try smaller m if the graph happens to have girth >= 16
    # Actually for Petersen lift, girth is determined by the voltage assignment
    # Let's try m from 52 upward

    # Use Min-Conflicts style local search for each m
    for m in range(52, 250):
        if time.time() > deadline - 3.0:
            break

        candidate_n = 10 * m
        if candidate_n >= best_n:
            break

        found = False
        # Multiple random restarts with local search
        n_restarts = 500
        for restart in range(n_restarts):
            if time.time() > deadline - 2.0:
                break

            # Initialize with random voltages
            # Bias towards values that generate the full group
            volts = [random.randint(1, m - 1) for _ in range(len(edges))]

            # Ensure connectivity
            if not is_connected(volts, m):
                # Try to fix by adjusting one voltage
                fixed = False
                for i in range(len(edges)):
                    old = volts[i]
                    for trial in range(10):
                        volts[i] = random.randint(1, m - 1)
                        if is_connected(volts, m):
                            fixed = True
                            break
                    if fixed:
                        break
                    volts[i] = old
                if not fixed:
                    continue

            # Count violations
            def count_violations(volts, m):
                viol = 0
                for sparse in walk_sparse:
                    s = 0
                    for e_id, c in sparse:
                        s += c * volts[e_id]
                    if s % m == 0:
                        viol += 1
                return viol

            current_viol = count_violations(volts, m)

            # Local search: Min-Conflicts
            max_iter = 200
            for it in range(max_iter):
                if time.time() > deadline - 2.0:
                    break
                if current_viol == 0:
                    break

                # Find a violated walk
                violated_walks = []
                for idx, sparse in enumerate(walk_sparse):
                    s = 0
                    for e_id, c in sparse:
                        s += c * volts[e_id]
                    if s % m == 0:
                        violated_walks.append((idx, sparse))

                if not violated_walks:
                    break

                # Pick a random violated walk
                _, sparse = random.choice(violated_walks)
                # Pick a random edge in this walk to change
                e_id, c = random.choice(sparse)

                # Try all possible values for this edge
                best_val = None
                best_new_viol = current_viol + 1

                old_val = volts[e_id]
                for trial_val in range(1, m):
                    if trial_val == old_val:
                        continue
                    # Compute new violation count efficiently
                    # Only need to recheck walks that involve e_id
                    new_viol = 0
                    for sparse2 in walk_sparse:
                        # Check if this walk involves e_id
                        involves = False
                        s = 0
                        for eid2, c_val2 in sparse2:
                            if eid2 == e_id:
                                involves = True
                                s += c_val2 * trial_val
                            else:
                                s += c_val2 * volts[eid2]
                        if s % m == 0:
                            new_viol += 1
                    if new_viol < best_new_viol:
                        best_new_viol = new_viol
                        best_val = trial_val

                if best_val is not None and best_new_viol < current_viol:
                    volts[e_id] = best_val
                    current_viol = best_new_viol
                else:
                    # Random move to escape local minimum
                    volts[e_id] = random.randint(1, m - 1)
                    current_viol = count_violations(volts, m)

            if current_viol == 0 and is_connected(volts, m):
                found = True
                break

        if found:
            best_n = candidate_n
            best_edges = build_lift_edges(volts, m)
            # Since we're going m upward, first valid is best for this approach
            break

    # Fallback to LUW if no valid lift found
    if best_edges is None:
        return _luw_graph(k, g)

    return best_n, best_edges


def _luw_graph(k, g):
    """Lazebnik-Ustimenko-Woldar graph as fallback."""
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
    index, order, edges_luw = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges_luw.append((index[vertex], index[other]))
    return len(order), edges_luw
